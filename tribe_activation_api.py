"""
TRIBE v2 — Activation API
=========================

API de un solo endpoint que recibe un video, lo procesa con TRIBE v2 y
devuelve un "número de activación" cerebral por cada lapso de tiempo.

Endpoint:
    POST /activation

Notas importantes de diseño:
  - El score se calcula en la RESOLUCIÓN NATIVA del modelo (una fila por TR).
    Esa es la única resolución temporal REAL (la señal BOLD es lenta, ~0.5-1 Hz).
  - Las rejillas 'second' / 'frame' / 'ms' se obtienen por INTERPOLACIÓN de la
    curva nativa. Se devuelven, pero marcadas como interpoladas: no añaden
    información, solo re-muestrean.

Requisitos:
    pip install fastapi "uvicorn[standard]" python-multipart numpy scipy
    # + el paquete tribev2 instalado (pip install -e . del repo)
    # + ffmpeg en el sistema

Arranque:
    uvicorn tribe_activation_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------

MODEL_NAME = os.environ.get("TRIBE_MODEL", "facebook/tribev2")
CACHE_FOLDER = os.environ.get("TRIBE_CACHE", "./cache")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 500 * 1024 * 1024))  # 500 MB
# TR de respaldo (segundos por fila) si NO se pueden leer timestamps de `segments`.
# Confirmado: TR = 1.49 s en el estudio algonauts2025
# (tribev2/studies/algonauts2025.py: "TR = 1.49 seconds", _FREQUENCY = 1/1.49).
FALLBACK_TR_SECONDS = float(os.environ.get("FALLBACK_TR", 1.49))

# Un único trabajo de inferencia a la vez por GPU.
_gpu_semaphore = asyncio.Semaphore(1)

# Handle global del modelo (se carga una sola vez en el startup).
_model = None


# ----------------------------------------------------------------------------
# Esquemas de entrada/salida
# ----------------------------------------------------------------------------

class ScoreMethod(str, Enum):
    rms = "rms"          # A(t) = sqrt(mean_v P[t,v]^2)   (basal ~1)
    abs = "abs"          # A(t) = mean_v |P[t,v]|         (basal ~0.798)
    threshold = "thr"    # A(t) = fraccion de vertices con |z| >= theta


class Granularity(str, Enum):
    tr = "tr"            # resolucion NATIVA (real)
    second = "second"    # interpolado a 1 Hz
    frame = "frame"      # interpolado a fps del video
    ms = "ms"            # interpolado a 1000 Hz (muy sobre-muestreado)


class Normalization(str, Enum):
    none = "none"
    zscore = "zscore"    # (A - mu)/sigma sobre el tiempo
    minmax = "minmax"    # a [0, 1]


class ActivationPoint(BaseModel):
    t_seconds: float
    score: float


class ActivationResponse(BaseModel):
    method: str
    granularity: str
    normalization: str
    interpolated: bool
    n_native_points: int
    n_returned_points: int
    tr_seconds: Optional[float]
    note: str
    series: list[ActivationPoint]


# ----------------------------------------------------------------------------
# Núcleo matemático: de la matriz P (T x V) a la curva A(t)
# ----------------------------------------------------------------------------

def compute_activation_curve(
    preds: np.ndarray,
    method: ScoreMethod = ScoreMethod.rms,
    theta: float = 1.96,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    preds: array (T, V) de respuestas BOLD predichas (z-scored por vertice).
    Devuelve A: array (T,) con un score por fila temporal.
    """
    P = np.asarray(preds, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"Se esperaba preds 2D (T, V), llego {P.shape}")

    if roi_mask is not None:
        P = P[:, roi_mask]
        if P.shape[1] == 0:
            raise ValueError("La mascara de ROI no selecciono ningun vertice")

    if method is ScoreMethod.rms:
        # A(t) = sqrt( (1/V) sum_v P[t,v]^2 )
        A = np.sqrt(np.mean(np.square(P), axis=1))
    elif method is ScoreMethod.abs:
        # A(t) = (1/V) sum_v |P[t,v]|
        A = np.mean(np.abs(P), axis=1)
    elif method is ScoreMethod.threshold:
        # A(t) = (1/V) sum_v 1[ |P[t,v]| >= theta ]
        A = np.mean((np.abs(P) >= theta).astype(np.float64), axis=1)
    else:
        raise ValueError(f"Metodo desconocido: {method}")

    return A


def normalize_curve(A: np.ndarray, mode: Normalization) -> np.ndarray:
    if mode is Normalization.none:
        return A
    if mode is Normalization.zscore:
        mu, sigma = A.mean(), A.std()
        return (A - mu) / sigma if sigma > 0 else A - mu
    if mode is Normalization.minmax:
        lo, hi = A.min(), A.max()
        return (A - lo) / (hi - lo) if hi > lo else np.zeros_like(A)
    raise ValueError(f"Normalizacion desconocida: {mode}")


def extract_timestamps(segments, n_rows: int, tr: float) -> np.ndarray:
    """
    Marca de tiempo (segundos) representativa de cada fila de `preds`.

    `model.predict()` (tribev2) devuelve `segments` como una LISTA de
    objetos-segmento, uno por fila, creados con
    `segment.copy(offset=t, duration=TR)` para t = 0, TR, 2*TR, ...
    Cada objeto expone `.start` (inicio en la linea de tiempo del estimulo),
    `.offset` (cursor dentro del segmento base) y `.duration` (= TR). El punto
    medio real del intervalo es  start + offset + duration/2.

    VERIFICAR UNA VEZ en tu instalacion (imprime `vars(segments[0])`):
    si en tu version de neuralset el `.start` del objeto copiado YA incorpora
    el `.offset`, elimina el sumando `offset` para no contarlo dos veces
    (sintoma de doble conteo: timestamps que avanzan ~2*TR en vez de ~TR).

    Orden de intento: (1) objetos-segmento; (2) array numerico (inicio,fin) o
    (n,) de timestamps; (3) respaldo: rejilla regular t = i * TR.
    """
    # 1) Caso real de tribev2: lista de objetos-segmento.
    try:
        ts = []
        for s in segments:
            start = getattr(s, "start", None)
            if start is None:
                raise AttributeError("segmento sin atributo .start")
            offset = float(getattr(s, "offset", 0.0) or 0.0)
            duration = float(getattr(s, "duration", tr) or tr)
            ts.append(float(start) + offset + duration / 2.0)
        if len(ts) == n_rows:
            return np.asarray(ts, dtype=np.float64)
    except (TypeError, AttributeError, ValueError):
        pass

    # 2) Caso numerico: array (n,2) [inicio,fin] o (n,) de timestamps.
    try:
        seg = np.asarray(segments, dtype=np.float64)
        if seg.ndim == 2 and seg.shape[0] == n_rows and seg.shape[1] >= 2:
            return (seg[:, 0] + seg[:, 1]) / 2.0       # punto medio del intervalo
        if seg.ndim == 1 and seg.shape[0] == n_rows:
            return seg                                  # ya son timestamps
    except (TypeError, ValueError):
        pass

    # 3) Respaldo: rejilla regular t = i * TR.
    return np.arange(n_rows, dtype=np.float64) * tr


def resample_curve(
    t_native: np.ndarray,
    A_native: np.ndarray,
    granularity: Granularity,
    fps: Optional[float],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Interpola A(t) a la rejilla pedida. Devuelve (t_grid, A_grid, interpolado?).
    """
    if granularity is Granularity.tr:
        return t_native, A_native, False

    t0, t1 = float(t_native[0]), float(t_native[-1])
    if granularity is Granularity.second:
        t_grid = np.arange(np.floor(t0), np.floor(t1) + 1, 1.0)
    elif granularity is Granularity.frame:
        if not fps or fps <= 0:
            raise ValueError("granularity=frame requiere 'fps' > 0")
        t_grid = np.arange(int(t0 * fps), int(t1 * fps) + 1) / fps
    elif granularity is Granularity.ms:
        t_grid = np.arange(int(t0 * 1000), int(t1 * 1000) + 1) / 1000.0
    else:
        raise ValueError(f"Granularidad desconocida: {granularity}")

    # Interpolacion lineal; usa kind='cubic' si quieres una curva mas suave.
    f = interp1d(t_native, A_native, kind="linear",
                 bounds_error=False, fill_value=(A_native[0], A_native[-1]))
    return t_grid, f(t_grid), True


# ----------------------------------------------------------------------------
# Pipeline de inferencia (bloqueante; se corre en un threadpool)
# ----------------------------------------------------------------------------

def run_inference(video_path: str):
    """Devuelve (preds, segments). Encapsula la llamada a TRIBE v2."""
    df = _model.get_events_dataframe(video_path=video_path)
    preds, segments = _model.predict(events=df)
    return np.asarray(preds), segments


# ----------------------------------------------------------------------------
# Ciclo de vida: cargar el modelo una vez
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from tribev2 import TribeModel  # import diferido para no fallar sin GPU
    _model = TribeModel.from_pretrained(MODEL_NAME, cache_folder=CACHE_FOLDER)
    yield
    _model = None


app = FastAPI(title="TRIBE v2 Activation API", version="1.0", lifespan=lifespan)


# ----------------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------------

@app.post("/activation", response_model=ActivationResponse)
async def activation(
    video: UploadFile = File(..., description="Archivo de video con audio"),
    method: ScoreMethod = Form(ScoreMethod.rms),
    granularity: Granularity = Form(Granularity.tr),
    normalization: Normalization = Form(Normalization.none),
    theta: float = Form(1.96),
    fps: Optional[float] = Form(None, description="Requerido si granularity=frame"),
):
    # --- Validacion basica ---
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(415, f"Tipo no soportado: {video.content_type}")

    data = await video.read()
    if len(data) == 0:
        raise HTTPException(400, "Archivo vacio")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Video demasiado grande")

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # --- Inferencia (1 a la vez por GPU, fuera del event loop) ---
        async with _gpu_semaphore:
            preds, segments = await asyncio.to_thread(run_inference, tmp_path)

        # --- Score nativo ---
        A_native = compute_activation_curve(preds, method=method, theta=theta)
        A_native = normalize_curve(A_native, normalization)
        t_native = extract_timestamps(segments, len(A_native), FALLBACK_TR_SECONDS)

        # --- Re-muestreo a la rejilla pedida ---
        t_grid, A_grid, interpolated = resample_curve(
            t_native, A_native, granularity, fps
        )

        series = [
            ActivationPoint(t_seconds=round(float(t), 4), score=round(float(a), 6))
            for t, a in zip(t_grid, A_grid)
        ]

        note = (
            "Resolucion nativa (real, una muestra por TR)."
            if not interpolated
            else "Valores INTERPOLADos de la curva nativa; la resolucion real "
                 "esta acotada por el TR (la senal BOLD es lenta)."
        )

        return ActivationResponse(
            method=method.value,
            granularity=granularity.value,
            normalization=normalization.value,
            interpolated=interpolated,
            n_native_points=len(A_native),
            n_returned_points=len(series),
            tr_seconds=FALLBACK_TR_SECONDS,
            note=note,
            series=series,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}
