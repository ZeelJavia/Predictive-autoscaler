"""
STEP 2 — FastAPI LSTM Serving Layer
─────────────────────────────────────
Model: lstm_autoscaler.onnx
Input : (1, 12, 5) → [cpu%, ram_kb, disk_write_kbs, t_sin, t_cos]  |  12 x 5min = 1hr history
Output: (1, 3)     → [cpu_scaled, ram_scaled, diskw_scaled]

Run:
    pip install fastapi uvicorn onnxruntime numpy pydantic
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import math
import time
import logging
import datetime
from collections import defaultdict, deque
from typing import Optional
from threading import Thread

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# ─── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger("lstm-api")

# ─── CONFIG ───────────────────────────────────────────────────────
ONNX_MODEL_PATH = "lstm_autoscaler.onnx"
WINDOW_SIZE     = 12
N_FEATURES      = 5

# MinMaxScaler bounds from Bitbrains training
# Column order: [cpu%, ram_kb, disk_write_kbs, t_sin, t_cos]
FEAT_MIN = np.array([0.0,    0.0,        0.0,       -1.0, -1.0], dtype=np.float32)
FEAT_MAX = np.array([100.0,  67108864.0, 500000.0,   1.0,  1.0], dtype=np.float32)

# Scaling thresholds
CPU_UP  = 75.0
CPU_DN  = 40.0
RAM_UP     = 80.0
DISK_UP    = 80.0
HYSTERESIS = 5.0    # dead-band to prevent scale-down flapping

# ─── PROMETHEUS METRICS ───────────────────────────────────────────
_m = {
    "predictions_total"    : 0,
    "scale_up_total"       : 0,
    "scale_down_total"     : 0,
    "stable_total"         : 0,
    "prediction_latency_ms": 0.0,
}

def prometheus_output():
    return "\n".join([
        "# HELP lstm_predictions_total Total LSTM inferences",
        "# TYPE lstm_predictions_total counter",
        f"lstm_predictions_total {_m['predictions_total']}",
        "",
        "# HELP lstm_scale_up_total Scale-up recommendations",
        "# TYPE lstm_scale_up_total counter",
        f"lstm_scale_up_total {_m['scale_up_total']}",
        "",
        "# HELP lstm_scale_down_total Scale-down recommendations",
        "# TYPE lstm_scale_down_total counter",
        f"lstm_scale_down_total {_m['scale_down_total']}",
        "",
        "# HELP lstm_stable_total Stable recommendations",
        "# TYPE lstm_stable_total counter",
        f"lstm_stable_total {_m['stable_total']}",
        "",
        "# HELP lstm_inference_latency_ms Last inference latency ms",
        "# TYPE lstm_inference_latency_ms gauge",
        f"lstm_inference_latency_ms {_m['prediction_latency_ms']:.3f}",
        "",
        "# HELP lstm_active_windows Active rolling windows tracked",
        "# TYPE lstm_active_windows gauge",
        f"lstm_active_windows {len(windows)}",
    ])

# ─── LOAD ONNX ────────────────────────────────────────────────────
log.info(f"Loading ONNX model: {ONNX_MODEL_PATH}")
try:
    session     = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    log.info(f"Model ready | Input: {session.get_inputs()[0].shape} | Output: {session.get_outputs()[0].shape}")
except Exception as e:
    log.error(f"Failed to load ONNX: {e}")
    raise

# ─── ROLLING WINDOWS ──────────────────────────────────────────────
# One deque per vm_id — holds last 60 normalized 5-feature vectors
windows:   dict = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
last_seen: dict = {}                  # tracks last activity timestamp per vm_id
WINDOW_TTL_SECONDS = 7200            # evict vm_id window after 2 hours of silence (>1 full window)

def _cleanup_stale_windows():
    """Daemon thread: prevents unbounded memory growth from inactive vm_ids."""
    while True:
        time.sleep(60)
        now   = time.time()
        stale = [k for k, t in list(last_seen.items()) if now - t > WINDOW_TTL_SECONDS]
        for k in stale:
            windows.pop(k, None)
            last_seen.pop(k, None)
            log.info(f"Evicted stale window for vm_id={k}")

Thread(target=_cleanup_stale_windows, daemon=True).start()

# ─── APP ──────────────────────────────────────────────────────────
app = FastAPI(
    title="LSTM Predictive Autoscaler",
    version="2.0.0",
    description="Input: (12,5) cpu/ram/diskw/sin/cos → Output: (3,) predicted cpu/ram/diskw"
)

# ─── SCHEMAS ──────────────────────────────────────────────────────
class Metrics(BaseModel):
    vm_id         : str   = Field(default="default")
    cpu_percent   : float = Field(..., ge=0.0, le=100.0)
    ram_kb        : float = Field(..., ge=0.0)
    disk_write_kbs: float = Field(default=0.0, ge=0.0)
    timestamp     : Optional[float] = None

class Prediction(BaseModel):
    status              : str
    vm_id               : str
    window_fill         : str
    predicted_cpu_pct   : Optional[float] = None
    predicted_ram_mb    : Optional[float] = None
    predicted_diskw_kbs : Optional[float] = None
    scale_recommendation: Optional[str]   = None
    pressure_score      : Optional[float] = None
    inference_ms        : Optional[float] = None

# ─── HELPERS ──────────────────────────────────────────────────────
def normalize(cpu, ram_kb, disk_kbs, t_sin, t_cos):
    """Same transform as MinMaxScaler(0,1) used in vm_data_generator."""
    raw = np.array([cpu, ram_kb, disk_kbs, t_sin, t_cos], dtype=np.float32)
    return np.clip(
        (raw - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-8),
        0.0, 1.0
    )

def denormalize(pred):
    """Convert model output back to real units."""
    cpu_pct    = float(pred[0]) * 100.0
    ram_mb     = float(pred[1]) * FEAT_MAX[1] / 1024.0
    diskw_kbs  = float(pred[2]) * FEAT_MAX[2]
    return cpu_pct, ram_mb, diskw_kbs

def get_recommendation(cpu_pct, ram_mb, diskw_kbs):
    ram_pct  = (ram_mb * 1024.0 / FEAT_MAX[1]) * 100.0
    disk_pct = (diskw_kbs / FEAT_MAX[2]) * 100.0
    # Weighted composite pressure: CPU 60%, RAM 25%, Disk 15%
    pressure = 0.60 * cpu_pct + 0.25 * ram_pct + 0.15 * disk_pct

    if cpu_pct > CPU_UP or ram_pct > RAM_UP or disk_pct > DISK_UP or pressure > 70.0:
        _m["scale_up_total"] += 1
        return "SCALE_UP", pressure
    elif cpu_pct < (CPU_DN - HYSTERESIS) and ram_pct < 50.0 and pressure < 35.0:
        _m["scale_down_total"] += 1
        return "SCALE_DOWN", pressure
    _m["stable_total"] += 1
    return "STABLE", pressure

# ─── ENDPOINTS ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"      : "healthy",
        "model"       : ONNX_MODEL_PATH,
        "window_size" : WINDOW_SIZE,
        "input_shape" : f"(1, {WINDOW_SIZE}, {N_FEATURES})",
        "output_shape": "(1, 3)",
        "features"    : ["cpu%", "ram_kb", "disk_write_kbs", "t_sin", "t_cos"],
        "outputs"     : ["cpu%", "ram_kb", "disk_write_kbs"]
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_output()

@app.post("/predict_scale", response_model=Prediction)
async def predict_scale(data: Metrics):
    vm_id = data.vm_id
    ts    = data.timestamp or time.time()

    # Time encoding
    dt     = datetime.datetime.fromtimestamp(ts)
    mofday = dt.hour * 60 + dt.minute
    t_sin  = math.sin(2 * math.pi * mofday / 1440.0)
    t_cos  = math.cos(2 * math.pi * mofday / 1440.0)

    # Normalize + append to rolling window
    scaled = normalize(data.cpu_percent, data.ram_kb, data.disk_write_kbs, t_sin, t_cos)
    windows[vm_id].append(scaled.tolist())
    last_seen[vm_id] = time.time()    # update TTL timestamp

    fill = f"{len(windows[vm_id])}/{WINDOW_SIZE}"

    # Not ready yet
    if len(windows[vm_id]) < WINDOW_SIZE:
        return Prediction(status="warming_up", vm_id=vm_id, window_fill=fill)

    # LSTM inference — shape (1, 60, 5)
    t0         = time.perf_counter()
    inp        = np.array([list(windows[vm_id])], dtype=np.float32)
    pred       = session.run([output_name], {input_name: inp})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    _m["predictions_total"]     += 1
    _m["prediction_latency_ms"]  = latency_ms

    cpu_pct, ram_mb, diskw_kbs = denormalize(pred[0])
    action, pressure            = get_recommendation(cpu_pct, ram_mb, diskw_kbs)

    log.info(
        f"[{vm_id}] CPU:{cpu_pct:.1f}% RAM:{ram_mb:.0f}MB "
        f"DiskW:{diskw_kbs:.0f}KB/s → {action} ({latency_ms:.1f}ms)"
    )

    return Prediction(
        status               = "success",
        vm_id                = vm_id,
        window_fill          = fill,
        predicted_cpu_pct    = round(cpu_pct, 2),
        predicted_ram_mb     = round(ram_mb, 2),
        predicted_diskw_kbs  = round(diskw_kbs, 2),
        scale_recommendation = action,
        pressure_score       = round(pressure, 2),
        inference_ms         = round(latency_ms, 3)
    )

@app.get("/window/{vm_id}")
def window_status(vm_id: str):
    n = len(windows.get(vm_id, []))
    return {
        "vm_id"   : vm_id,
        "filled"  : n,
        "required": WINDOW_SIZE,
        "ready"   : n >= WINDOW_SIZE
    }
