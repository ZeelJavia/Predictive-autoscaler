"""
STEP 4 — LSTM Consumer + Kubernetes Autoscaler
────────────────────────────────────────────────
Reads metrics from Kafka → builds rolling window (12 × 5-min readings = 1 hour)
→ runs LSTM ONNX inference → makes scale decision → patches Minikube deployment.

Feature order (must match vm_data_generator in training):
  [cpu%, ram_kb, disk_write_kbs, t_sin, t_cos]

Output order (must match y = scaled_data[i+WINDOW_SIZE, 0:3]):
  [cpu%, ram_kb, disk_write_kbs]  — all in 0-1 scale

Setup:
    pip install kafka-python redis onnxruntime numpy kubernetes
    Make sure Kafka + Redis are running: docker-compose up -d
"""

import json
import math
import time
import logging
import numpy as np
import redis
import onnxruntime as ort
from kafka import KafkaConsumer
from datetime import datetime
from kubernetes import client, config

# ─── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger("consumer")

# ─── CONFIG ───────────────────────────────────────────────────────
KAFKA_TOPIC       = 'vm-metrics'
KAFKA_BROKERS     = ['localhost:9092']
REDIS_HOST        = 'localhost'
REDIS_PORT        = 6379
WINDOW_SIZE       = 12
ONNX_MODEL_PATH   = "lstm_autoscaler.onnx"   # place in same folder
TARGET_DEPLOYMENT = 'cpu-burner'
TARGET_NAMESPACE  = 'default'

# Scaling policy
MAX_PODS          = 10
MIN_PODS          = 2
COOLDOWN_SECONDS  = 120    # 2 min cooldown between scale events
SCALE_UP_STEP     = 2      # add N pods per scale-up  (avoids doubling overshoot)
SCALE_DOWN_STEP   = 1      # remove N pods per scale-down (gradual, safe)
CPU_UP_THRESH     = 75.0
CPU_DN_THRESH     = 40.0
RAM_UP_THRESH     = 80.0
DISK_UP_THRESH    = 80.0
HYSTERESIS        = 5.0    # dead-band to prevent flip-flopping

# MinMaxScaler bounds — column order: [cpu%, ram_kb, diskw, t_sin, t_cos]
# These must match the scaler fitted during training on Bitbrains data
FEAT_MIN = np.array([0.0,    0.0,        0.0,       -1.0, -1.0], dtype=np.float32)
FEAT_MAX = np.array([100.0,  67108864.0, 500000.0,   1.0,  1.0], dtype=np.float32)

# ─── HELPERS ──────────────────────────────────────────────────────
def normalize(cpu, ram_kb, disk_kbs, t_sin, t_cos):
    """Replicate MinMaxScaler(0,1) from training."""
    raw = np.array([cpu, ram_kb, disk_kbs, t_sin, t_cos], dtype=np.float32)
    return np.clip((raw - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-8), 0.0, 1.0).tolist()

def denormalize(pred):
    """Convert (0-1) model output → real units."""
    cpu_pct    = float(pred[0]) * 100.0
    ram_kb     = float(pred[1]) * FEAT_MAX[1]
    diskw_kbs  = float(pred[2]) * FEAT_MAX[2]
    return cpu_pct, ram_kb, diskw_kbs

def time_features(ts):
    dt     = datetime.fromtimestamp(ts)
    mofday = dt.hour * 60 + dt.minute
    return (
        math.sin(2 * math.pi * mofday / 1440.0),
        math.cos(2 * math.pi * mofday / 1440.0)
    )

# ─── CONNECT INFRASTRUCTURE ───────────────────────────────────────
log.info("Connecting to infrastructure...")
try:
    # Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r.ping()
    log.info("Redis connected.")

    # Kafka
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        auto_offset_reset='latest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    log.info("Kafka connected.")

    # ONNX
    session     = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    log.info(f"LSTM ONNX ready | Input:{session.get_inputs()[0].shape} Output:{session.get_outputs()[0].shape}")

except Exception as e:
    log.error(f"Init failed: {e}")
    exit(1)

# ─── KUBERNETES ───────────────────────────────────────────────────
log.info("Connecting to Kubernetes (Minikube)...")
try:
    config.load_kube_config()
    k8s_apps = client.AppsV1Api()
    log.info("Kubernetes connected.")
    DRY_RUN = False
except Exception as e:
    log.warning(f"Kubernetes unavailable: {e}")
    log.warning("Running in DRY RUN mode — scaling commands will be logged only.")
    k8s_apps = None
    DRY_RUN  = True

def get_replicas():
    if DRY_RUN: return 2
    try:
        dep = k8s_apps.read_namespaced_deployment(TARGET_DEPLOYMENT, TARGET_NAMESPACE)
        return dep.spec.replicas or 1
    except Exception as e:
        log.warning(f"Could not read replicas: {e}")
        return 2

def scale_to(replicas):
    if DRY_RUN:
        log.info(f"[DRY RUN] Scale {TARGET_DEPLOYMENT} → {replicas} pods")
        return True
    try:
        dep = k8s_apps.read_namespaced_deployment(TARGET_DEPLOYMENT, TARGET_NAMESPACE)
        dep.spec.replicas = replicas
        k8s_apps.patch_namespaced_deployment(TARGET_DEPLOYMENT, TARGET_NAMESPACE, dep)
        log.info(f"SCALED: {TARGET_DEPLOYMENT} → {replicas} pods")
        return True
    except Exception as e:
        log.error(f"Scale failed: {e}")
        return False

# ─── SCALING ENGINE ───────────────────────────────────────────────
last_scale_time = 0

def make_decision(pred_cpu, pred_ram_kb, pred_disk_kbs, redis_key):
    global last_scale_time

    pred_ram_pct  = min((pred_ram_kb  / FEAT_MAX[1]) * 100.0, 100.0)
    pred_disk_pct = min((pred_disk_kbs / FEAT_MAX[2]) * 100.0, 100.0)

    # Weighted composite pressure score
    pressure = (0.60 * pred_cpu) + (0.25 * pred_ram_pct) + (0.15 * pred_disk_pct)

    log.info(
        f"Predicted → CPU:{pred_cpu:.1f}%  "
        f"RAM:{pred_ram_kb/1024:.0f}MB  "
        f"DiskW:{pred_disk_kbs:.0f}KB/s  "
        f"Pressure:{pressure:.1f}"
    )

    # Cooldown check
    elapsed = time.time() - last_scale_time
    if elapsed < COOLDOWN_SECONDS:
        log.info(f"COOLDOWN: {int(COOLDOWN_SECONDS - elapsed)}s remaining — no action")
        return

    current = get_replicas()

    # ── SCALE UP ──────────────────────────────────────────────────
    if (pred_cpu       > CPU_UP_THRESH  or
        pred_ram_pct   > RAM_UP_THRESH  or
        pred_disk_pct  > DISK_UP_THRESH or
        pressure       > 70.0):

        if current >= MAX_PODS:
            log.warning(f"AT MAX CAPACITY ({MAX_PODS} pods) — cannot scale up")
            return
        new = min(current + SCALE_UP_STEP, MAX_PODS)

        # Veto if post-scale load is already too low — avoids unnecessary scale-up
        load_after_up = pred_cpu * (current / new) if new > 0 else 0
        if load_after_up < CPU_DN_THRESH:
            log.warning(f"SCALE UP VETOED: post-scale CPU ~{load_after_up:.1f}% — load too low")
            return

        log.info(f"SCALE UP: {current} → {new} pods")
        if scale_to(new):
            r.delete(redis_key)           # flush stale pre-scale window
            last_scale_time = time.time()

    # ── SCALE DOWN ────────────────────────────────────────────────
    elif (pred_cpu      < (CPU_DN_THRESH - HYSTERESIS) and
          pred_ram_pct  < 50.0 and
          pressure      < 35.0):

        if current <= MIN_PODS:
            log.info(f"STABLE at minimum ({MIN_PODS} pods) — no action")
            return
        new = max(current - SCALE_DOWN_STEP, MIN_PODS)

        # Anti-thrash: verify post-scale CPU won't spike back
        load_after = pred_cpu * (current / new)
        if load_after > (CPU_UP_THRESH - HYSTERESIS):
            log.warning(f"SCALE DOWN VETOED: post-scale CPU ~{load_after:.1f}% — too risky")
            return

        log.info(f"SCALE DOWN: {current} → {new} pods (post-scale CPU ~{load_after:.1f}%)")
        if scale_to(new):
            r.delete(redis_key)           # flush stale pre-scale window
            last_scale_time = time.time()

    else:
        log.info(f"STABLE — no action (pressure={pressure:.1f})")

# ─── MAIN LOOP ────────────────────────────────────────────────────
log.info("Consumer running — waiting for metrics from Kafka...")
log.info(f"Model: {ONNX_MODEL_PATH} | Window: {WINDOW_SIZE} samples x 5min = {WINDOW_SIZE*5}min history | Features: 5 | Outputs: 3")

RECONNECT_DELAY = 10   # seconds to wait before reconnecting to Kafka

running = True
while running:
    try:
        for message in consumer:
            data  = message.value
            vm_id = data.get('vm_id', 'default')

            # Extract raw metrics from Kafka payload
            cpu_pct        = float(data.get('cpu_percent', 0.0))
            ram_kb         = float(data.get('ram_kb', 0.0))
            disk_write_kbs = float(data.get('disk_write_kbs', 0.0))
            timestamp      = float(data.get('timestamp', time.time()))

            # Time features
            t_sin, t_cos = time_features(timestamp)

            # Normalize — 5 features, same order as training column_stack
            scaled = normalize(cpu_pct, ram_kb, disk_write_kbs, t_sin, t_cos)

            # Redis rolling window — pipeline batches 3 ops into 1 round-trip
            redis_key = f"history:{vm_id}"
            pipe = r.pipeline()
            pipe.lpush(redis_key, json.dumps(scaled))
            pipe.ltrim(redis_key, 0, WINDOW_SIZE - 1)
            pipe.llen(redis_key)
            _, _, wlen = pipe.execute()

            log.info(
                f"[{vm_id}] CPU:{cpu_pct:.1f}%  "
                f"RAM:{ram_kb/1024:.0f}MB  "
                f"DiskW:{disk_write_kbs:.0f}KB/s  "
                f"Window:{wlen}/{WINDOW_SIZE}"
            )

            # Only predict when window is full (60 samples)
            if wlen == WINDOW_SIZE:
                raw    = r.lrange(redis_key, 0, -1)
                window = [json.loads(x) for x in reversed(raw)]   # oldest first

                # Shape: (1, 60, 5)
                inp  = np.array([window], dtype=np.float32)
                pred = session.run([output_name], {input_name: inp})[0]

                pred_cpu, pred_ram_kb, pred_disk = denormalize(pred[0])

                log.info("─" * 55)
                log.info(f"LSTM PREDICTION for [{vm_id}]:")
                log.info(f"  CPU       : {pred_cpu:.2f}%")
                log.info(f"  RAM       : {pred_ram_kb/1024:.1f} MB")
                log.info(f"  Disk Write: {pred_disk:.0f} KB/s")

                make_decision(pred_cpu, pred_ram_kb, pred_disk, redis_key)
                log.info("─" * 55)

    except KeyboardInterrupt:
        log.info("Consumer stopped by user.")
        running = False
    except Exception as e:
        log.error(f"Kafka error: {e} — reconnecting in {RECONNECT_DELAY}s")
        time.sleep(RECONNECT_DELAY)
        try:
            consumer.close()
        except Exception:
            pass
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKERS,
            auto_offset_reset='latest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        log.info("Kafka reconnected — resuming.")

consumer.close()
log.info("Done.")

