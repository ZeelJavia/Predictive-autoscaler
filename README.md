<div align="center">

# 🚀 LSTM Predictive Autoscaler

### *Scale before the spike hits — not after*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green?logo=onnx)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![Kafka](https://img.shields.io/badge/Apache-Kafka-black?logo=apachekafka)](https://kafka.apache.org)
[![Redis](https://img.shields.io/badge/Redis-7.0-red?logo=redis)](https://redis.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-grade, **AI-powered cloud autoscaler** that uses a stacked LSTM neural network to predict future CPU, RAM, and Disk I/O usage and proactively scale compute replicas — eliminating the cold-start delay that plagues traditional threshold-based autoscalers.

[📖 Deep Explanation](docs/PROJECT_DEEP_EXPLANATION.md) · [🚀 Quick Start](#quick-start) · [🏗️ Architecture](#architecture) · [📊 Results](#results)

</div>

---

## The Problem with Reactive Autoscaling

Traditional autoscalers (AWS HPA, CloudWatch) react **after** the spike:

```
[Traffic Spike] ──► [CPU > 80% detected] ──► [Scale triggered] ──► [Pod ready in 60s]
                                                                          ▲
                                                       Users experience degradation HERE
```

This project solves it with **prediction**:

```
[Rolling window of metrics] ──► [LSTM predicts load] ──► [Scale NOW] ──► [Pod ready BEFORE spike]
```

---

## Architecture

### System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TRAINING PHASE  (Google Colab / GPU)                 ║
║                                                                              ║
║   GWA-Bitbrains Dataset (1,250 real VM traces)                               ║
║           │                                                                  ║
║           ▼                                                                  ║
║   vm_data_generator()  ◄── sliding window  ◄── MinMaxScaler (per-VM)        ║
║           │                                                                  ║
║           ▼                                                                  ║
║   ┌────────────────────────────────────────────────────┐                    ║
║   │  Input (12 timesteps × 5 features)                  │                    ║
║   │     [cpu%,  ram_kb,  disk_kbs,  t_sin,  t_cos]     │                    ║
║   │                    ↓                               │                    ║
║   │         LSTM(128, return_sequences=True)            │                    ║
║   │                  + Dropout(0.2)                     │                    ║
║   │                    ↓                               │                    ║
║   │         LSTM(64, return_sequences=False)            │                    ║
║   │                  + Dropout(0.2)                     │                    ║
║   │                    ↓                               │                    ║
║   │              Dense(32, ReLU)                        │                    ║
║   │                    ↓                               │                    ║
║   │          Dense(3, Linear)  ◄── Huber Loss           │                    ║
║   │   [pred_cpu%,  pred_ram_kb,  pred_disk_kbs]        │                    ║
║   └────────────────────────────────────────────────────┘                    ║
║           │                                                                  ║
║           ▼                                                                  ║
║   lstm_epoch_N.keras  ──► export_to_onnx.py  ──► lstm_autoscaler.onnx      ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                    ┌────────────────┘
                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRODUCTION INFERENCE PIPELINE                             ║
║                                                                              ║
║  ┌─────────────────┐    ┌──────────────┐                                   ║
║  │  cpu-burner app │◄───│ Locust Load  │                                   ║
║  │  (Flask, 2+     │    │  Test        │                                   ║
║  │   pods)         │    │ (/compute)   │                                   ║
║  └────────┬────────┘    └──────────────┘                                   ║
║           │                                                                  ║
║           │ pod metrics (cpu%, ram, disk) every 5 min                        ║
║           ▼                                                                  ║
║  ┌─────────────────┐                        ┌──────────────────────────┐   ║
║  │  pipeline/      │                        │  serving/main.py         │   ║
║  │  producer.py    │──POST /predict_scale──►│  FastAPI  +  ONNX model  │   ║
║  │  (every 5 min)  │                        │  Rolling window per VM   │   ║
║  └────────┬────────┘                        └──────────────────────────┘   ║
║           │                                                                  ║
║           │  Kafka topic: 'vm-metrics'                                       ║
║           ▼                                                                  ║
║  ┌─────────────────┐     ┌──────────────┐                                   ║
║  │  Apache Kafka   │────►│  pipeline/   │                                   ║
║  │  (message       │     │  consumer.py │                                   ║
║  │   broker)       │     │             │◄──► Redis (rolling window)         ║
║  └─────────────────┘     │             │                                    ║
║                          │  ONNX Infer │                                    ║
║  ┌─────────────────┐     │     │       │                                    ║
║  │  Redis          │◄───►│  Pressure   │                                    ║
║  │  (window store) │     │  Score      │                                    ║
║  └─────────────────┘     │     │       │                                    ║
║                          │     ▼       │                                    ║
║                          │  SCALE UP / │                                    ║
║                          │  SCALE DOWN │                                    ║
║                          │  STABLE     │                                    ║
║                          └──────┬──────┘                                   ║
║                                 │ patch replicas                            ║
║                                 ▼                                           ║
║                    ┌─────────────────────────┐                              ║
║                    │  Container Orchestrator  │                              ║
║                    │  (2 → 4 → 2 pods)        │                              ║
║                    └─────────────────────────┘                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Decision Engine Flow

```
Every metric received (every 5min):
│
├─► Normalize features  ──►  Push to Redis rolling window
│
└─► If window full (12 readings = 1 hour of history):
    │
    ├─► LSTM ONNX inference  (< 1ms)
    │       Input:  (1, 12, 5)
    │       Output: [pred_cpu%, pred_ram_kb, pred_diskw_kbs]
    │
    ├─► Compute pressure score:
    │       pressure = 0.60×cpu + 0.25×ram + 0.15×disk
    │
    ├─► SCALE UP   if  cpu > 75%  OR  ram > 80%  OR  pressure > 70
    │       └─► Veto if post-scale load < scale-down threshold
    │
    ├─► SCALE DOWN if  cpu < 35%  AND  ram < 50%  AND  pressure < 35
    │       └─► Veto if post-scale CPU would spike back up
    │
    ├─► STABLE     otherwise (hysteresis dead-band: ±5%)
    │
    └─► Cooldown: 120s between any two scaling events
```

---

## Project Structure

```
lstm-predictive-autoscaler/
│
├── 📓 notebooks/
│   └── EC2_computation_prediction.ipynb   # Full training pipeline (run in Colab)
│
├── 🧠 serving/
│   ├── main.py                            # FastAPI inference API  (uvicorn main:app)
│   ├── app.py                             # CPU-burner Flask app   (target workload)
│   └── export_to_onnx.py                  # Keras → ONNX conversion (run in Colab)
│
├── 🔄 pipeline/
│   ├── producer.py                        # Scrapes pod metrics → Kafka
│   └── consumer.py                        # Kafka → Redis → ONNX → scale decision
│
├── 🏗️ infra/
│   ├── docker-compose.yml                 # Kafka + Zookeeper + Redis + Prometheus + Grafana
│   ├── deployment.yaml                    # Container orchestration manifests
│   ├── prometheus.yml                     # Prometheus scrape config
│   └── Dockerfile.cpuburner              # Image for cpu-burner app
│
│
├── 🔫 load_test/
│   └── locustfile.py                      # Locust load test (NormalUser + SpikeUser)
│
├── 📦 model/
│   └── lstm_autoscaler.onnx               # Trained model (490KB, CPU-inference)
│
├── 📄 docs/
│   ├── PROJECT_DEEP_EXPLANATION.md        # Full cell-by-cell technical deep dive
│   └── HOW_TO_RUN.md                      # Step-by-step run guide
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Docker Desktop (for Kafka + Redis + Prometheus)
# Minikube (for local container orchestration — local testing only)
minikube start --cpus=4 --memory=4096
minikube addons enable metrics-server
```

### Step 1 — Start Infrastructure

```bash
cd infra/
docker-compose up -d

# Verify: all services should be Up
docker-compose ps
```

### Step 2 — Deploy Target App

```bash
# Build the CPU-burner image inside Minikube
minikube docker-env --shell powershell | Invoke-Expression
docker build -t cpu-burner-app:v2 -f infra/Dockerfile.cpuburner .

# Deploy
kubectl apply -f infra/deployment.yaml
kubectl get pods -w
# Wait until 2/2 pods show Running
```

### Step 3 — Start FastAPI Serving Layer

```bash
# Place lstm_autoscaler.onnx in the same directory, then:
cd serving/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
curl http://localhost:8000/health
# {"status":"healthy","model":"lstm_autoscaler.onnx","window_size":12,...}
```

### Step 4 — Start Producer (Terminal 2)

```bash
cd pipeline/
python producer.py

# Output every 5 minutes:
# Pods:2 | CPU:23.4% | RAM:45MB | DiskW:0KB/s
```

### Step 5 — Start Consumer / Autoscaler (Terminal 3)

```bash
cd pipeline/
python consumer.py

# After 12 messages (1 hour = 12 x 5-min readings), you'll see:
# LSTM PREDICTION for [cpu-burner-cluster]:
#   CPU       : 27.34%
#   RAM       : 1823.1 MB
#   Disk Write: 320 KB/s
# STABLE — no action (pressure=17.4)
```

### Step 6 — Trigger a Spike (Terminal 4)

```bash
cd load_test/
locust -f locustfile.py SpikeUser --users 100 --spawn-rate 20 \
  --headless --run-time 3m \
  --host http://$(minikube ip):30000

# Watch consumer terminal:
# SCALE UP: 2 → 4 pods
```

---

## Model Architecture

```
Input: (batch, 12, 5)
  └── [cpu%, ram_kb, disk_write_kbs, time_sin, time_cos]

LSTM(128, return_sequences=True)   →  (batch, 12, 128)   params: 68,608
Dropout(0.2)
LSTM(64,  return_sequences=False)  →  (batch, 64)         params: 49,408
Dropout(0.2)
Dense(32, activation='relu')       →  (batch, 32)         params:  2,080
Dense(3,  activation='linear')     →  (batch, 3)          params:     99
                                                          ───────────────
Output: [pred_cpu_scaled, pred_ram_scaled, pred_disk_scaled]  Total: 120,195
```

| Component | Choice | Justification |
|---|---|---|
| **Architecture** | Stacked LSTM | Sequential, multi-variate time-series — LSTM's core strength |
| **Loss** | Huber (δ=1.0) | Robust to server metric anomalies (spikes) unlike MSE |
| **Time encoding** | sin/cos | Cyclical representation — 11:59pm adjacent to midnight |
| **Scaler** | MinMax per-VM | Each VM has its own profile; relative changes matter |
| **Window** | 12 timesteps | 1 hr history at 5-min intervals — captures ramp-ups and daily patterns |
| **Output format** | ONNX | 490KB, no TensorFlow runtime, <1ms CPU inference |

---

## Dataset

**GWA-Bitbrains** — Real telemetry from 1,250 production VMs (Bitbrains datacenter, August 2013)

| Split | VMs | Samples |
|---|---|---|
| Training | 1,000 | ~8.8 million sliding windows |
| Validation | 250 (held out) | ~2.3 million sliding windows |

Validation uses **entirely unseen VM profiles** — true out-of-distribution evaluation.

---

## Results

### Model Performance — Final Epoch

| Metric | Train | Validation |
|---|---|---|
| Accuracy | 88.83% | **92.87%** |
| MAE  | 0.0430 | 0.0314 |
| Huber Loss | 0.0033 | 0.0022 |

> `val_accuracy > train_accuracy` — model **generalizes** to entirely unseen VM profiles

### Inference Performance (CPU-only, ONNXRuntime)

| Metric | Value |
|---|---|
| **Average latency** | **0.18 ms** |
| p95 | 0.30 ms |
| p99 | 3.10 ms |
| Model size | **490 KB** |

---

## Scaling Policy

| Parameter | Value | Reason |
|---|---|---|
| **Scrape interval** | 5 minutes (300s) | Matches Bitbrains 5-min timestamp resolution |
| `MIN_PODS` | 2 | Always maintain HA baseline |
| `MAX_PODS` | 10 | Cost guard — prevent runaway scaling |
| `SCALE_UP_STEP` | +2 pods | Avoid overshoot from single-pod increments |
| `SCALE_DOWN_STEP` | -1 pod | Conservative — one pod at a time |
| `COOLDOWN_SECONDS` | 120s | Pods need time to start and stabilize |
| `HYSTERESIS` | 5% | Dead-band prevents flip-flopping at threshold |
| `CPU_UP_THRESH` | 75% | Scale up before saturation |
| `CPU_DN_THRESH` | 40% | Scale down only when genuinely underloaded |
| `Pressure weights` | 60/25/15 | CPU dominant, then RAM, then Disk |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Training** | TensorFlow 2.x, Keras, Google Colab T4 GPU |
| **Model Format** | ONNX (via tf2onnx), ONNXRuntime |
| **API Layer** | FastAPI, Uvicorn, Pydantic |
| **Message Broker** | Apache Kafka + Zookeeper |
| **State Store** | Redis 7 (rolling window buffer) |
| **Observability** | Prometheus |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Container replica scaling (Minikube for local testing) |
| **Load Testing** | Locust |
| **Target App** | Flask (CPU-burning prime sieve) |
| **Dataset** | GWA-Bitbrains (Kaggle) |

---

## How It Differs from Standard HPA

| Feature | Standard HPA | This Project |
|---|---|---|
| **Trigger** | Reactive (current CPU) | Predictive (future CPU) |
| **Metrics used** | 1 (CPU only) | 3 (CPU + RAM + Disk I/O) |
| **Time awareness** | None | Sinusoidal time-of-day encoding |
| **Cooldown logic** | Fixed delay | Smart cooldown + hysteresis |
| **Scale veto** | None | Post-scale load simulation |
| **VM history** | None | 12-step rolling window per VM |
| **Model** | Threshold rules | 120K param stacked LSTM |
| **Inference latency** | N/A | < 1ms (ONNX, CPU) |

---

## Training Your Own Model

1. Open `notebooks/EC2_computation_prediction.ipynb` in Google Colab
2. Connect to a T4 GPU runtime (`Runtime → Change runtime type → T4 GPU`)
3. Set up Kaggle API credentials
4. Run all cells sequentially
5. After training, run `serving/export_to_onnx.py` to export `.onnx`
6. Download `lstm_autoscaler.onnx` from Google Drive to `model/`

---

## API Reference

### `POST /predict_scale`

```json
// Request
{
  "vm_id": "my-cluster",
  "cpu_percent": 82.5,
  "ram_kb": 12582912,
  "disk_write_kbs": 340.0,
  "timestamp": 1718480000.0
}

// Response
{
  "status": "success",
  "vm_id": "my-cluster",
  "window_fill": "12/12",
  "predicted_cpu_pct": 84.78,
  "predicted_ram_mb": 12450.3,
  "predicted_diskw_kbs": 412.0,
  "scale_recommendation": "SCALE_UP",
  "pressure_score": 63.47,
  "inference_ms": 0.183
}
```

### `GET /health`

Returns model status, input/output shapes, feature names.

### `GET /metrics`

Prometheus-format counters: `lstm_predictions_total`, `lstm_scale_up_total`, `lstm_inference_latency_ms`, etc.

### `GET /window/{vm_id}`

Returns current rolling window fill status for a specific VM.

---

## Project Documentation

| Document | Description |
|---|---|
| [docs/PROJECT_DEEP_EXPLANATION.md](docs/PROJECT_DEEP_EXPLANATION.md) | Cell-by-cell notebook explanation with full technical justification |
| [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Step-by-step run guide with troubleshooting |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ using TensorFlow · ONNX · FastAPI · Kafka · Redis**

*Trained on GWA-Bitbrains — 1,250 real VM traces from Bitbrains datacenter*

</div>
