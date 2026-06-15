"""
STEP 3 — Kubernetes Metrics Producer
──────────────────────────────────────
Collects live metrics from pods every 5 minutes (300s).
Sends to Kafka topic 'vm-metrics'.

Kafka payload (matches consumer.py + main.py exactly):
  { vm_id, timestamp, cpu_percent, ram_kb, disk_write_kbs }

Setup:
    pip install kafka-python kubernetes
    minikube start
    minikube addons enable metrics-server
    kubectl apply -f ../step5_k8s/deployment.yaml
"""

import time
import json
import logging
from kafka import KafkaProducer
from kubernetes import client, config

# ─── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger("producer")

# ─── CONFIG ───────────────────────────────────────────────────────
KAFKA_TOPIC       = 'vm-metrics'
KAFKA_BROKERS     = ['localhost:9092']
TARGET_NAMESPACE  = 'default'
TARGET_DEPLOYMENT = 'cpu-burner'
SCRAPE_INTERVAL   = 300  # seconds (5 minutes) — must match Bitbrains dataset interval

# ─── INIT ─────────────────────────────────────────────────────────
log.info("Initializing...")
try:
    config.load_kube_config()   # reads ~/.kube/config (Minikube sets this)
    custom_api = client.CustomObjectsApi()
    k8s_apps   = client.AppsV1Api()

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=3
    )
    log.info("Kubernetes + Kafka connected.")
except Exception as e:
    log.error(f"Init failed: {e}")
    exit(1)

# ─── UNIT PARSERS ─────────────────────────────────────────────────
def cpu_to_millicores(s):
    s = str(s)
    if s.endswith('m'): return float(s[:-1])
    if s.endswith('n'): return float(s[:-1]) / 1_000_000.0
    return float(s) * 1000.0

def mem_to_kb(s):
    s = str(s)
    if s.endswith('Ki'): return float(s[:-2])
    if s.endswith('Mi'): return float(s[:-2]) * 1024.0
    if s.endswith('Gi'): return float(s[:-2]) * 1024.0 * 1024.0
    return float(s) / 1024.0

# ─── FETCH DEPLOYMENT LIMITS ──────────────────────────────────────
def get_limits():
    try:
        dep        = k8s_apps.read_namespaced_deployment(TARGET_DEPLOYMENT, TARGET_NAMESPACE)
        max_cpu_m  = 500.0
        max_ram_kb = 256.0 * 1024.0
        for c in dep.spec.template.spec.containers:
            if c.resources and c.resources.limits:
                lim = c.resources.limits
                if 'cpu'    in lim: max_cpu_m  = cpu_to_millicores(lim['cpu'])
                if 'memory' in lim: max_ram_kb = mem_to_kb(lim['memory'])
        log.info(f"Deployment limits → CPU:{max_cpu_m}m  RAM:{max_ram_kb/1024:.0f}MB")
        return max_cpu_m, max_ram_kb
    except Exception as e:
        log.warning(f"Could not read limits: {e} — using defaults 500m/256MB")
        return 500.0, 262144.0

MAX_CPU_MC, MAX_RAM_KB = get_limits()
scrape_count = 0   # used to refresh limits periodically

# ─── DISK WRITE TRACKING (rate = delta / interval) ────────────────
# metrics-server doesn't expose disk — we use ephemeral-storage delta
prev_disk_bytes: dict = {}

def get_disk_write_kbs(pod_name, current_bytes):
    global prev_disk_bytes
    prev = prev_disk_bytes.get(pod_name, current_bytes)
    delta_kb = max(0.0, (current_bytes - prev) / 1024.0)
    rate_kbs = delta_kb / SCRAPE_INTERVAL
    prev_disk_bytes[pod_name] = current_bytes
    return rate_kbs

# ─── MAIN LOOP ────────────────────────────────────────────────────
log.info(f"Scraping '{TARGET_DEPLOYMENT}' every {SCRAPE_INTERVAL}s...")

try:
    while True:
        try:
            # Refresh deployment limits every 100 scrapes (~8 hours at 5min interval)
            # Keeps cpu_pct accurate if you live-patch resource limits
            scrape_count += 1
            if scrape_count % 100 == 0:
                MAX_CPU_MC, MAX_RAM_KB = get_limits()
                log.info(f"Limits refreshed → CPU:{MAX_CPU_MC}m  RAM:{MAX_RAM_KB/1024:.0f}MB")
            metrics = custom_api.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=TARGET_NAMESPACE,
                plural="pods"
            )

            total_cpu_mc   = 0.0
            total_ram_kb   = 0.0
            total_disk_kbs = 0.0
            pod_count      = 0

            for pod in metrics.get('items', []):
                pod_name = pod['metadata']['name']
                if TARGET_DEPLOYMENT not in pod_name:
                    continue

                for c in pod.get('containers', []):
                    u = c.get('usage', {})
                    total_cpu_mc += cpu_to_millicores(u.get('cpu', '0m'))
                    total_ram_kb += mem_to_kb(u.get('memory', '0Ki'))

                    # Disk write rate approximation from ephemeral-storage
                    disk_bytes = float(u.get('ephemeral-storage', '0').replace('Ki', '')) \
                        if isinstance(u.get('ephemeral-storage', 0), str) else 0.0
                    total_disk_kbs += get_disk_write_kbs(pod_name, disk_bytes * 1024)

                pod_count += 1

            if pod_count > 0:
                avg_cpu_mc   = total_cpu_mc   / pod_count
                avg_ram_kb   = total_ram_kb   / pod_count
                avg_disk_kbs = total_disk_kbs / pod_count
                cpu_pct      = min((avg_cpu_mc / MAX_CPU_MC) * 100.0, 100.0)

                # Payload — keys match consumer.py and main.py exactly
                payload = {
                    'vm_id'         : f'{TARGET_DEPLOYMENT}-cluster',
                    'timestamp'     : int(time.time()),
                    'cpu_percent'   : round(cpu_pct, 2),
                    'ram_kb'        : round(avg_ram_kb, 2),
                    'disk_write_kbs': round(avg_disk_kbs, 2),
                }

                producer.send(KAFKA_TOPIC, payload)
                log.info(
                    f"Pods:{pod_count} | "
                    f"CPU:{cpu_pct:.1f}% | "
                    f"RAM:{avg_ram_kb/1024:.0f}MB | "
                    f"DiskW:{avg_disk_kbs:.0f}KB/s"
                )
            else:
                log.warning(f"No '{TARGET_DEPLOYMENT}' pods found. Is the deployment running?")
                log.warning("Run: kubectl apply -f ../step5_k8s/deployment.yaml")

        except Exception as e:
            log.error(f"Scrape error: {e}")

        time.sleep(SCRAPE_INTERVAL)

except KeyboardInterrupt:
    log.info("Producer stopped.")
finally:
    producer.close()
    log.info("Kafka producer closed.")
