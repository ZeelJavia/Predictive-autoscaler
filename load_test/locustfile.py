"""
STEP 6 — Locust Load Test
──────────────────────────
Simulates real user traffic hitting cpu-burner pods.
Triggers CPU spikes → producer detects → LSTM predicts → consumer scales.

Run (with web UI):
    pip install locust
    locust -f locustfile.py
    Open http://localhost:8089
    Host: http://$(minikube ip):30000

Run (headless):
    locust -f locustfile.py --users 50 --spawn-rate 5 --headless --run-time 10m \
        --host http://$(minikube ip):30000
"""

from locust import HttpUser, task, between


class NormalUser(HttpUser):
    """
    Simulates steady background traffic.
    50 users → moderate CPU load → tests scale-down.
    """
    wait_time = between(1, 3)

    @task(3)
    def compute_primes(self):
        """Hits the CPU-burning endpoint."""
        with self.client.get("/", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def health_check(self):
        """Light request — simulates monitoring ping."""
        self.client.get("/health", name="/health")


class SpikeUser(HttpUser):
    """
    Aggressive traffic — use separately to simulate sudden spike.
    100 users + fast spawn → triggers SCALE UP prediction.

    Run separately:
        locust -f locustfile.py SpikeUser --users 100 --spawn-rate 20 \
            --headless --run-time 3m --host http://$(minikube ip):30000
    """
    wait_time = between(0.1, 0.5)

    @task
    def spike(self):
        self.client.get("/")
