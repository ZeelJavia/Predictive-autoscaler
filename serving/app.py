from flask import Flask
import math

app = Flask(__name__)

@app.route('/')
def burn_cpu():
    primes = []
    for num in range(2, 100000):
        is_prime = True
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return f"Calculated {len(primes)} primes. CPU burn complete!\n"

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
