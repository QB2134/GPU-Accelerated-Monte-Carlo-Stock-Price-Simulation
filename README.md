🧠 GPU-Accelerated Monte Carlo Stock Price Simulation

This project runs a Monte Carlo simulation of stock-price dynamics under Geometric Brownian Motion (GBM) using CuPy for GPU acceleration.
It estimates the fair value of a European call option, visualizes sample paths, and compares the simulated terminal-price distribution to the theoretical log-normal curve.

🔢 Run Summary
Metric	Result
Monte Carlo Call Option Price	$ 6.04 (K = 110)
Simulations	1 000 000 paths
Steps	252 (daily over 1 year)
GPU Runtime	≈ 1.13 seconds (CuPy on CUDA)
⚙️ Model Setup

The simulation assumes:

Risk-neutral drift = r – 0.5 σ²

Random shocks ∼ Normal(0, 1)

Payoff = max(S_T – K, 0)

Discounting = exp(–r T) × E[payoff]

Parameters

S0 = 100       # initial stock price
K  = 110       # strike price
r  = 0.05      # risk-free rate
sigma = 0.2    # volatility
T  = 1         # years
steps = 252    # daily steps
N  = 1_000_000 # number of simulations

📈 Visual Outputs

1️⃣ Sample Simulated Stock Price Paths


2️⃣ Empirical Distribution of Final Prices


3️⃣ Simulated vs Theoretical Log-Normal PDF


The red curve closely follows the simulated histogram, confirming that the model reproduces the log-normal distribution implied by GBM.

💻 Tech Stack

Python 3.10 +

CuPy – GPU array computations

NumPy / SciPy – mathematical utilities

Matplotlib – visualization

Install dependencies:

pip install cupy-cuda12x matplotlib scipy numpy


(Use the CuPy build matching your CUDA version.)

⚡ Performance Notes

1 M paths × 252 steps completed in ≈ 1.13 s on GPU

Typical NumPy CPU runtime ≈ 30 s for same load

Easily scales via batch simulation for 10 M + paths

✍️ Author

Qudus Bawa-Allah
