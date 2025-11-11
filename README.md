🧠 GPU-Accelerated Monte Carlo Stock Price Simulation

This project implements a Monte Carlo simulation of stock-price dynamics under Geometric Brownian Motion (GBM) using CuPy (GPU) for large-scale stochastic modeling and option pricing.
It visualizes sample price paths, the empirical terminal distribution, and overlays the theoretical log-normal PDF for validation.

📊 Key Results
Metric	Value
Monte Carlo Call Option Price	$ 6.04 ( K = 110 )
Simulations	1,000,000 paths
Steps	252 (daily over 1 year)
GPU Runtime	≈ 1.13 seconds
⚙️ Model Summary

The stock price evolves as

𝑆
𝑡
+
Δ
𝑡
=
𝑆
𝑡
 
𝑒
(
𝑟
−
1
2
𝜎
2
)
Δ
𝑡
+
𝜎
Δ
𝑡
 
𝑍
𝑡
,
𝑍
𝑡
∼
𝑁
(
0
,
1
)
S
t+Δt
	​

=S
t
	​

e
(r−
2
1
	​

σ
2
)Δt+σ
Δt
	​

Z
t
	​

,Z
t
	​

∼N(0,1)

under the risk-neutral measure where 
𝑟
r replaces the expected return μ.
The call-option payoff is

max
⁡
(
𝑆
𝑇
−
𝐾
,
0
)
,
𝐶
0
=
𝑒
−
𝑟
𝑇
 
𝐸
[
max
⁡
(
𝑆
𝑇
−
𝐾
,
0
)
]
.
max(S
T
	​

−K,0),C
0
	​

=e
−rT
E[max(S
T
	​

−K,0)].

Parameters used in this experiment:

S0 = 100       # initial stock price
K  = 110       # strike
r  = 0.05      # risk-free rate
sigma = 0.2    # volatility
T  = 1         # years
steps = 252    # daily steps
N  = 1_000_000 # GPU simulations

📈 Visualizations

1️⃣ Sample Simulated Stock Price Paths

2️⃣ Empirical Distribution of Final Prices

3️⃣ Simulated vs Theoretical Log-Normal PDF

The red curve closely follows the simulated histogram, confirming that the model correctly reproduces the log-normal distribution implied by GBM.

💻 Tech Stack

Python 3.10 +

CuPy – GPU-accelerated array math

Matplotlib – visualization

SciPy – log-normal PDF for theoretical fit

Install dependencies:

pip install cupy-cuda12x matplotlib scipy numpy


(Use the correct CuPy build for your CUDA version.)

🧮 Performance

1 million paths × 252 steps computed entirely on GPU in ≈ 1.13 s

Equivalent CPU (NumPy) version typically > 30 s

Linear scalability with batch simulation for 10 M+ paths

✍️ Author

Qudus Bawa-Allah
