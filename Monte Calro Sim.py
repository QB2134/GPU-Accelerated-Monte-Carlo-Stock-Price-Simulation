import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import cupy as cp
from scipy.stats import lognorm

#Parameters:
S0 = 100          #inital stock price
K = 110           #strike price
r = 0.05          #risk-free rate
sigma = 0.2       #volatility
T = 1             #Time period (1 year here)
steps = 252       #daily time steps
N = 1000000      #number of simulated paths
dt = T / steps    #size of time step

# Risk-neutral drift explanation
# Under the *real world* measure, expected growth = μ (expected return).
# But under the *risk-neutral* measure (used for option pricing),
# we replace μ with r (risk-free rate), because we assume investors
# are indifferent to risk — only risk-free growth matters in pricing.
# Therefore, the drift term inside exp() becomes:
#(r - 0.5 * sigma^2) * dt
# The "-0.5*sigma^2" adjusts for the lognormal variance bias.

#Start timer
start_time = time.time()


#This is to generate random numbers
Z = cp.random.standard_normal((N, steps))

# This is to Simulate possible Stock paths
S = cp.zeros((N, steps + 1))
S[:, 0] = S0
for t in range(1, steps + 1):
    S[:, t] = S[:, t-1] * cp.exp((r - 0.5 * sigma**2) * dt + sigma * cp.sqrt(dt) * Z[:, t-1])

#Computing option payoffs at maturity
ST = S[:, -1]
call_payoff = cp.maximum(ST - K, 0)

#Future payoffs must be discounted by e^{-rT} since they occur T years later.
#This enforces the “no arbitrage” principle in risk-neutral pricing.
discounted_price = cp.exp(-r * T) * cp.mean(call_payoff)

print(f"Monte Carlo estimated cal option price: {discounted_price.get():.2f}")

#Length of simulation code
end_time = time.time()
elapsed = end_time - start_time

print(f"Simulation completed in {elapsed:.4f} seconds using GPU.")

#Coded in order to convert a small subet to cpu for plotting
S_cpu = S[:100].get()
plt.figure(figsize=(10,6))
for i in range(10):
    plt.plot(S_cpu[i])
plt.title('Sample Simulated Stock Price Paths')
plt.xlabel('Time Space')
plt.ylabel('Stock Price')
plt.show()

#Historgram for final prices
plt.hist(S_cpu[:, -1], bins=50, color='skyblue', edgecolor='k')
plt.title('Distribution of Final Stock Prices')
plt.xlabel('Price at T')
plt.ylabel('Frequency')
plt.show()

#compute theoretical parameters
mu_ln = np.log(S0) + (r - 0.5 * sigma**2) * T
sigma_ln = sigma * np.sqrt(T)

#create a range of prices for the density curve
s_vals = np.linspace(0.5*S0, 2.5*S0, 1000)
pdf_vals = lognorm.pdf(s_vals, s=sigma_ln, scale=np.exp(mu_ln))

#plot histogram of simulated prices
plt.hist(S_cpu[:, -1], bins=50, color='skyblue', edgecolor='k', density=True, label='Simulated')

#overlay theoretical log-normal density
plt.plot(s_vals, pdf_vals, 'r-', lw=2, label='Theoretical Log-normal PDF')
plt.title('Distribution of Final Stock Prices vs Log-normal PDF')
plt.xlabel('Price at T')
plt.ylabel('Density')
plt.legend()
plt.show()