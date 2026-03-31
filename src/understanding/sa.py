import numpy as np
from scipy.stats import norm

# dnorm() equivalent - PDF
print("PDF value at 0:", norm.pdf(0, loc=0, scale=1))

# pnorm() equivalent - CDF
print("CDF value at 1:", norm.cdf(1, loc=0, scale=1))

# qnorm() equivalent - Quantile
print("Quantile for 0.95 probability:", norm.ppf(0.95, loc=0, scale=1))

# rnorm() equivalent - Random normal numbers
print("Random normal values (mean=50, sd=10):", np.random.normal(loc=50, scale=10, size=5))

# --- Exercises ---
# 1. Generate 10 random numbers (mean=100, sd=15)
rand_nums = np.random.normal(loc=100, scale=15, size=10)
print("\n10 Random Numbers (mean=100, sd=15):", rand_nums)

# 2. Cumulative probability for value 2 (standard normal)
cum_prob = norm.cdf(2, loc=0, scale=1)
print("\nCumulative Probability for value 2:", cum_prob)