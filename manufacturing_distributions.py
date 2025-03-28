"""Show examples of common distributions and their relevance in statistics."""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

distributions_detailed = [
    {
        "name": "Skewed (Right)",
        "data": stats.skewnorm.rvs(a=10, loc=0, scale=1, size=10000),
        "params": r"Skew Normal (a=10): Right-skewed",
        "meaning": "Long tail on the right. Common in tool wear or delay-heavy processes.",
    },
    {
        "name": "Skewed (Left)",
        "data": stats.skewnorm.rvs(a=-10, loc=0, scale=1, size=10000),
        "params": r"Skew Normal (a=-10): Left-skewed",
        "meaning": "Long tail on the left. Seen in early-start conditions or processes that improve over time.",
    },
    {
        "name": "Bimodal",
        "data": np.concatenate(
            [
                np.random.normal(loc=-2, scale=0.5, size=5000),
                np.random.normal(loc=2, scale=0.5, size=5000),
            ]
        ),
        "params": r"Normal (μ=-2,2; σ=0.5)",
        "meaning": "Two peaks indicate multiple sources, machines, or shifts influencing the data.",
    },
    {
        "name": "Uniform",
        "data": np.random.uniform(low=0, high=1, size=10000),
        "params": r"Uniform (low=0, high=1)",
        "meaning": "All outcomes equally likely. Seen in randomized stress testing or sampling.",
    },
    {
        "name": "Exponential",
        "data": np.random.exponential(scale=1, size=10000),
        "params": r"Exponential (λ=1)",
        "meaning": "Models time between rare events (e.g., breakdowns or failures).",
    },
    {
        "name": "Weibull",
        "data": stats.weibull_min.rvs(c=1.5, size=10000),
        "params": r"Weibull (shape=1.5)",
        "meaning": "Common in reliability/failure modeling. Shape dictates failure behavior.",
    },
    {
        "name": "Poisson",
        "data": np.random.poisson(lam=3, size=10000),
        "params": r"Poisson (λ=3)",
        "meaning": "Discrete count data. Good for defects per batch or failures per hour.",
    },
    {
        "name": "Log-Normal",
        "data": np.random.lognormal(mean=0, sigma=0.5, size=10000),
        "params": r"Log-Normal (μ=0, σ=0.5)",
        "meaning": "Skewed right. Used when data can't be negative and grows multiplicatively.",
    },
    {
        "name": "Normal",
        "data": np.random.normal(loc=0, scale=1, size=10000),
        "params": r"Normal (μ=0, σ=1)",
        "meaning": "The classic bell curve. Often assumed but not always accurate in practice.",
    },
]

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(20, 14))
axs = axs.flatten()

for ax, dist in zip(axs, distributions_detailed):
    ax.hist(dist["data"], bins=50, density=True, alpha=0.7, edgecolor="black")
    ax.set_title(f"{dist['name']}\n{dist['params']}\n{dist['meaning']}", fontsize=10)
    ax.grid(True)

plt.tight_layout()
plt.show()
