"""Demonstrate how to calculate different types of correlation in Python."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import random

output_noise = 0.4
input_noise = 0.2
lag = 10.0

min_x = 0
max_x = 20
n_points = 100


x = np.linspace(min_x, max_x, n_points)
lagged_and_noisy_x = x.copy()
for i in range(len(x)):
    lagged_and_noisy_x[i] = lagged_and_noisy_x[i] + random() * input_noise + lag

y = np.sin(lagged_and_noisy_x)
for i in range(len(y)):
    y[i] = y[i] + random() * output_noise


df = pd.DataFrame(
    {
        "x": x,
        "y": y,
    }
)


def time_lagged_correlation(df, shifts, x="x", y="y", method="pearson"):
    """Return the correlations between two time series with different time lags."""
    correlations = []
    shifted = df[y].shift(shifts)
    for next_shifted in shifted:
        correlation = df[x].corr(shifted[next_shifted], method=method)
        correlations.append([next_shifted, correlation])
    return correlations


def cross_correlation(df, method="pearson"):
    cross_correlation = time_lagged_correlation(
        df, range(-int(len(df) / 2 * 0.5), int(len(df) / 2 * 0.5)), method=method
    )
    return cross_correlation


# Pearson correlation
# Assumes linear relationship and normal distribution
print("Pearson Correlations:")
print(df.corr(method="pearson"))
# Works for nonlinear correlations that must not have a normal distribution
# It measures monotonic relationships (i.e., variables increase or decrease together)
print("Spearman rank correlation:")
print(df.corr(method="spearman"))
# Doesn't need linear relationship or normal distribution
# # Measures the strength of ordinal associations.
# Best for small datasets and robust against outliers.
# this needs scipy installed
print("Kendall Tau Correlation:")
print(df.corr(method="kendall"))

# Time-lagged correlation
print("Time-lagged correlation:")
print(cross_correlation(df, method="kendall"))
print(max(cross_correlation(df, method="kendall"), key=lambda x: x[1]))

# Partial-auto-correlation
# This is the correlation between a variable and a lagged version of itself
from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(df["y"], lags=100)
# plt.show()

# If we want to measure relationship between a scalar and a binary value use:
# df['binary_var'].corr(df['continuous_var'])
# This is a special case of spearman, where the binary variable has to be 0|1
# Conversion possible as follows:
# df['binary_var'] = df['binary_var'].map({'class1': 0, 'class2': 1})


df.plot(x="x", y="y")
plt.show()
