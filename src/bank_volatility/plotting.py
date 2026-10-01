from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_price_and_returns(prices: pd.Series, returns: pd.Series, ticker: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(prices.index, prices.values)
    ax.set_title(f"{ticker}: closing price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(returns.index, returns.values)
    ax.set_title(f"{ticker}: daily log-returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_return_distribution(returns: pd.Series, ticker: str):
    values = pd.Series(returns).dropna().to_numpy()
    mu, sigma = float(values.mean()), float(values.std(ddof=0))
    grid = np.linspace(*np.quantile(values, [0.001, 0.999]), 600)
    kde = gaussian_kde(values)(grid)
    normal = norm.pdf(grid, loc=mu, scale=sigma)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grid, kde, label="Empirical KDE")
    ax.plot(grid, normal, label="Normal approximation")
    ax.set_title(f"{ticker}: log-return distribution")
    ax.set_xlabel("Log-return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()


def plot_acf_pacf_pair(series: pd.Series, title: str, lags: int = 30):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(pd.Series(series).dropna(), lags=lags, ax=ax)
    ax.set_title(f"{title}: ACF")
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(pd.Series(series).dropna(), lags=lags, ax=ax, method="ywm")
    ax.set_title(f"{title}: PACF")
    fig.tight_layout()
