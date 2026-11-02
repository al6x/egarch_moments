"""pip install -r requirements.txt, then python play.py."""
import csv
import math
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.optimize import minimize

N_TRIALS, N_SIM, SEED = 100, 100_000, 0
NAMES = ("mu", "omega", "beta", "alpha", "gamma", "nu")
STATS = ("mean", "P(R≤q25)", "P(R≥q50)", "P(R≥q75)")
BOUNDS = ((-2, 2), (-10, 10), (.0001, .999), (0, 2), (0, 2), (2.05, 100))


@njit(cache=True)
def t_constants(nu):
    g = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
    return 2 * math.sqrt(nu - 2) * math.exp(g) / ((nu - 1) * math.sqrt(math.pi)), g - .5 * math.log(math.pi * (nu - 2))


@njit(cache=True)
def loglik(q, r, warmup):
    mu, omega, beta, alpha, gamma, nu = q
    ea, c = t_constants(nu)
    m, llh = 0., 0.
    for i in range(len(r)):
        v = omega + m
        if not math.isfinite(v) or abs(v) > 700: return -np.inf
        z = (r[i] - mu) * math.exp(-v / 2)
        if not math.isfinite(z * z): return -np.inf
        if i >= warmup: llh += c - v / 2 - (nu + 1) / 2 * math.log1p(z * z / (nu - 2))
        m = beta * m + alpha * (abs(z) - ea) - gamma * z
    return llh


@njit(cache=True)
def simulate(q, z, burn):
    mu, omega, beta, alpha, gamma, nu = q
    ea, _ = t_constants(nu)
    r, m = np.empty(len(z) - burn), 0.
    for i in range(len(z)):
        v = omega + m
        if not math.isfinite(v) or abs(v) > 700: raise FloatingPointError("nonfinite simulation scale")
        if i >= burn: r[i - burn] = mu + math.exp(v / 2) * z[i]
        m = beta * m + alpha * (abs(z[i]) - ea) - gamma * z[i]
    return r


def moments(r, levels=None):
    s = np.r_[0., np.cumsum(r)]
    x = s[21:] - s[:-21]
    tails = np.quantile(x, [.25, .5, .75]) if levels is None else [
        np.mean(x <= levels[0]), np.mean(x >= levels[1]), np.mean(x >= levels[2])]
    result = np.r_[x.mean(), tails]
    assert np.all(np.isfinite(result))
    return result


def main():
    root = Path(__file__).resolve().parent
    with (root / "spy.csv").open() as f: rows = list(csv.DictReader(f))
    dates, prices = [date.fromisoformat(x["date"]) for x in rows], np.array([float(x["close"]) for x in rows])
    assert all(a < b for a, b in zip(dates, dates[1:])) and np.all(np.isfinite(prices) & (prices > 0))
    r, warmup = np.diff(np.log(prices)), np.searchsorted(dates[1:], dates[1] + timedelta(days=91), side="right")
    center, scale, count = r.mean(), np.mean(np.abs(r - r.mean())), len(r) - warmup
    assert scale > 0 and count > 21
    rn = (r - center) / scale
    loss = lambda q: -loglik(q, rn, warmup) / count
    fits = []
    for beta in (.9, .98):
        initial = [0., np.log(np.var(rn)), beta, .15, .1, 8.]
        fit = minimize(loss, initial, method="L-BFGS-B", bounds=BOUNDS,
                       options={"maxiter": 2000, "ftol": 1e-11, "maxls": 50})
        if not fit.success: raise RuntimeError(f"MLE not converged: {fit.message}")
        fits.append(fit)
    q = min(fits, key=lambda f: f.fun).x
    burn = max(1000, math.ceil(math.log(1e-10) / math.log(q[2])))
    levels = moments(r)[1:]  # Fixed observed thresholds, lower tail at q25, upper at q50/q75.

    def sample(n, rng):
        z = rng.standard_t(q[5], n + burn) * np.sqrt((q[5] - 2) / q[5])
        return moments(center + scale * simulate(q, z, burn), levels)

    rng, real = np.random.default_rng(SEED + 1), moments(r, levels)
    trials = np.array([sample(len(r), rng) for _ in range(N_TRIALS)])
    point = sample(N_SIM, np.random.default_rng(SEED))
    widths = np.diff(np.quantile(trials, [.1, .9], axis=0), axis=0)[0]
    assert np.all(widths > 0) and np.all(real != 0)
    ratios, (lo, hi) = point / real, np.quantile(trials / real, [.1, .9], axis=0)
    print(f"SPY {dates[1]} to {dates[-1]}, {count} likelihood targets")
    print(f"LLH={-count * (loss(q) + np.log(scale)):.6f}, moment error={np.mean(((point-real)/widths)**2):.6f}")
    raw = q.copy()
    raw[0], raw[1] = center + scale * q[0], q[1] + 2 * np.log(scale)
    print("Parameters (raw returns):", dict(zip(NAMES, raw.tolist())))
    for name, ratio, low, high in zip(STATS, ratios, lo, hi): print(f"{name:>4}: {ratio:.4f} [{low:.4f}, {high:.4f}]")
    print("Sample mean only: t-EGARCH has no finite unconditional absolute mean for nonzero volatility shocks.")
    out = root / "results"
    out.mkdir(exist_ok=True)
    with (out / "moments.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "threshold", "observed", "simulated", "ratio", "q10", "q90"])
        w.writerows(zip(STATS, ["", *levels], real, point, ratios, lo, hi))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.vlines(range(4), lo, hi, color=".75", linewidth=12, label="10–90% across simulated histories")
    ax.scatter(range(4), ratios, color="black", zorder=3, label="Long simulation / observed")
    ax.axhline(1, color="steelblue", linewidth=1)
    ax.set(xticks=range(4), xticklabels=STATS, yscale="log", ylim=(.5, 2),
           ylabel="Simulated / observed",
           title="SPY · 21-day log returns\nTail probabilities at observed quantiles, plus mean ratio")
    ax.set_yticks([.5, .75, 1, 1.5, 2], labels=["0.5", "0.75", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    fig.savefig(out / "moments.png", dpi=180)
    plt.close(fig)
    formulas = [
        r"$r_t = \mu + \sqrt{h_t}\,z_t,\qquad"
        r"z_t \overset{\mathrm{iid}}{\sim} \sqrt{\frac{\nu-2}{\nu}}\,T_\nu$",
        r"$\log h_t = \omega + m_t$",
        r"$m_t = \beta m_{t-1} + \alpha\,(|z_{t-1}|-\mathbb{E}|z_t|) - \gamma z_{t-1}$",
        r"$t:\ \mathrm{trading\ days}.\quad \mathbb{E}z_t=0,\quad \mathbb{E}z_t^2=1,\quad"
        r"0<\beta<1,\quad \alpha,\gamma\geq0,\quad \nu>2.$"]
    fig = plt.figure(figsize=(12, 3.4), facecolor="white")
    for y, text in zip([.83, .62, .40, .12], formulas): fig.text(.04, y, text, fontsize=21)
    fig.savefig(out / "model.png", dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


if __name__ == "__main__": main()
