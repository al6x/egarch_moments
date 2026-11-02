Модель E-GARCH-T фиттинг MLE, данные 20 лет истории SP500

    rₜ = μ + √hₜ zₜ
    log hₜ = ω + mₜ
    mₜ = βmₜ₋₁ + α(|zₜ₋₁| − E|zₜ|) − γzₜ₋₁
    zₜ ∼ √((ν−2)/ν) Tν, iid

Относительная ошибка модель/данные для среднего и квантилей q25, q50, q75 для лог прибыли 1мес.
Серый диапазон — q10–q90 отношений по симуляциям модели.

![Model diagnostics, simulated/observed ratios with 10–90% bands](results/moments.png)

Можно запустить самостоятельно, данные приложены.

# Run it

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python play.py

All code in play.py, settings at the top. Python 3.10–3.13.
Daily MLE only, first 91 calendar days used for filter warmup.

Overlapping 21-day log-return sums. Plain mean and quantile ratios, without centering.

    mean_ratio = mean_sim / mean_obs
    quantile_ratio[p] = Q_sim(p) / Q_obs(p), p = 0.25, 0.5, 0.75

Ratios are unstable near a zero observed mean or quantile.
Negative ratios fall outside the logarithmic plot.
10–90% bands across 100 simulated histories of observed length, fitted parameters fixed.
Point estimates from a separate 100,000-day simulation.
Simulation clips standardized shocks at p and 1−p quantiles, p=1/(20×262) per tail,
including burn-in and state updates. MLE and E|zₜ| centering use the uncapped Student-t.
Clipped shocks are not rescaled to unit variance.

    moment_error = mean(((simulated-observed)/(trial_q90-trial_q10))**2)

Uncapped Student-t EGARCH has no finite unconditional absolute-return mean for nonzero α or γ.
The simulation cap makes simulated moments finite, diagnostics refer to this capped process.

spy.csv is the stored SPY USD close series from bc-prices, SPY:ARCX/price/close,
2004-12-31 to 2026-09-30, as used by garch/play.jl. No additional dividend adjustment.
Output: LLH and diagnostics in terminal, results/moments.csv and results/moments.png.
Model formula: results/model.png.
