"""Synthetic end-to-end smoke test for strategy.py (no network needed)."""
import numpy as np
import pandas as pd
import strategy as S

rng = np.random.default_rng(42)
n_days, n_stocks = 1600, 60
dates = pd.bdate_range("2015-01-02", periods=n_days)

# Synthetic FF5 factors + RF
ff = pd.DataFrame(
    rng.normal(0.0002, 0.008, size=(n_days, 5)),
    index=dates, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
ff["RF"] = 0.0001

# Synthetic stock prices with true factor loadings
betas_true = rng.normal(0, 1, size=(n_stocks, 5))
eps = rng.normal(0, 0.012, size=(n_days, n_stocks))
rets = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values @ betas_true.T + eps + 0.0003
tickers = [f"STK{i:02d}" for i in range(n_stocks)]
prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0),
                      index=dates, columns=tickers)
spy = pd.DataFrame({"SPY": 100 * np.cumprod(1 + ff["Mkt-RF"] + ff["RF"])},
                   index=dates)

cfg = S.Config(start_date="2015-01-01", universe=tuple(tickers))
R = S.run_backtest(cfg, ff, prices, spy)

v = R["variants"]
assert not v.empty, "variants empty"
assert v.notna().all().all(), "NaNs in variants"
assert (R["signals"].abs() <= 1.0 + 1e-9).all().all(), "signal out of bounds"
assert (R["leverage"] <= cfg.max_leverage + 1e-9).all(), "leverage cap broken"

print(R["stats_table"].to_string())
print(f"\nNW alpha: {R['alpha_ann']:.2%} p.a. (t={R['alpha_t']:.2f})")
print(f"Avg annual turnover: {R['turnover'].mean()*252:.1f}x")

g = S.sensitivity_grid(R["sleeve_df"], cfg)
print("\nSensitivity grid:\n", g.round(2).to_string())
print("\n✅ PIPELINE OK")
