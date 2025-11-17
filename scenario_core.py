

import numpy as _np
import pandas as _pd
import statsmodels.api as _sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from math import floor
import warnings
warnings.filterwarnings('ignore')


ADD_TREND = True
TREND_DEG = 1
week_freq = 'W-FRI'



import pandas as pd
import numpy as np


def aggregate_to_weekly(df, date_col="date", week_freq="W-WED", agg="last",
                        drop_all_nan=True, require_all_cols=None):
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    if agg == "last":
        w = d.resample(week_freq).last()
    elif agg == "mean":
        w = d.resample(week_freq).mean()
    elif agg == "sum":
        w = d.resample(week_freq).sum()
    else:
        w = d.resample(week_freq).agg(agg)

 

    if drop_all_nan:
        w = w.dropna(how="all")
    if require_all_cols is not None:
        w = w.loc[w[require_all_cols].notna().all(axis=1)]
    return w



import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


WINDOW_LEN = 331   # окно оценки ECM (нед.)
LAGY, LAGX = 1, 1  # лаги Δy и ΔX в ECM
WEEKS_H    = 108   # горизонт прогноза (нед.)
N_PATHS    = 200   # число стохастических путей
SEED       = 123
CORR_WIN   = 104   # окно (нед.) для μ/σ/корреляций по ΔX
ADD_TREND  = True  # тренд в коинтеграции
TREND_DEG  = 1     # степень тренда (обычно 1 — линейный)


def build_ecm_matrix(y_win, X_win, alpha, delta, beta_dict, lagy, lagx):
    """
    Δy_t ~ const + ECT_{t-1} + лаги Δy + лаги ΔX, где
    ECT_t = y_t - alpha - delta*t - Σ beta_k * X_t^{(k)}  (в уровнях)
    """
    t  = np.arange(len(y_win), dtype=float) + 1
    ECT = y_win - alpha - delta*t - sum(beta_dict[k]*X_win[k] for k in beta_dict)
    Dy  = y_win.diff()
    dX  = X_win.diff()

    Z = pd.DataFrame(index=y_win.index)
    Z["ECT_L1"] = ECT.shift(1)
    for i in range(1, lagy+1):
        Z[f"Dy_L{i}"] = Dy.shift(i)
    for col in X_win.columns:
        for j in range(1, lagx+1):
            Z[f"D{col}_L{j}"] = dX[col].shift(j)

    ecm = pd.concat([Dy.rename("Dy"), Z], axis=1).dropna()
    Y = ecm["Dy"]
    Xreg = sm.add_constant(ecm.drop(columns="Dy"), has_constant="add")
    return Y, Xreg


def fit_short_run_on_last_window_ols(y, X, window_len, lagy, lagx,
                                     add_trend=True, trend_degree=1):
   
    y_win = y.iloc[-window_len:]
    X_win = X.iloc[-window_len:]

    
    t = np.arange(len(y_win), dtype=float) + 1
    design = X_win.copy()
    if add_trend and trend_degree >= 1:
        for p in range(1, trend_degree + 1):
            design[f"t^{p}"] = t**p
    design = sm.add_constant(design, has_constant="add")

    ols_fit = sm.OLS(y_win, design).fit()
    alpha = float(ols_fit.params.get("const", 0.0))
    delta = float(ols_fit.params.get("t^1", 0.0)) if add_trend and trend_degree >= 1 else 0.0
    beta_dict = {col: float(ols_fit.params.get(col, 0.0)) for col in X_win.columns}

    
    Y_tr, Xreg_tr = build_ecm_matrix(y_win, X_win, alpha, delta, beta_dict, lagy, lagx)
    ecm_fit = sm.OLS(Y_tr, Xreg_tr).fit()
    return alpha, delta, beta_dict, ecm_fit.params.copy()


def _safe_loc_last_le(series_or_df, ts, col=None):
    obj = series_or_df[col] if col is not None else series_or_df
    try:
        v = obj.loc[ts]
        if isinstance(v, pd.Series):
            v = v.iloc[-1]
        return float(v)
    except KeyError:
        idx = obj.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            pos = 0
        return float(obj.iloc[pos])

def _wk(obj):
    o = obj.copy()
    if isinstance(o.index, pd.PeriodIndex):
        o.index = o.index.to_timestamp()
    if not isinstance(o.index, pd.DatetimeIndex):
        raise TypeError("Ожидается DatetimeIndex/PeriodIndex.")
    return o.sort_index()

def simulate_ecm_scenario_weekly(y_hist, X_hist, X_future,
                                 alpha, delta, beta_dict, sr_params,
                                 lagy, lagx, window_len):
   
    y_hist  = _wk(y_hist)
    X_hist  = _wk(X_hist)
    X_future= _wk(X_future)

    X_all  = pd.concat([X_hist, X_future], axis=0)[X_hist.columns]
    X_all  = X_all.loc[~X_all.index.duplicated(keep="last")]
    dX_all = X_all.diff()
    y_sim  = y_hist.copy()

    t_num_start = len(y_hist.iloc[-window_len:])
    fut_idx = X_future.index

    for step, tdate in enumerate(fut_idx, start=1):
        prev = tdate - pd.Timedelta(weeks=1)
        t_num = t_num_start + step - 1

        ect = (_safe_loc_last_le(y_sim, prev)
               - alpha - delta * t_num
               - sum(beta_dict[k] * _safe_loc_last_le(X_all, prev, col=k) for k in beta_dict))

        row = {"const": 1.0, "ECT_L1": ect}
        Dy_sim = y_sim.diff()
        for i in range(1, lagy+1):
            row[f"Dy_L{i}"] = _safe_loc_last_le(Dy_sim, prev - pd.Timedelta(weeks=i-1))
        for col in X_hist.columns:
            for j in range(1, lagx+1):
                row[f"D{col}_L{j}"] = _safe_loc_last_le(dX_all, prev - pd.Timedelta(weeks=j-1), col=col)

        s = pd.Series(row).reindex(sr_params.index).fillna(0.0)
        Dy_hat = float(np.dot(sr_params.values, s.values))
        y_next = _safe_loc_last_le(y_sim, prev) + Dy_hat
        y_sim.loc[tdate] = y_next

    return y_sim.loc[fut_idx]


def build_future_index_weekly(last_date, weeks=108, week_freq=week_freq):
    
    start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).to_period(week_freq).to_timestamp()
    return pd.date_range(start, periods=weeks, freq=week_freq)

def _weekly_rate_from_annual(r_year):
    
    return (1.0 + r_year)**(1.0/52.0) - 1.0

def path_from_two_annual_rates_weekly_level(x_last, r_year1, r_year2, horizon_weeks, idx):
    
    w1 = _weekly_rate_from_annual(r_year1)
    w2 = _weekly_rate_from_annual(r_year2)
    inc = np.r_[np.full(min(52, horizon_weeks), w1), np.full(max(0, horizon_weeks-52), w2)]
    out = np.empty(horizon_weeks); cur = x_last
    for i in range(horizon_weeks):
        cur = cur * (1.0 + inc[i]); out[i] = cur
    return pd.Series(out, index=idx)

def path_trend52w_level(x_hist, horizon_weeks, idx):
    mu = x_hist.diff().iloc[-52:].mean()
    out = np.empty(horizon_weeks); cur = x_hist.iloc[-1]
    for i in range(horizon_weeks):
        cur = cur + mu; out[i] = cur
    return pd.Series(out, index=idx)

def path_mean_revert_level(x_hist, horizon_weeks, idx):
    mu = x_hist.iloc[-260:].mean() if len(x_hist) >= 260 else x_hist.mean()
    steps = np.linspace(x_hist.iloc[-1], mu, horizon_weeks+1)[1:]
    return pd.Series(steps, index=idx)

def build_scenarios_X_weekly(X_hist, horizon_weeks=WEEKS_H):
    last_date = X_hist.index[-1]
    fut_idx   = build_future_index_weekly(last_date, weeks=horizon_weeks, week_freq=week_freq)
    last      = X_hist.iloc[-1]
    return {
        "Baseline": pd.DataFrame({
            "SP500": path_from_two_annual_rates_weekly_level(last["SP500"], 0.06, 0.06, horizon_weeks, fut_idx),
            "DXY":   path_from_two_annual_rates_weekly_level(last["DXY"],   0.00, 0.00, horizon_weeks, fut_idx),
            "OIL":   path_from_two_annual_rates_weekly_level(last["OIL"],   0.00, 0.00, horizon_weeks, fut_idx),
        }, index=fut_idx),
        "Risk_on": pd.DataFrame({
            "SP500": path_from_two_annual_rates_weekly_level(last["SP500"], 0.12, 0.12, horizon_weeks, fut_idx),
            "DXY":   path_from_two_annual_rates_weekly_level(last["DXY"],  -0.05, -0.05, horizon_weeks, fut_idx),
            "OIL":   path_from_two_annual_rates_weekly_level(last["OIL"],   0.10, 0.10, horizon_weeks, fut_idx),
        }, index=fut_idx),
        "Risk_off": pd.DataFrame({
            "SP500": path_from_two_annual_rates_weekly_level(last["SP500"], -0.40, -0.40, horizon_weeks, fut_idx),
            "DXY":   path_from_two_annual_rates_weekly_level(last["DXY"],    0.20, 0.20, horizon_weeks, fut_idx),
            "OIL":   path_from_two_annual_rates_weekly_level(last["OIL"],   -0.15, -0.20, horizon_weeks, fut_idx),
        }, index=fut_idx),
        "Strong_dollar": pd.DataFrame({
            "SP500": path_from_two_annual_rates_weekly_level(last["SP500"],  -0.20, -0.20, horizon_weeks, fut_idx),
            "DXY":   path_from_two_annual_rates_weekly_level(last["DXY"],     0.20, 0.20, horizon_weeks, fut_idx),
            "OIL":   path_from_two_annual_rates_weekly_level(last["OIL"],    -0.10, -0.10, horizon_weeks, fut_idx),
        }, index=fut_idx),
        "Oil_shock_up": pd.DataFrame({
            "SP500": path_from_two_annual_rates_weekly_level(last["SP500"], -0.05, 0.04, horizon_weeks, fut_idx),
            "DXY":   path_from_two_annual_rates_weekly_level(last["DXY"],    0.02, 0.00, horizon_weeks, fut_idx),
            "OIL":   path_from_two_annual_rates_weekly_level(last["OIL"],    0.30, 0.00, horizon_weeks, fut_idx),
        }, index=fut_idx),
        "Trend_52w":   pd.DataFrame({c: path_trend52w_level(X_hist[c], horizon_weeks, fut_idx) for c in X_hist.columns}, index=fut_idx),
        "Mean_revert": pd.DataFrame({c: path_mean_revert_level(X_hist[c], horizon_weeks, fut_idx) for c in X_hist.columns}, index=fut_idx),
    }


def simulate_X_paths_with_sv_weekly(X_det_future, X_hist, n_paths=N_PATHS, corr_window=CORR_WIN, seed=SEED):
    rng = np.random.default_rng(seed)
    cols = list(X_det_future.columns)
    T = len(X_det_future)

    dX_hist = X_hist.diff().dropna()
    if corr_window is not None and len(dX_hist) > corr_window:
        dX_hist = dX_hist.iloc[-corr_window:]

    mu = dX_hist.mean()
    sigma = dX_hist.std().replace(0.0, 1e-12)
    R = dX_hist.corr().fillna(0.0).values
    R = R + 1e-9 * np.eye(len(cols))  # стабилизация
    L = np.linalg.cholesky(R)

    paths = []
    for _ in range(n_paths):
        X_path = X_det_future.copy()
        prev = X_hist.iloc[-1].copy()

        for t in range(T):
            z_std  = rng.standard_normal(len(cols))
            z_corr = L @ z_std

            dX_t = {}
            for j, col in enumerate(cols):
                eps_t = float(mu[col] + sigma[col] * z_corr[j])
                det_step = (
                    X_det_future.iloc[t, X_det_future.columns.get_loc(col)]
                    - (prev[col] if t == 0 else X_det_future.iloc[t-1, X_det_future.columns.get_loc(col)])
                )
                dX_t[col] = det_step + eps_t

            new = prev.copy()
            for col in cols:
                new[col] = prev[col] + dX_t[col]
            X_path.iloc[t] = new.values
            prev = new

        paths.append(X_path)
    return paths

