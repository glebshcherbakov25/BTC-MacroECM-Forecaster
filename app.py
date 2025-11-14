import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from typing import Dict, List

from scenario_core import (
    aggregate_to_weekly,
    fit_short_run_on_last_window_ols,
    simulate_ecm_scenario_weekly,
    path_from_two_annual_rates_weekly_level,
    path_trend52w_level,
    path_mean_revert_level,
)

st.set_page_config(page_title="BTC Scenario ECM Dashboard (v2)", layout="wide")

st.title("Сценарный прогноз BTC (ECM, weekly) — v2")
st.caption("Задавайте сценарии для SP500 / DXY / OIL и получайте прогноз траекторий BTC. "
           "Есть кнопка «Смоделировать», фан‑чарты и расчёт доходности портфеля.")

# ========== Sidebar: Data & hyperparams ==========
with st.sidebar:
    st.header("Данные")
    start_date = st.date_input("Начало истории", dt.date(2010,1,1))
    end_date = st.date_input("Конец истории", dt.date.today())
    use_auto = st.checkbox("Скачать автоматически (BTC-USD, ^SPX, DX-Y.NYB, BZ=F)", value=True)

    btc_csv = st.file_uploader("BTC CSV (date,BTC)", type=["csv"], key="btc")
    spx_csv = st.file_uploader("SP500 CSV (date,SP500)", type=["csv"], key="spx")
    dxy_csv = st.file_uploader("DXY CSV (date,DXY)", type=["csv"], key="dxy")
    oil_csv = st.file_uploader("OIL CSV (date,OIL)", type=["csv"], key="oil")

    st.markdown("---")
    st.subheader("Гиперпараметры")
    WINDOW_LEN = st.number_input("Окно оценки (недель)", 60, 520, 331, 1)
    LAGY       = st.number_input("Лаги ΔBTC (LAGY)", 1, 4, 1, 1)
    LAGX       = st.number_input("Лаги ΔX (LAGX)", 0, 4, 1, 1)
    H          = st.number_input("Горизонт прогноза (недель)", 12, 260, 108, 4)
    N_PATHS    = st.number_input("Стохастических путей", 50, 2000, 400, 50)
    ADD_TREND  = st.checkbox("Тренд в коинтеграции", value=True)
    TREND_DEG  = st.selectbox("Степень тренда", [1,2], index=0)

@st.cache_data(show_spinner=False)
def _download(symbol: str, start: str, end: str, col: str, rename_to: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
    except Exception:
        return pd.DataFrame(columns=["date", rename_to])
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", rename_to])
    df = df.reset_index()
    if "Date" in df.columns: df = df.rename(columns={"Date":"date"})
    elif "Datetime" in df.columns: df = df.rename(columns={"Datetime":"date"})
    elif "date" not in df.columns and hasattr(df.index, "inferred_type") and "date" not in df.columns:
        df = df.reset_index().rename(columns={"index":"date"})
    if col not in df.columns:
        col = "Close" if "Close" in df.columns else df.columns[1]
    out = df[["date", col]].rename(columns={col: rename_to}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")

def _ensure_date_col(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "date" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index":"date"})
        elif d.index.name == "date":
            d = d.reset_index()
        else:
            for c in ["Date","Datetime","DATE","timestamp","time"]:
                if c in d.columns:
                    d = d.rename(columns={c:"date"})
                    break
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d.dropna(subset=["date"]).sort_values("date")

def _load_or_download(start, end, btc_csv, spx_csv, dxy_csv, oil_csv, use_auto=True) -> pd.DataFrame:
    frames = {}
    # BTC
    if btc_csv is not None:
        d = pd.read_csv(btc_csv); d = _ensure_date_col(d)
        col = "BTC" if "BTC" in d.columns else ([c for c in d.columns if c.lower() in ("btc","close","adj close","price") and c!="date"]+[d.columns[1]])[0]
        frames["BTC"] = d[["date", col]].rename(columns={col:"BTC"})
    elif use_auto:
        frames["BTC"] = _download("BTC-USD", str(start), str(end), "Adj Close", "BTC")
    # SP500
    if spx_csv is not None:
        d = pd.read_csv(spx_csv); d = _ensure_date_col(d)
        col = "SP500" if "SP500" in d.columns else ([c for c in d.columns if ("sp" in c.lower() or "s&p" in c.lower()) and c!="date"]+[d.columns[1]])[0]
        frames["SP500"] = d[["date", col]].rename(columns={col:"SP500"})
    elif use_auto:
        frames["SP500"] = _download("^SPX", str(start), str(end), "Close", "SP500")
    # DXY
    if dxy_csv is not None:
        d = pd.read_csv(dxy_csv); d = _ensure_date_col(d)
        col = "DXY" if "DXY" in d.columns else ([c for c in d.columns if ("dxy" in c.lower() or "usd index" in c.lower() or "dollar index" in c.lower()) and c!="date"]+[d.columns[1]])[0]
        frames["DXY"] = d[["date", col]].rename(columns={col:"DXY"})
    elif use_auto:
        frames["DXY"] = _download("DX-Y.NYB", str(start), str(end), "Close", "DXY")
    # OIL
    if oil_csv is not None:
        d = pd.read_csv(oil_csv); d = _ensure_date_col(d)
        col = "OIL" if "OIL" in d.columns else ([c for c in d.columns if any(k in c.lower() for k in ("oil","brent","close","price")) and c!="date"]+[d.columns[1]])[0]
        frames["OIL"] = d[["date", col]].rename(columns={col:"OIL"})
    elif use_auto:
        frames["OIL"] = _download("BZ=F", str(start), str(end), "Close", "OIL")

    required = ["BTC","SP500","DXY","OIL"]
    base = next((k for k in required if k in frames and not frames[k].empty), None)
    if base is None:
        return pd.DataFrame(columns=["date"]+required)
    df = _ensure_date_col(frames[base])
    for k in required:
        if k == base: continue
        if k in frames and not frames[k].empty:
            d = _ensure_date_col(frames[k])
            df = df.merge(d, on="date", how="outer")
        else:
            df[k] = np.nan
    df = _ensure_date_col(df).drop_duplicates(subset=["date"], keep="last")
    return df

# ===== Load daily & weekly =====
df_daily = _load_or_download(start_date, end_date, btc_csv, spx_csv, dxy_csv, oil_csv, use_auto)
if df_daily.empty:
    st.error("Пустые данные. Задай другой период/загрузи CSV."); st.stop()

st.subheader("1) История (daily)")
st.dataframe(df_daily.tail(10), use_container_width=True)

dfw = aggregate_to_weekly(df_daily, date_col="date", week_freq="W-FRI", agg="last",
                          drop_all_nan=True, require_all_cols=["BTC","SP500","DXY","OIL"]).dropna(how="any")
if dfw.empty:
    st.error("После weekly-агрегации всё пропало. Проверь столбцы и период."); st.stop()

st.subheader("2) Недельные данные (W-FRI, уровни)")
st.dataframe(dfw.tail(10), use_container_width=True)

if len(dfw) < int(WINDOW_LEN) + max(int(LAGY), int(LAGX)) + 5:
    st.error(f"Слишком короткая история: {len(dfw)} недель. Уменьши WINDOW_LEN или измени период.")
    st.stop()

y = dfw["BTC"].astype(float)
X = dfw[["SP500","DXY","OIL"]].astype(float)

# ===== Fit ECM =====
try:
    alpha, delta, beta_last, sr_params = fit_short_run_on_last_window_ols(
        y, X, int(WINDOW_LEN), int(LAGY), int(LAGX), add_trend=bool(ADD_TREND), trend_degree=int(TREND_DEG)
    )
except Exception as e:
    st.exception(e); st.stop()

with st.expander("Коинтеграция (OLS) → ECM: параметры"):
    st.json({"alpha": alpha, "delta": delta, "beta": beta_last, **{f"sr_{k}": float(v) for k,v in sr_params.items()}})

# ===== Scenario controls =====
st.header("3) Сценарии факторов")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**SP500**")
    sp_mode = st.radio("Режим", ["Custom годовые темпы", "Тренд 52 недель", "Возврат к среднему"], key="spm")
    if sp_mode == "Custom годовые темпы":
        sp_r1 = st.number_input("Годовой темп (первые 52 недели), %", -100.0, 200.0, 6.0, 0.5)
        sp_r2 = st.number_input("Годовой темп (далее), %", -100.0, 200.0, 6.0, 0.5)
with c2:
    st.markdown("**DXY**")
    dx_mode = st.radio("Режим", ["Custom годовые темпы", "Тренд 52 недель", "Возврат к среднему"], key="dxm")
    if dx_mode == "Custom годовые темпы":
        dx_r1 = st.number_input("Годовой темп (первые 52 недели), %", -100.0, 200.0, 0.0, 0.5)
        dx_r2 = st.number_input("Годовой темп (далее), %", -100.0, 200.0, 0.0, 0.5)
with c3:
    st.markdown("**OIL**")
    oil_mode = st.radio("Режим", ["Custom годовые темпы", "Тренд 52 недель", "Возврат к среднему"], key="oilm")
    if oil_mode == "Custom годовые темпы":
        oil_r1 = st.number_input("Годовой темп (первые 52 недели), %", -100.0, 200.0, 0.0, 0.5)
        oil_r2 = st.number_input("Годовой темп (далее), %", -100.0, 200.0, 0.0, 0.5)

# deterministic future
last = X.iloc[-1]
fut_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=int(H), freq="W-FRI")

def build_det():
    data = {}
    # SP500
    if sp_mode == "Custom годовые темпы":
        data["SP500"] = path_from_two_annual_rates_weekly_level(float(last["SP500"]), sp_r1/100.0, sp_r2/100.0, int(H), fut_idx)
    elif sp_mode == "Тренд 52 недель":
        data["SP500"] = path_trend52w_level(X["SP500"], int(H), fut_idx)
    else:
        data["SP500"] = path_mean_revert_level(X["SP500"], int(H), fut_idx)
    # DXY
    if dx_mode == "Custom годовые темпы":
        data["DXY"] = path_from_two_annual_rates_weekly_level(float(last["DXY"]), dx_r1/100.0, dx_r2/100.0, int(H), fut_idx)
    elif dx_mode == "Тренд 52 недель":
        data["DXY"] = path_trend52w_level(X["DXY"], int(H), fut_idx)
    else:
        data["DXY"] = path_mean_revert_level(X["DXY"], int(H), fut_idx)
    # OIL
    if oil_mode == "Custom годовые темпы":
        data["OIL"] = path_from_two_annual_rates_weekly_level(float(last["OIL"]), oil_r1/100.0, oil_r2/100.0, int(H), fut_idx)
    elif oil_mode == "Тренд 52 недель":
        data["OIL"] = path_trend52w_level(X["OIL"], int(H), fut_idx)
    else:
        data["OIL"] = path_mean_revert_level(X["OIL"], int(H), fut_idx)
    return pd.DataFrame(data, index=fut_idx)

X_future_det = build_det()
st.subheader("Детерминированные пути факторов (preview)")
st.dataframe(X_future_det.head(6), use_container_width=True)

# stochastic fan for factors
def make_stochastic_paths(X_hist: pd.DataFrame, X_det_future: pd.DataFrame, n_paths: int, seed: int = 123):
    dX_hist = X_hist.diff().dropna().iloc[-104:]
    mu = dX_hist.mean()
    sigma = dX_hist.std().replace(0.0, 1e-9)
    cols = list(X_hist.columns)
    R = dX_hist.corr().fillna(0.0).values
    R = R + 1e-9 * np.eye(len(cols))
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    T = len(X_det_future)
    paths = []
    for _ in range(n_paths):
        X_path = X_det_future.copy()
        prev = X_hist.iloc[-1].copy()
        for t in range(T):
            z = rng.standard_normal(len(cols))
            zc = L @ z
            dX_noise = {cols[j]: float(mu[cols[j]] + sigma[cols[j]] * zc[j]) for j in range(len(cols))}
            det_step = (X_det_future.iloc[t] - (prev if t==0 else X_det_future.iloc[t-1]))
            new = prev + det_step + pd.Series(dX_noise)
            X_path.iloc[t] = new.values
            prev = new
        paths.append(X_path)
    return paths

st.markdown("---")
run = st.button("🚀 Смоделировать")

if run:
    with st.spinner("Считаю стохастические траектории факторов и ECM‑прогноз BTC..."):
        X_paths = make_stochastic_paths(X, X_future_det, int(N_PATHS))
        # BTC simulations
        btc_paths = []
        for Xp in X_paths:
            y_sim = simulate_ecm_scenario_weekly(y, X, Xp, alpha, delta, beta_last, sr_params,
                                                 int(LAGY), int(LAGX), int(WINDOW_LEN))
            btc_paths.append(y_sim)

        idx = btc_paths[0].index
        arr = np.column_stack([s.reindex(idx).values for s in btc_paths])
        fan_btc = pd.DataFrame({
            "p10": np.quantile(arr, 0.10, axis=1),
            "p50": np.quantile(arr, 0.50, axis=1),
            "p90": np.quantile(arr, 0.90, axis=1),
        }, index=idx)

        # factor fans
        def fan_from_paths(paths: List[pd.DataFrame], col: str) -> pd.DataFrame:
            idx = paths[0].index
            arr = np.column_stack([p[col].values for p in paths])
            return pd.DataFrame({"p10": np.quantile(arr,0.10,axis=1),
                                 "p50": np.quantile(arr,0.50,axis=1),
                                 "p90": np.quantile(arr,0.90,axis=1)}, index=idx)
        fan_spx = fan_from_paths(X_paths, "SP500")
        fan_dxy = fan_from_paths(X_paths, "DXY")
        fan_oil = fan_from_paths(X_paths, "OIL")

    # ======= Plots (BTC + factors) =======
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11,7))
    gs = GridSpec(3,1, height_ratios=[2.2,1.2,1.2], hspace=0.25)

    # BTC
    ax1 = fig.add_subplot(gs[0,0])
    hist_tail = y.index[-1] - pd.DateOffset(years=3)
    hist_cut = y[y.index >= hist_tail]
    ax1.plot(hist_cut.index, hist_cut.values, lw=1.4, color="0.3", label="BTC (факт)")
    ax1.fill_between(fan_btc.index, fan_btc["p10"], fan_btc["p90"], alpha=0.25, label="BTC P10–P90")
    ax1.plot(fan_btc.index, fan_btc["p50"], lw=2.0, label="BTC медиана")
    ax1.axvline(fan_btc.index[0], color="k", lw=0.8, ls="--")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper left")

    # SP500 & DXY (twinx)
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(X.index, X["SP500"], lw=1.0, label="SP500 (история)")
    ax2.fill_between(fan_spx.index, fan_spx["p10"], fan_spx["p90"], alpha=0.15, label="SP500 P10–P90")
    ax2.plot(fan_spx.index, fan_spx["p50"], lw=2.0, label="SP500 медиана")
    ax2_t = ax2.twinx()
    ax2_t.plot(X.index, X["DXY"], lw=1.0, label="DXY (история)", color="tab:orange")
    ax2_t.fill_between(fan_dxy.index, fan_dxy["p10"], fan_dxy["p90"], alpha=0.15, color="tab:orange")
    ax2_t.plot(fan_dxy.index, fan_dxy["p50"], lw=2.0, color="tab:orange", label="DXY медиана")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left"); ax2_t.legend(loc="upper right")

    # OIL
    ax3 = fig.add_subplot(gs[2,0])
    ax3.plot(X.index, X["OIL"], lw=1.0, label="OIL (история)")
    ax3.fill_between(fan_oil.index, fan_oil["p10"], fan_oil["p90"], alpha=0.15, label="OIL P10–P90")
    ax3.plot(fan_oil.index, fan_oil["p50"], lw=2.0, label="OIL медиана")
    ax3.grid(alpha=0.3); ax3.legend(loc="upper left")

    st.pyplot(fig, use_container_width=True)

    # ======= Portfolio returns =======
    st.header("Доходность портфеля (по медианному пути BTC)")
    c1, c2, c3 = st.columns(3)
    with c1:
        init_cap = st.number_input("Начальный капитал", 100.0, 1e9, 10000.0, 100.0, format="%.2f", key="cap")
    with c2:
        w_btc = st.slider("Доля BTC в портфеле", 0.0, 1.0, 0.5, 0.05, key="wbtc")
    with c3:
        horizon_weeks = st.number_input("Горизонт (недель)", 4, int(H), min(52, int(H)), 4, key="hweeks")

    p50 = fan_btc["p50"]
    ret_btc = (p50 / p50.iloc[0] - 1.0)
    ret_port = w_btc * ret_btc
    st.write(f"**Доходность BTC (P50) за {int(horizon_weeks)} недель:** {ret_btc.iloc[int(horizon_weeks)-1]*100:.2f}%")
    st.write(f"**Доходность портфеля (P50) за {int(horizon_weeks)} недель:** {ret_port.iloc[int(horizon_weeks)-1]*100:.2f}%")
    st.write(f"**Итоговая стоимость портфеля:** {init_cap * (1 + ret_port.iloc[int(horizon_weeks)-1]):,.2f}")

    # Export CSV
    st.subheader("Выгрузка данных")
    out = pd.concat(
        [p50.rename("BTC_P50"),
         fan_btc["p10"].rename("BTC_P10"),
         fan_btc["p90"].rename("BTC_P90"),
         X_future_det.add_prefix("DET_").reindex(p50.index)], axis=1
    )
    st.dataframe(out.head(8), use_container_width=True)
    st.download_button("Скачать прогноз (CSV)", data=out.to_csv(index=True).encode("utf-8"),
                       file_name="btc_scenario_forecast.csv", mime="text/csv")
else:
    st.info("Выберите сценарии и нажмите **«Смоделировать»** для построения графиков и расчёта доходности.")

