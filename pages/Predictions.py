import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(page_title="Prédictions ETF", page_icon="🔮", layout="wide")
st.title("🔮 Algorithme Prédictif ETF (Random Forest)")
st.write("Ce module utilise une intelligence artificielle pour anticiper les tendances à 6 mois.")

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

TICKERS = {
    "PAASI":  "PAASI.PA",
    "PINDIA": "PINR.PA",
    "PUST":   "PUST.PA",
    "EFENSE": "DEFS.PA",
    "IWSC":   "IWSC.PA",
}

ETF_PARAMS = {
    "PAASI":  {"mu": 0.07,  "sigma": 0.16, "price": 28.5},
    "PINDIA": {"mu": 0.12,  "sigma": 0.20, "price": 45.2},
    "PUST":   {"mu": 0.15,  "sigma": 0.22, "price": 62.8},
    "EFENSE": {"mu": 0.18,  "sigma": 0.19, "price": 38.1},
    "IWSC":   {"mu": 0.09,  "sigma": 0.17, "price": 31.4},
}

HORIZON_JOURS = 126
TRAIN_YEARS   = 5
N_ESTIMATORS  = 300
RANDOM_STATE  = 42

# ════════════════════════════════════════════════════════════════════
#  FONCTIONS
# ════════════════════════════════════════════════════════════════════

def simulate_etf(name, n_days=1260):
    p = ETF_PARAMS[name]
    mu = p["mu"] / 252
    sigma = p["sigma"] / np.sqrt(252)
    S0 = p["price"]
    np.random.seed(RANDOM_STATE + hash(name) % 1000)
    eps = np.random.normal(0, 1, n_days)
    eps = 0.6 * eps + 0.4 * np.roll(eps, 1)
    log_returns = (mu - 0.5 * sigma**2) + sigma * eps
    closes = S0 * np.exp(np.cumsum(log_returns))
    daily_vol = np.abs(np.random.normal(0, sigma * 0.5, n_days))
    highs  = closes * (1 + daily_vol)
    lows   = closes * (1 - daily_vol)
    opens  = np.roll(closes, 1); opens[0] = S0
    volumes = np.random.lognormal(np.log(500_000), 0.4, n_days)
    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                         "Close": closes, "Volume": volumes}, index=dates)


def download_data(tickers_dict, years=5):
    n_days = years * 252
    data = {}
    source_info = {}
    for name, ticker in tickers_dict.items():
        loaded = False
        try:
            import yfinance as yf
            end = datetime.today()
            start = end - timedelta(days=years * 365 + 30)
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) > 200:
                    data[name] = df
                    source_info[name] = f"✅ Yahoo Finance ({len(df)} jours)"
                    loaded = True
        except Exception:
            pass
        if not loaded:
            data[name] = simulate_etf(name, n_days=n_days)
            source_info[name] = f"🔁 Données simulées (Yahoo indisponible)"
    return data, source_info


def compute_features(df):
    feat = pd.DataFrame(index=df.index)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    for lag in [1, 2, 3, 5, 10, 21, 63]:
        feat[f"ret_{lag}d"] = close.pct_change(lag)
    for w in [5, 10, 20, 50, 200]:
        feat[f"ma_{w}"] = close / close.rolling(w).mean() - 1
    for w in [10, 21, 63]:
        feat[f"vol_{w}d"] = close.pct_change().rolling(w).std() * np.sqrt(252)
    for period in [14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        feat[f"rsi_{period}"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    feat["macd"] = macd / close
    feat["macd_sig"] = signal / close
    feat["macd_hist"] = (macd - signal) / close
    for w in [20, 50]:
        ma_b = close.rolling(w).mean()
        sd_b = close.rolling(w).std()
        feat[f"bb_up_{w}"] = (close - (ma_b + 2*sd_b)) / close
        feat[f"bb_lo_{w}"] = (close - (ma_b - 2*sd_b)) / close
        feat[f"bb_wid_{w}"] = 4 * sd_b / ma_b
    tr = pd.concat([high-low, (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close
    feat["atr_21"] = tr.rolling(21).mean() / close
    feat["vol_r20"] = vol / (vol.rolling(20).mean() + 1)
    feat["vol_r5"]  = vol / (vol.rolling(5).mean() + 1)
    feat["dow"]     = df.index.dayofweek / 4.0
    feat["month"]   = df.index.month / 12.0
    feat["quarter"] = df.index.quarter / 4.0
    feat["mom_acc"] = close.pct_change(21) - close.pct_change(63)
    feat["drawdown"] = close / close.rolling(252, min_periods=1).max() - 1
    return feat.replace([np.inf, -np.inf], np.nan)


def build_dataset(data, horizon):
    datasets = {}
    for name, df in data.items():
        feat = compute_features(df)
        future_ret = df["Close"].pct_change(horizon).shift(-horizon) * 100
        direction  = (future_ret > 0).astype(int)
        combined = feat.copy()
        combined["__y_ret__"] = future_ret
        combined["__y_dir__"] = direction
        combined = combined.dropna()
        X     = combined.drop(columns=["__y_ret__", "__y_dir__"])
        y_dir = combined["__y_dir__"]
        y_ret = combined["__y_ret__"]
        datasets[name] = {"X": X, "y_dir": y_dir, "y_ret": y_ret}
    return datasets


def safe_n_splits(n_samples, max_splits=5, min_train=200):
    possible = (n_samples - min_train) // max(min_train // max_splits, 1)
    return max(2, min(possible, max_splits))


def train_and_evaluate(datasets, progress_bar=None):
    results = {}
    etf_list = list(datasets.keys())
    for i, name in enumerate(etf_list):
        if progress_bar:
            progress_bar.progress((i) / len(etf_list),
                                  text=f"Entraînement {name}...")
        ds = datasets[name]
        X, y_dir, y_ret = ds["X"], ds["y_dir"], ds["y_ret"]
        n = len(X)
        n_splits = safe_n_splits(n)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=8,
                                     min_samples_leaf=20, max_features="sqrt",
                                     class_weight="balanced", n_jobs=-1,
                                     random_state=RANDOM_STATE)
        acc_scores, all_y_true, all_y_pred = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, y_tr = X.iloc[train_idx], y_dir.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx],  y_dir.iloc[test_idx]
            if len(y_tr.unique()) < 2 or len(X_tr) < 50:
                continue
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc_scores.append(accuracy_score(y_te, y_pred))
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())
        clf.fit(X, y_dir)

        reg = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=8,
                                    min_samples_leaf=20, max_features="sqrt",
                                    n_jobs=-1, random_state=RANDOM_STATE)
        mae_scores, r2_scores = [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, r_tr = X.iloc[train_idx], y_ret.iloc[train_idx]
            X_te, r_te = X.iloc[test_idx],  y_ret.iloc[test_idx]
            if len(X_tr) < 50:
                continue
            reg.fit(X_tr, r_tr)
            r_pred = reg.predict(X_te)
            mae_scores.append(mean_absolute_error(r_te, r_pred))
            r2_scores.append(r2_score(r_te, r_pred))
        reg.fit(X, y_ret)

        feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
        results[name] = {
            "clf": clf, "reg": reg, "X": X, "y_dir": y_dir, "y_ret": y_ret,
            "acc_mean":  float(np.mean(acc_scores)) if acc_scores else 0.5,
            "mae_mean":  float(np.mean(mae_scores)) if mae_scores else 0.0,
            "r2_mean":   float(np.mean(r2_scores))  if r2_scores  else 0.0,
            "top_features": feat_imp.nlargest(10),
        }
    if progress_bar:
        progress_bar.progress(1.0, text="✅ Entraînement terminé !")
    return results


def predict_current(results, data):
    preds = []
    for name, res in results.items():
        last = res["X"].iloc[[-1]]
        dir_pred  = int(res["clf"].predict(last)[0])
        dir_proba = res["clf"].predict_proba(last)[0]
        pct_pred  = float(res["reg"].predict(last)[0])
        conf = dir_proba[dir_pred] * 100
        last_price = float(data[name]["Close"].iloc[-1]) if name in data else None
        est_price  = last_price * (1 + pct_pred / 100) if last_price else None
        preds.append({
            "ETF":             name,
            "Direction":       "HAUSSE" if dir_pred == 1 else "BAISSE",
            "Confiance (%)":   round(conf, 1),
            "Variation est.%": round(pct_pred, 2),
            "Accuracy CV":     f"{res['acc_mean']:.1%}",
            "MAE CV (%)":      round(res["mae_mean"], 2),
            "Prix actuel (€)": round(last_price, 2) if last_price else "N/A",
            "Prix estimé (€)": round(est_price, 2)  if est_price  else "N/A",
        })
    return pd.DataFrame(preds)


def backtest_strategy(results, data):
    bt_results = {}
    for name, res in results.items():
        if name not in data:
            continue
        X, y_dir = res["X"], res["y_dir"]
        n_splits = safe_n_splits(len(X))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oos_preds = pd.Series(np.nan, index=X.index)
        for train_idx, test_idx in tscv.split(X):
            X_tr, y_tr = X.iloc[train_idx], y_dir.iloc[train_idx]
            if len(y_tr.unique()) < 2 or len(X_tr) < 50:
                continue
            m = RandomForestClassifier(n_estimators=100, max_depth=8,
                                       min_samples_leaf=20, n_jobs=-1,
                                       random_state=RANDOM_STATE)
            m.fit(X_tr, y_tr)
            oos_preds.iloc[test_idx] = m.predict(X.iloc[test_idx])
        oos_preds = oos_preds.dropna()
        if len(oos_preds) == 0:
            continue
        close     = data[name]["Close"]
        daily_ret = close.pct_change().reindex(oos_preds.index)
        strat_ret = daily_ret * oos_preds
        cum_bh    = (1 + daily_ret).cumprod()
        cum_strat = (1 + strat_ret).cumprod()
        sharpe = lambda r: float((r.mean() / (r.std() + 1e-9)) * np.sqrt(252))
        maxdd  = lambda c: float(((c - c.cummax()) / c.cummax()).min()) * 100
        bt_results[name] = {
            "cum_bh": cum_bh, "cum_strat": cum_strat,
            "total_bh":    float(cum_bh.iloc[-1] - 1) * 100,
            "total_strat": float(cum_strat.iloc[-1] - 1) * 100,
            "sharpe_bh":   sharpe(daily_ret),
            "sharpe_strat": sharpe(strat_ret),
            "mdd_bh":   maxdd(cum_bh),
            "mdd_strat": maxdd(cum_strat),
        }
    return bt_results


PALETTE = ["#00D4FF", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA"]
BG, BG2, BORDER = "#0D1117", "#161B22", "#30363D"

def _style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_color(BORDER)

def plot_all(results, bt_results, pred_df, data, save_dir="."):
    plt.style.use("dark_background")
    cmap = {n: PALETTE[i] for i, n in enumerate(results.keys())}
    os.makedirs(save_dir, exist_ok=True)

    # Fig 1 : Feature importance + accuracy
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    fig.suptitle(f"🔮 PRÉDICTIONS ETF — Horizon 6 mois | {datetime.today().strftime('%d/%m/%Y')}",
                 fontsize=15, fontweight="bold", color="white", y=0.99)
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    tbl = ax_t.table(cellText=pred_df.values.tolist(),
                     colLabels=pred_df.columns.tolist(),
                     cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(BG2); cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="#60A5FA", fontweight="bold")
        else:
            row = pred_df.iloc[r-1]
            col = "#34D399" if row["Direction"] == "HAUSSE" else "#F87171"
            cell.set_text_props(color=col if c == 1 else "white")
    ax_t.set_title("Tableau des Prédictions (Horizon ≈ 6 mois)",
                   color="#60A5FA", fontsize=11, pad=12)
    for i, name in enumerate(list(results.keys())[:5]):
        r, c = [(1,0),(1,1),(1,2),(2,0),(2,1)][i]
        ax = fig.add_subplot(gs[r, c]); _style_ax(ax)
        top = results[name]["top_features"]
        ax.barh(range(len(top)), top.values, color=cmap[name], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=7, color="white")
        ax.set_title(f"Top Features — {name}", color=cmap[name], fontsize=9, fontweight="bold")
    ax_m = fig.add_subplot(gs[2, 2]); _style_ax(ax_m)
    names = list(results.keys())
    accs  = [results[n]["acc_mean"]*100 for n in names]
    bars  = ax_m.bar(np.arange(len(names)), accs,
                     color=[cmap[n] for n in names], alpha=0.85)
    ax_m.axhline(50, color="#EF4444", linestyle="--", alpha=0.8, label="50% aléatoire")
    ax_m.set_xticks(np.arange(len(names)))
    ax_m.set_xticklabels(names, rotation=20, fontsize=8, color="white")
    ax_m.set_ylim(0, 100); ax_m.legend(fontsize=7)
    ax_m.set_title("Accuracy Classification", color="white", fontsize=9, fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax_m.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                  f"{acc:.1f}%", ha="center", fontsize=7, color="white")
    fig.savefig(f"{save_dir}/fig1_predictions.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # Fig 2 : Backtesting
    if bt_results:
        n = len(bt_results)
        fig2, axes = plt.subplots(n, 1, figsize=(16, 4*n))
        fig2.patch.set_facecolor(BG)
        if n == 1: axes = [axes]
        fig2.suptitle("Backtesting — Signal RF vs Buy & Hold",
                      fontsize=13, fontweight="bold", color="white")
        for ax, (name, bt) in zip(axes, bt_results.items()):
            _style_ax(ax)
            c = cmap.get(name, "#60A5FA")
            bh = bt["cum_bh"].dropna(); st = bt["cum_strat"].dropna()
            ax.plot(bh.index, bh.values, color="#94A3B8", linewidth=1.2,
                    label=f"Buy & Hold ({bt['total_bh']:+.1f}%)")
            ax.plot(st.index, st.values, color=c, linewidth=1.8,
                    label=f"Signal RF ({bt['total_strat']:+.1f}%)")
            ax.fill_between(st.index, 1, st.values, alpha=0.08, color=c)
            ax.axhline(1, color="white", linestyle="--", alpha=0.3)
            ax.set_title(f"{name} | Sharpe RF: {bt['sharpe_strat']:.2f} vs BH: {bt['sharpe_bh']:.2f} | MaxDD RF: {bt['mdd_strat']:.1f}%",
                         color=c, fontsize=9)
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{(x-1)*100:+.0f}%"))
        fig2.tight_layout()
        fig2.savefig(f"{save_dir}/fig2_backtest.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig2)

    # Fig 3 : Prix
    if data:
        n = len(data)
        fig3, axes = plt.subplots(n, 1, figsize=(16, 3.5*n))
        fig3.patch.set_facecolor(BG)
        if n == 1: axes = [axes]
        fig3.suptitle("Historique des Prix + MAs + Bollinger",
                      fontsize=13, fontweight="bold", color="white")
        for ax, (name, df) in zip(axes, data.items()):
            _style_ax(ax)
            c = cmap.get(name, "#60A5FA"); cl = df["Close"]
            ax.plot(cl.index, cl.values, color=c, linewidth=1.1, label="Prix")
            ax.plot(cl.index, cl.rolling(50).mean(), color="#F59E0B",
                    linewidth=1, alpha=0.8, label="MA50")
            ax.plot(cl.index, cl.rolling(200).mean(), color="#EF4444",
                    linewidth=1, alpha=0.8, label="MA200")
            ma20 = cl.rolling(20).mean(); sd20 = cl.rolling(20).std()
            ax.fill_between(cl.index, ma20-2*sd20, ma20+2*sd20,
                            alpha=0.07, color=c, label="Bollinger 2σ")
            ax.set_title(name, color=c, fontsize=9, fontweight="bold")
            ax.legend(fontsize=7)
        fig3.tight_layout()
        fig3.savefig(f"{save_dir}/fig3_prix.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig3)

# ════════════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ════════════════════════════════════════════════════════════════════

if st.button("🚀 Calculer et afficher les prédictions en direct"):
    OUTPUT_DIR = "output_etf_predictor"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    progress = st.progress(0, text="📥 Chargement des données...")

    try:
        # Données
        data, source_info = download_data(TICKERS, years=TRAIN_YEARS)
        progress.progress(0.15, text="🔧 Construction des features...")

        # Affiche les sources
        cols = st.columns(5)
        for i, (name, info) in enumerate(source_info.items()):
            cols[i].caption(f"**{name}** : {info}")

        # Features
        datasets = build_dataset(data, horizon=HORIZON_JOURS)
        progress.progress(0.30, text="🏋️ Entraînement du Random Forest...")

        # Entraînement
        results = train_and_evaluate(datasets, progress_bar=progress)
        progress.progress(0.80, text="🔮 Calcul des prédictions...")

        # Prédictions
        pred_df   = predict_current(results, data)
        progress.progress(0.90, text="📈 Backtesting...")

        bt_results = backtest_strategy(results, data)
        progress.progress(0.95, text="📊 Génération des graphiques...")

        plot_all(results, bt_results, pred_df, data, save_dir=OUTPUT_DIR)
        progress.progress(1.0, text="✅ Terminé !")

        st.success("✅ Analyses complétées avec succès !")

        # ── Tableau des prédictions ──────────────────────────────
        st.write("---")
        st.subheader("📊 Tableau des Prédictions (Horizon ~6 mois)")

        def color_direction(val):
            if val == "HAUSSE":
                return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
            elif val == "BAISSE":
                return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
            return ""

        st.dataframe(pred_df.style.map(color_direction, subset=["Direction"]),
                     use_container_width=True)

        # ── Graphiques ──────────────────────────────────────────
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Backtesting")
            if os.path.exists(f"{OUTPUT_DIR}/fig2_backtest.png"):
                st.image(f"{OUTPUT_DIR}/fig2_backtest.png",
                         caption="Stratégie RF vs Buy & Hold")
        with col2:
            st.subheader("📉 Analyse Technique")
            if os.path.exists(f"{OUTPUT_DIR}/fig3_prix.png"):
                st.image(f"{OUTPUT_DIR}/fig3_prix.png",
                         caption="Historique + Bollinger")

        st.write("---")
        if os.path.exists(f"{OUTPUT_DIR}/fig1_predictions.png"):
            st.image(f"{OUTPUT_DIR}/fig1_predictions.png",
                     caption="Feature Importance + Accuracy par ETF")

        st.info("ℹ️ Si Yahoo Finance est indisponible, les prédictions utilisent des données simulées réalistes.")

    except Exception as e:
        progress.empty()
        st.error(f"❌ Erreur : {e}")
        st.exception(e)
