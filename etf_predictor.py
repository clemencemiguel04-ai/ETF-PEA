"""
╔══════════════════════════════════════════════════════════════════════╗
║         ALGORITHME PRÉDICTIF ETF — RANDOM FOREST                    ║
║         Horizon : 6 mois | PAASI, PINDIA, PUST, EFENSE, IWSC        ║
║                                                                      ║
║  UTILISATION :                                                       ║
║    pip install yfinance scikit-learn pandas numpy matplotlib         ║
║    python etf_predictor.py                                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import joblib

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

# Paramètres réels approx. de chaque ETF (mu annuel, sigma annuel, prix actuel)
ETF_PARAMS = {
    "PAASI":  {"mu": 0.07,  "sigma": 0.16, "price": 28.5},
    "PINDIA": {"mu": 0.12,  "sigma": 0.20, "price": 45.2},
    "PUST":   {"mu": 0.15,  "sigma": 0.22, "price": 62.8},
    "EFENSE": {"mu": 0.18,  "sigma": 0.19, "price": 38.1},
    "IWSC":   {"mu": 0.09,  "sigma": 0.17, "price": 31.4},
}

HORIZON_JOURS = 126   # ≈ 6 mois de trading
TRAIN_YEARS   = 5
N_ESTIMATORS  = 300
RANDOM_STATE  = 42

# ════════════════════════════════════════════════════════════════════
#  1. DONNÉES : Yahoo Finance → fallback simulées
# ════════════════════════════════════════════════════════════════════

def simulate_etf(name: str, n_days: int = 1260) -> pd.DataFrame:
    """Génère un historique OHLCV réaliste (GBM + autocorrélation)."""
    p  = ETF_PARAMS[name]
    mu = p["mu"] / 252
    sigma = p["sigma"] / np.sqrt(252)
    S0 = p["price"]

    np.random.seed(RANDOM_STATE + hash(name) % 1000)
    dt = 1
    eps = np.random.normal(0, 1, n_days)
    # Légère autocorrélation (momentum)
    eps = 0.6 * eps + 0.4 * np.roll(eps, 1)

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
    closes = S0 * np.exp(np.cumsum(log_returns))

    # OHLC réalistes
    daily_vol = np.abs(np.random.normal(0, sigma * 0.5, n_days))
    highs  = closes * (1 + daily_vol)
    lows   = closes * (1 - daily_vol)
    opens  = np.roll(closes, 1)
    opens[0] = S0
    volumes = np.random.lognormal(np.log(500_000), 0.4, n_days)

    end   = datetime.today()
    dates = pd.bdate_range(end=end, periods=n_days)

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }, index=dates)
    return df


def download_data(tickers_dict: dict, years: int = 5) -> dict:
    """Télécharge depuis Yahoo Finance, simule si indisponible."""
    print("\n" + "═"*60)
    print("  📥  CHARGEMENT DES DONNÉES")
    print("═"*60)

    n_days = years * 252
    data = {}

    for name, ticker in tickers_dict.items():
        loaded = False
        try:
            import yfinance as yf
            end   = datetime.today()
            start = end - timedelta(days=years * 365 + 30)
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c
                              for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) > 200:
                    data[name] = df
                    print(f"  ✅  {name:8s} | {len(df):5d} jours "
                          f"(Yahoo Finance) | "
                          f"{df.index[0].date()} → {df.index[-1].date()}")
                    loaded = True
        except Exception:
            pass

        if not loaded:
            df = simulate_etf(name, n_days=n_days)
            data[name] = df
            print(f"  🔁  {name:8s} | {len(df):5d} jours "
                  f"(données simulées — Yahoo indisponible)")

    return data


# ════════════════════════════════════════════════════════════════════
#  2. INGÉNIERIE DES FEATURES (~40 indicateurs)
# ════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    feat  = pd.DataFrame(index=df.index)
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # Rendements passés
    for lag in [1, 2, 3, 5, 10, 21, 63]:
        feat[f"ret_{lag}d"] = close.pct_change(lag)

    # Distance aux moyennes mobiles
    for w in [5, 10, 20, 50, 200]:
        ma = close.rolling(w).mean()
        feat[f"ma_{w}"] = close / ma - 1

    # Volatilité réalisée
    for w in [10, 21, 63]:
        feat[f"vol_{w}d"] = close.pct_change().rolling(w).std() * np.sqrt(252)

    # RSI
    for period in [14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12  = close.ewm(span=12).mean()
    ema26  = close.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    feat["macd"]      = macd / close
    feat["macd_sig"]  = signal / close
    feat["macd_hist"] = (macd - signal) / close

    # Bollinger Bands
    for w in [20, 50]:
        ma_b = close.rolling(w).mean()
        sd_b = close.rolling(w).std()
        feat[f"bb_up_{w}"]  = (close - (ma_b + 2 * sd_b)) / close
        feat[f"bb_lo_{w}"]  = (close - (ma_b - 2 * sd_b)) / close
        feat[f"bb_wid_{w}"] = 4 * sd_b / ma_b

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close
    feat["atr_21"] = tr.rolling(21).mean() / close

    # Volume relatif
    feat["vol_r20"] = vol / (vol.rolling(20).mean() + 1)
    feat["vol_r5"]  = vol / (vol.rolling(5).mean() + 1)

    # Saisonnalité
    feat["dow"]     = df.index.dayofweek / 4.0
    feat["month"]   = df.index.month / 12.0
    feat["quarter"] = df.index.quarter / 4.0

    # Momentum accélération
    feat["mom_acc"] = close.pct_change(21) - close.pct_change(63)

    # Drawdown vs 52w high
    roll_max = close.rolling(252, min_periods=1).max()
    feat["drawdown"] = close / roll_max - 1

    return feat.replace([np.inf, -np.inf], np.nan)


def build_dataset(data: dict, horizon: int) -> dict:
    print("\n" + "═"*60)
    print(f"  🔧  FEATURES  (horizon = {horizon}j ≈ 6 mois)")
    print("═"*60)

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
        print(f"  📊  {name:8s} | {len(X):5d} obs | {X.shape[1]} features")

    return datasets


# ════════════════════════════════════════════════════════════════════
#  3. ENTRAÎNEMENT + BACKTESTING (TimeSeriesSplit robuste)
# ════════════════════════════════════════════════════════════════════

def safe_n_splits(n_samples: int, max_splits: int = 5,
                  min_train: int = 200) -> int:
    """Calcule un nombre de folds safe selon la taille des données."""
    possible = (n_samples - min_train) // max(min_train // max_splits, 1)
    return max(2, min(possible, max_splits))


def train_and_evaluate(datasets: dict) -> dict:
    print("\n" + "═"*60)
    print("  🏋️   ENTRAÎNEMENT + VALIDATION CROISÉE TEMPORELLE")
    print("═"*60)

    results = {}

    for name, ds in datasets.items():
        X, y_dir, y_ret = ds["X"], ds["y_dir"], ds["y_ret"]
        n = len(X)
        n_splits = safe_n_splits(n)
        print(f"\n  ▶  {name}  ({n} obs, {n_splits} folds)")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # ── Classifieur ──────────────────────────────────────────
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        acc_scores = []
        all_y_true, all_y_pred = [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            y_tr = y_dir.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y_dir.iloc[test_idx]

            if len(y_tr.unique()) < 2 or len(X_tr) < 50:
                continue

            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc_scores.append(accuracy_score(y_te, y_pred))
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())

        clf.fit(X, y_dir)  # Modèle final sur tout

        # ── Régresseur ───────────────────────────────────────────
        reg = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        mae_scores, r2_scores = [], []
        all_r_true, all_r_pred = [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            r_tr = y_ret.iloc[train_idx]
            X_te = X.iloc[test_idx]
            r_te = y_ret.iloc[test_idx]

            if len(X_tr) < 50:
                continue

            reg.fit(X_tr, r_tr)
            r_pred = reg.predict(X_te)
            mae_scores.append(mean_absolute_error(r_te, r_pred))
            r2_scores.append(r2_score(r_te, r_pred))
            all_r_true.extend(r_te.tolist())
            all_r_pred.extend(r_pred.tolist())

        reg.fit(X, y_ret)

        # Top features
        feat_imp    = pd.Series(clf.feature_importances_, index=X.columns)
        top_features = feat_imp.nlargest(10)

        acc_mean = float(np.mean(acc_scores)) if acc_scores else 0.5
        mae_mean = float(np.mean(mae_scores)) if mae_scores else 0.0
        r2_mean  = float(np.mean(r2_scores))  if r2_scores  else 0.0

        print(f"     Accuracy direction  : {acc_mean:.1%}  (référence = 50%)")
        print(f"     MAE variation (%)   : ±{mae_mean:.1f}%")
        print(f"     R² régression       : {r2_mean:.3f}")

        results[name] = {
            "clf": clf, "reg": reg,
            "X": X, "y_dir": y_dir, "y_ret": y_ret,
            "acc_mean": acc_mean, "mae_mean": mae_mean, "r2_mean": r2_mean,
            "top_features": top_features,
            "all_y_true": all_y_true, "all_y_pred": all_y_pred,
            "all_r_true": all_r_true, "all_r_pred": all_r_pred,
        }

    return results


# ════════════════════════════════════════════════════════════════════
#  4. PRÉDICTIONS ACTUELLES
# ════════════════════════════════════════════════════════════════════

def predict_current(results: dict, data: dict) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("  🔮  PRÉDICTIONS → Dans ~6 mois")
    print("═"*60)

    preds = []
    for name, res in results.items():
        clf = res["clf"]
        reg = res["reg"]
        X   = res["X"]

        last = X.iloc[[-1]]
        dir_pred  = int(clf.predict(last)[0])
        dir_proba = clf.predict_proba(last)[0]
        pct_pred  = float(reg.predict(last)[0])

        label = "📈 HAUSSE" if dir_pred == 1 else "📉 BAISSE"
        conf  = dir_proba[dir_pred] * 100
        sign  = "+" if pct_pred > 0 else ""

        last_price = float(data[name]["Close"].iloc[-1]) if name in data else None
        est_price  = last_price * (1 + pct_pred / 100) if last_price else None

        print(f"\n  {name:8s}  →  {label}  (confiance {conf:.0f}%)")
        print(f"             Variation estimée : {sign}{pct_pred:.1f}%")
        if last_price:
            print(f"             Prix actuel       : {last_price:.2f} €")
            print(f"             Prix estimé +6m   : {est_price:.2f} €")

        preds.append({
            "ETF":              name,
            "Direction":        "HAUSSE" if dir_pred == 1 else "BAISSE",
            "Confiance (%)":    round(conf, 1),
            "Variation est.%":  round(pct_pred, 2),
            "Accuracy CV":      f"{res['acc_mean']:.1%}",
            "MAE CV (%)":       round(res["mae_mean"], 2),
            "Prix actuel (€)":  round(last_price, 2) if last_price else "N/A",
            "Prix estimé (€)":  round(est_price, 2)  if est_price  else "N/A",
        })

    return pd.DataFrame(preds)


# ════════════════════════════════════════════════════════════════════
#  5. BACKTESTING STRATÉGIE
# ════════════════════════════════════════════════════════════════════

def backtest_strategy(results: dict, data: dict) -> dict:
    print("\n" + "═"*60)
    print("  📈  BACKTESTING — Signal RF vs Buy & Hold")
    print("═"*60)

    bt_results = {}
    for name, res in results.items():
        if name not in data:
            continue

        X     = res["X"]
        y_dir = res["y_dir"]
        n     = len(X)
        n_splits = safe_n_splits(n)
        tscv  = TimeSeriesSplit(n_splits=n_splits)

        oos_preds = pd.Series(np.nan, index=X.index)

        for train_idx, test_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            y_tr = y_dir.iloc[train_idx]
            X_te = X.iloc[test_idx]

            if len(y_tr.unique()) < 2 or len(X_tr) < 50:
                continue

            m = RandomForestClassifier(
                n_estimators=100, max_depth=8,
                min_samples_leaf=20, n_jobs=-1,
                random_state=RANDOM_STATE
            )
            m.fit(X_tr, y_tr)
            oos_preds.iloc[test_idx] = m.predict(X_te)

        oos_preds = oos_preds.dropna()
        if len(oos_preds) == 0:
            continue

        close     = data[name]["Close"]
        daily_ret = close.pct_change().reindex(oos_preds.index)
        strat_ret = daily_ret * oos_preds

        cum_bh    = (1 + daily_ret).cumprod()
        cum_strat = (1 + strat_ret).cumprod()

        total_bh    = float(cum_bh.iloc[-1] - 1) * 100
        total_strat = float(cum_strat.iloc[-1] - 1) * 100

        sharpe = lambda r: float((r.mean() / (r.std() + 1e-9)) * np.sqrt(252))
        maxdd  = lambda c: float(((c - c.cummax()) / c.cummax()).min()) * 100

        sharpe_bh    = sharpe(daily_ret)
        sharpe_strat = sharpe(strat_ret)
        mdd_bh       = maxdd(cum_bh)
        mdd_strat    = maxdd(cum_strat)

        print(f"\n  {name}")
        print(f"     Buy & Hold   : {total_bh:+.1f}%  | Sharpe {sharpe_bh:.2f}  | MaxDD {mdd_bh:.1f}%")
        print(f"     Signal RF    : {total_strat:+.1f}%  | Sharpe {sharpe_strat:.2f}  | MaxDD {mdd_strat:.1f}%")

        bt_results[name] = {
            "cum_bh": cum_bh, "cum_strat": cum_strat,
            "total_bh": total_bh, "total_strat": total_strat,
            "sharpe_bh": sharpe_bh, "sharpe_strat": sharpe_strat,
            "mdd_bh": mdd_bh, "mdd_strat": mdd_strat,
        }

    return bt_results


# ════════════════════════════════════════════════════════════════════
#  6. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════

PALETTE = ["#00D4FF", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA"]
BG      = "#0D1117"
BG2     = "#161B22"
BORDER  = "#30363D"


def _style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def plot_all(results, bt_results, pred_df, data, save_dir="."):
    plt.style.use("dark_background")
    cmap = {n: PALETTE[i] for i, n in enumerate(results.keys())}
    os.makedirs(save_dir, exist_ok=True)

    # ── Fig 1 : Prédictions + Feature Importance ─────────────────
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    fig.suptitle(
        f"🔮  PRÉDICTIONS ETF — Horizon 6 mois | Random Forest\n"
        f"Généré le {datetime.today().strftime('%d/%m/%Y')}",
        fontsize=15, fontweight="bold", color="white", y=0.99
    )

    # Tableau résumé
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=pred_df.values.tolist(),
        colLabels=pred_df.columns.tolist(),
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(BG2)
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="#60A5FA", fontweight="bold")
        else:
            row = pred_df.iloc[r - 1]
            col = "#34D399" if row["Direction"] == "HAUSSE" else "#F87171"
            cell.set_text_props(color=col if c == 1 else "white")
    ax_t.set_title("📊  Tableau des Prédictions (Horizon ≈ 6 mois)",
                   color="#60A5FA", fontsize=11, pad=12)

    # Top features
    etfs   = list(results.keys())
    posits = [(1,0),(1,1),(1,2),(2,0),(2,1)]
    for i, name in enumerate(etfs[:5]):
        r, c = posits[i]
        ax = fig.add_subplot(gs[r, c])
        _style_ax(ax)
        top = results[name]["top_features"]
        ax.barh(range(len(top)), top.values, color=cmap[name], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=7, color="white")
        ax.set_title(f"Top Features — {name}", color=cmap[name],
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=7, color="#9CA3AF")

    # Accuracy bar
    ax_m = fig.add_subplot(gs[2, 2])
    _style_ax(ax_m)
    names = list(results.keys())
    accs  = [results[n]["acc_mean"]*100 for n in names]
    xpos  = np.arange(len(names))
    bars  = ax_m.bar(xpos, accs, color=[cmap[n] for n in names], alpha=0.85)
    ax_m.axhline(50, color="#EF4444", linestyle="--", alpha=0.8, label="50% (aléatoire)")
    ax_m.set_xticks(xpos)
    ax_m.set_xticklabels(names, rotation=20, fontsize=8, color="white")
    ax_m.set_ylabel("Accuracy (%)", color="#9CA3AF", fontsize=8)
    ax_m.set_title("Accuracy Classification", color="white", fontsize=9, fontweight="bold")
    ax_m.legend(fontsize=7)
    ax_m.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                  f"{acc:.1f}%", ha="center", fontsize=7, color="white")

    fig.savefig(f"{save_dir}/fig1_predictions.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  💾  {save_dir}/fig1_predictions.png")

    # ── Fig 2 : Backtesting ──────────────────────────────────────
    if bt_results:
        n = len(bt_results)
        fig2, axes = plt.subplots(n, 1, figsize=(16, 4*n))
        fig2.patch.set_facecolor(BG)
        if n == 1: axes = [axes]
        fig2.suptitle("📈  Backtesting — Signal RF vs Buy & Hold",
                      fontsize=13, fontweight="bold", color="white")

        for ax, (name, bt) in zip(axes, bt_results.items()):
            _style_ax(ax)
            c = cmap.get(name, "#60A5FA")
            bh = bt["cum_bh"].dropna()
            st = bt["cum_strat"].dropna()
            ax.plot(bh.index, bh.values, color="#94A3B8",
                    linewidth=1.2, alpha=0.7,
                    label=f"Buy & Hold  ({bt['total_bh']:+.1f}%)")
            ax.plot(st.index, st.values, color=c, linewidth=1.8,
                    label=f"Signal RF   ({bt['total_strat']:+.1f}%)")
            ax.fill_between(st.index, 1, st.values, alpha=0.08, color=c)
            ax.axhline(1, color="white", linestyle="--", alpha=0.3)
            ax.set_title(
                f"{name}  |  Sharpe RF: {bt['sharpe_strat']:.2f}  vs  "
                f"BH: {bt['sharpe_bh']:.2f}  |  MaxDD RF: {bt['mdd_strat']:.1f}%",
                color=c, fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            ax.set_ylabel("Perf. cumulée", color="#9CA3AF", fontsize=8)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{(x-1)*100:+.0f}%"))

        fig2.tight_layout()
        fig2.savefig(f"{save_dir}/fig2_backtest.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print(f"  💾  {save_dir}/fig2_backtest.png")

    # ── Fig 3 : Historique des prix ───────────────────────────────
    if data:
        n = len(data)
        fig3, axes = plt.subplots(n, 1, figsize=(16, 3.5*n))
        fig3.patch.set_facecolor(BG)
        if n == 1: axes = [axes]
        fig3.suptitle("📉  Historique des Prix + MAs + Bollinger",
                      fontsize=13, fontweight="bold", color="white")

        for ax, (name, df) in zip(axes, data.items()):
            _style_ax(ax)
            c = cmap.get(name, "#60A5FA")
            cl = df["Close"]
            ax.plot(cl.index, cl.values, color=c, linewidth=1.1, label="Prix")
            ax.plot(cl.index, cl.rolling(50).mean(), color="#F59E0B",
                    linewidth=1, alpha=0.8, label="MA50")
            ax.plot(cl.index, cl.rolling(200).mean(), color="#EF4444",
                    linewidth=1, alpha=0.8, label="MA200")
            ma20 = cl.rolling(20).mean()
            sd20 = cl.rolling(20).std()
            ax.fill_between(cl.index, ma20-2*sd20, ma20+2*sd20,
                            alpha=0.07, color=c, label="Bollinger 2σ")
            ax.set_title(name, color=c, fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_ylabel("Prix (€)", color="#9CA3AF", fontsize=8)

        fig3.tight_layout()
        fig3.savefig(f"{save_dir}/fig3_prix.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig3)
        print(f"  💾  {save_dir}/fig3_prix.png")

    # ── Fig 4 : Corrélations entre ETFs ──────────────────────────
    if len(data) > 1:
        rets = pd.DataFrame({n: d["Close"].pct_change() for n, d in data.items()})
        corr = rets.corr()

        fig4, ax = plt.subplots(figsize=(8, 6))
        fig4.patch.set_facecolor(BG)
        _style_ax(ax)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns_ok = False
        try:
            import seaborn as sns
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        vmin=-1, vmax=1, ax=ax, mask=mask,
                        linewidths=0.5, linecolor=BORDER,
                        annot_kws={"size": 10})
            sns_ok = True
        except ImportError:
            im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, color="white")
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.columns, color="white")
            plt.colorbar(im, ax=ax)

        ax.set_title("🔗  Matrice de Corrélation (rendements)",
                     color="white", fontsize=11, fontweight="bold")
        fig4.tight_layout()
        fig4.savefig(f"{save_dir}/fig4_correlations.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig4)
        print(f"  💾  {save_dir}/fig4_correlations.png")


# ════════════════════════════════════════════════════════════════════
#  7. EXPORT EXCEL
# ════════════════════════════════════════════════════════════════════

def export_excel(pred_df, results, bt_results, save_dir="."):
    try:
        path = f"{save_dir}/rapport_predictions.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pred_df.to_excel(writer, sheet_name="Prédictions", index=False)

            rows = [{
                "ETF": n,
                "Accuracy CV": f"{r['acc_mean']:.1%}",
                "MAE CV (%)": f"{r['mae_mean']:.2f}",
                "R²": f"{r['r2_mean']:.3f}",
                "Nb obs": len(r["X"]),
                "Nb features": r["X"].shape[1],
            } for n, r in results.items()]
            pd.DataFrame(rows).to_excel(writer, sheet_name="Métriques", index=False)

            if bt_results:
                bt_rows = [{
                    "ETF": n,
                    "BH Total (%)": f"{b['total_bh']:.1f}",
                    "RF Total (%)": f"{b['total_strat']:.1f}",
                    "Sharpe BH": f"{b['sharpe_bh']:.2f}",
                    "Sharpe RF": f"{b['sharpe_strat']:.2f}",
                    "MaxDD BH (%)": f"{b['mdd_bh']:.1f}",
                    "MaxDD RF (%)": f"{b['mdd_strat']:.1f}",
                } for n, b in bt_results.items()]
                pd.DataFrame(bt_rows).to_excel(writer, sheet_name="Backtesting", index=False)

        print(f"  💾  {path}")
    except Exception as e:
        print(f"  ⚠️  Export Excel ignoré : {e}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("\n╔" + "═"*58 + "╗")
    print("║   ALGORITHME PRÉDICTIF ETF — RANDOM FOREST              ║")
    print("║   Horizon 6 mois | 5 ETFs PEA | Backtesting complet     ║")
    print("╚" + "═"*58 + "╝")

    OUT = "output_etf_predictor"
    os.makedirs(OUT, exist_ok=True)

    data      = download_data(TICKERS, years=TRAIN_YEARS)
    datasets  = build_dataset(data, horizon=HORIZON_JOURS)
    results   = train_and_evaluate(datasets)
    pred_df   = predict_current(results, data)
    bt_results = backtest_strategy(results, data)

    print("\n" + "═"*60)
    print("  📊  GÉNÉRATION DES GRAPHIQUES")
    print("═"*60)
    plot_all(results, bt_results, pred_df, data, save_dir=OUT)

    os.makedirs(f"{OUT}/models", exist_ok=True)
    for name, res in results.items():
        joblib.dump(res["clf"], f"{OUT}/models/{name}_clf.pkl")
        joblib.dump(res["reg"], f"{OUT}/models/{name}_reg.pkl")
    print(f"  💾  Modèles : {OUT}/models/")

    export_excel(pred_df, results, bt_results, save_dir=OUT)

    print("\n" + "═"*60)
    print("  ✅  PRÉDICTIONS FINALES")
    print("═"*60)
    print(pred_df.to_string(index=False))
    print("\n  ⚠️  Données simulées si Yahoo Finance indisponible.")
    print("  ⚠️  Ce script est éducatif — pas un conseil en investissement.")
    print("═"*60 + "\n")

    return pred_df, results, bt_results


if __name__ == "__main__":
    main()
