"""
╔══════════════════════════════════════════════════════════════════════╗
║         ALGORITHME PRÉDICTIF ETF — RANDOM FOREST                    ║
║         Horizon : 6 mois | Tickers : PAASI, PINR, PUST,            ║
║                            EFENSE (DEFS), IWSC                      ║
║                                                                      ║
║  UTILISATION :                                                       ║
║    pip install yfinance scikit-learn pandas numpy matplotlib seaborn ║
║    python etf_predictor.py                                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import os

# ─── Machine Learning ───────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              mean_absolute_error, r2_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ─── Data ───────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False
    print("⚠️  yfinance non installé. Installez-le avec : pip install yfinance")

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

TICKERS = {
    "PAASI": "PANX.PA",   # Amundi MSCI AC Asia Pacific (Actif et stable sur Yahoo Finance)
    "PINDIA": "PINR.PA",   # Amundi PEA MSCI India
    "PUST":  "PUST.PA",    # Lyxor PEA Nasdaq-100
    "EFENSE": "GUARD.PA",  # BNP Paribas Easy Bloomberg Europe Defense
    "IWSC":  "WPEA.PA",    # iShares MSCI World Swap PEA
}

HORIZON_JOURS = 126          # ≈ 6 mois de trading (21j × 6)
TRAIN_YEARS   = 5            # années d'historique pour l'entraînement
N_ESTIMATORS  = 500          # arbres dans la forêt
N_SPLITS      = 5            # folds pour la validation croisée temporelle
RANDOM_STATE  = 42

# ════════════════════════════════════════════════════════════════════
#  1. TÉLÉCHARGEMENT DES DONNÉES
# ════════════════════════════════════════════════════════════════════

def download_data(tickers_dict: dict, years: int = 5) -> dict:
    """Télécharge l'historique de prix depuis Yahoo Finance."""
    print("\n" + "═"*60)
    print("  📥  TÉLÉCHARGEMENT DES DONNÉES")
    print("═"*60)

    data = {}
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    for name, ticker in tickers_dict.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  ⚠️  {name} ({ticker}) : aucune donnée trouvée")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c
                          for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            data[name] = df
            print(f"  ✅  {name:8s} | {len(df):5d} jours | "
                  f"{df.index[0].date()} → {df.index[-1].date()}")
        except Exception as e:
            print(f"  ❌  {name} ({ticker}) : erreur — {e}")

    return data


# ════════════════════════════════════════════════════════════════════
#  2. INGÉNIERIE DES FEATURES
# ════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Calcule ~40 features techniques sur une série OHLCV.
    Toutes calculées uniquement sur le passé (pas de data leakage).
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Rendements passés ──────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 21, 63]:
        feat[f"ret_{lag}d"] = close.pct_change(lag)

    # ── Moyennes mobiles ──────────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        ma = close.rolling(w).mean()
        feat[f"ma_{w}"] = close / ma - 1   # distance relative à la MA

    # ── Volatilité réalisée ───────────────────────────────────────
    for w in [10, 21, 63]:
        feat[f"vol_{w}d"] = close.pct_change().rolling(w).std() * np.sqrt(252)

    # ── RSI ───────────────────────────────────────────────────────
    for period in [14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # ── MACD ──────────────────────────────────────────────────────
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    feat["macd"]        = macd / close
    feat["macd_signal"] = signal / close
    feat["macd_hist"]   = (macd - signal) / close

    # ── Bandes de Bollinger ────────────────────────────────────────
    for w in [20, 50]:
        ma_bb  = close.rolling(w).mean()
        std_bb = close.rolling(w).std()
        feat[f"bb_upper_{w}"] = (close - (ma_bb + 2 * std_bb)) / close
        feat[f"bb_lower_{w}"] = (close - (ma_bb - 2 * std_bb)) / close
        feat[f"bb_width_{w}"] = 4 * std_bb / ma_bb

    # ── ATR (Average True Range) ──────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close
    feat["atr_21"] = tr.rolling(21).mean() / close

    # ── Volume ────────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20).mean()
    feat["vol_ratio_20"] = vol / (vol_ma20 + 1)
    feat["vol_ratio_5"]  = vol / (vol.rolling(5).mean() + 1)

    # ── Saisonnalité ──────────────────────────────────────────────
    feat["day_of_week"]  = df.index.dayofweek / 4.0
    feat["month"]        = df.index.month / 12.0
    feat["quarter"]      = df.index.quarter / 4.0

    # ── Momentum de momentum ─────────────────────────────────────
    feat["mom_accel"] = close.pct_change(21) - close.pct_change(63)

    # ── Drawdown glissant ─────────────────────────────────────────
    roll_max = close.rolling(252, min_periods=1).max()
    feat["drawdown"] = close / roll_max - 1

    return feat.replace([np.inf, -np.inf], np.nan)


def build_dataset(data: dict, horizon: int) -> dict:
    """
    Construit X (features) et y (cibles) pour chaque ETF.
    Cibles :
      - direction  : 1 si Close(t+horizon) > Close(t), sinon 0
      - pct_change : rendement sur l'horizon en %
    """
    print("\n" + "═"*60)
    print(f"  🔧  CONSTRUCTION DES FEATURES  (horizon = {horizon}j ≈ 6 mois)")
    print("═"*60)

    datasets = {}
    for name, df in data.items():
        feat = compute_features(df, name)

        # Cibles (horizon j dans le futur)
        future_ret = df["Close"].pct_change(horizon).shift(-horizon) * 100
        direction  = (future_ret > 0).astype(int)

        # Combinaison
        combined = feat.copy()
        combined["__target_ret__"]  = future_ret
        combined["__target_dir__"]  = direction
        combined = combined.dropna()

        # Sépare X et y
        X = combined.drop(columns=["__target_ret__", "__target_dir__"])
        y_dir = combined["__target_dir__"]
        y_ret = combined["__target_ret__"]

        datasets[name] = {
            "X":     X,
            "y_dir": y_dir,
            "y_ret": y_ret,
            "dates": combined.index
        }
        print(f"  📊  {name:8s} | {len(X):5d} observations | "
              f"{X.shape[1]} features")

    return datasets


# ════════════════════════════════════════════════════════════════════
#  3. ENTRAÎNEMENT ET ÉVALUATION
# ════════════════════════════════════════════════════════════════════

def train_and_evaluate(datasets: dict) -> dict:
    """
    Pour chaque ETF :
      - Entraîne un RF classifieur  (direction)
      - Entraîne un RF régresseur   (% de variation)
      - Évalue via TimeSeriesSplit  (backtesting correct sans data leakage)
    """
    print("\n" + "═"*60)
    print("  🏋️   ENTRAÎNEMENT + BACKTESTING (TimeSeriesSplit)")
    print("═"*60)

    results = {}
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    for name, ds in datasets.items():
        X, y_dir, y_ret = ds["X"], ds["y_dir"], ds["y_ret"]
        print(f"\n  ▶  {name}")

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

        acc_scores, prec_scores = [], []
        all_y_true, all_y_pred = [], []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_dir.iloc[train_idx], y_dir.iloc[test_idx]

            if len(y_tr.unique()) < 2:
                continue

            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            acc  = accuracy_score(y_te, y_pred)
            acc_scores.append(acc)
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())

        # Modèle final sur tout le dataset
        clf.fit(X, y_dir)

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

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            r_tr, r_te = y_ret.iloc[train_idx], y_ret.iloc[test_idx]

            reg.fit(X_tr, r_tr)
            r_pred = reg.predict(X_te)

            mae_scores.append(mean_absolute_error(r_te, r_pred))
            r2_scores.append(r2_score(r_te, r_pred))
            all_r_true.extend(r_te.tolist())
            all_r_pred.extend(r_pred.tolist())

        reg.fit(X, y_ret)

        # ── Importance des features ───────────────────────────────
        feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
        top_features = feat_imp.nlargest(10)

        # ── Métriques ────────────────────────────────────────────
        acc_mean  = np.mean(acc_scores)
        mae_mean  = np.mean(mae_scores)
        r2_mean   = np.mean(r2_scores)

        print(f"     Accuracy (direction)  : {acc_mean:.1%}  "
              f"(aléatoire = 50%)")
        print(f"     MAE (variation %)     : ±{mae_mean:.1f}%")
        print(f"     R²  (régression)      : {r2_mean:.3f}")

        results[name] = {
            "clf":          clf,
            "reg":          reg,
            "X":            X,
            "y_dir":        y_dir,
            "y_ret":        y_ret,
            "dates":        ds["dates"],
            "acc_scores":   acc_scores,
            "mae_scores":   mae_scores,
            "r2_scores":    r2_scores,
            "acc_mean":     acc_mean,
            "mae_mean":     mae_mean,
            "r2_mean":      r2_mean,
            "top_features": top_features,
            "all_y_true":   all_y_true,
            "all_y_pred":   all_y_pred,
            "all_r_true":   all_r_true,
            "all_r_pred":   all_r_pred,
            "clf_full":     clf,
            "reg_full":     reg,
        }

    return results


# ════════════════════════════════════════════════════════════════════
#  4. PRÉDICTION ACTUELLE (J → J+126)
# ════════════════════════════════════════════════════════════════════

def predict_current(results: dict, data: dict) -> pd.DataFrame:
    """Génère la prédiction pour aujourd'hui → +6 mois."""
    print("\n" + "═"*60)
    print(f"  🔮  PRÉDICTIONS ACTUELLES (horizon ≈ 6 mois)")
    print("═"*60)

    preds = []
    today = datetime.today().date()
    target_date = today + timedelta(days=HORIZON_JOURS * 1.4)  # ≈ 6 mois

    for name, res in results.items():
        clf = res["clf"]
        reg = res["reg"]
        X   = res["X"]

        # Dernière ligne de features connue
        last_features = X.iloc[[-1]]
        last_date     = X.index[-1].date()

        direction_pred  = clf.predict(last_features)[0]
        direction_proba = clf.predict_proba(last_features)[0]
        pct_pred        = reg.predict(last_features)[0]

        label = "📈 HAUSSE" if direction_pred == 1 else "📉 BAISSE"
        conf  = direction_proba[direction_pred] * 100
        sign  = "+" if pct_pred > 0 else ""

        print(f"\n  {name:8s}  →  {label}  ({conf:.0f}% confiance)")
        print(f"             Variation estimée : {sign}{pct_pred:.1f}%")
        print(f"             Basé sur données du : {last_date}")

        # Prix actuel
        if name in data:
            last_price = float(data[name]["Close"].iloc[-1])
            est_price  = last_price * (1 + pct_pred / 100)
            print(f"             Prix actuel  : {last_price:.2f} €")
            print(f"             Prix estimé  : {est_price:.2f} €  "
                  f"(dans ~6 mois)")
        else:
            last_price = None
            est_price  = None

        preds.append({
            "ETF":            name,
            "Direction":      "HAUSSE" if direction_pred == 1 else "BAISSE",
            "Confiance (%)":  round(conf, 1),
            "Variation est. (%)": round(pct_pred, 2),
            "Accuracy CV (%)":    round(res["acc_mean"] * 100, 1),
            "MAE CV (%)":         round(res["mae_mean"], 2),
            "Prix actuel (€)":    round(last_price, 2) if last_price else "N/A",
            "Prix estimé (€)":    round(est_price, 2)  if est_price  else "N/A",
        })

    return pd.DataFrame(preds)


# ════════════════════════════════════════════════════════════════════
#  5. BACKTESTING SIMULÉ : STRATÉGIE "SUIVRE LE SIGNAL"
# ════════════════════════════════════════════════════════════════════

def backtest_strategy(results: dict, data: dict) -> dict:
    """
    Stratégie simple : investir quand le modèle prédit HAUSSE,
    rester en cash sinon. Compare vs Buy & Hold.
    """
    print("\n" + "═"*60)
    print("  📈  BACKTESTING STRATÉGIE (Signal RF vs Buy & Hold)")
    print("═"*60)

    bt_results = {}

    for name, res in results.items():
        if name not in data:
            continue

        X     = res["X"]
        y_dir = res["y_dir"]
        dates = res["dates"]
        clf   = res["clf"]

        # Prédictions out-of-sample via walk-forward
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        oos_preds = pd.Series(index=X.index, dtype=float)

        for train_idx, test_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            y_tr = y_dir.iloc[train_idx]
            X_te = X.iloc[test_idx]

            if len(y_tr.unique()) < 2:
                continue

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            oos_preds.iloc[test_idx] = preds

        oos_preds = oos_preds.dropna()

        # Rendements journaliers
        close      = data[name]["Close"]
        daily_ret  = close.pct_change().reindex(oos_preds.index)

        # Stratégie RF : investi si signal = 1
        strat_ret  = daily_ret * oos_preds

        # Métriques
        cum_bh   = (1 + daily_ret).cumprod()
        cum_strat = (1 + strat_ret).cumprod()

        total_bh   = float(cum_bh.iloc[-1] - 1) * 100
        total_strat = float(cum_strat.iloc[-1] - 1) * 100

        # Sharpe ratio annualisé
        sharpe_strat = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(252)
        sharpe_bh    = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)

        # Max drawdown
        def max_dd(cum):
            roll_max = cum.cummax()
            dd = (cum - roll_max) / roll_max
            return float(dd.min()) * 100

        mdd_strat = max_dd(cum_strat)
        mdd_bh    = max_dd(cum_bh)

        print(f"\n  {name}")
        print(f"     Buy & Hold  : {total_bh:+.1f}%  |  "
              f"Sharpe: {sharpe_bh:.2f}  |  Max DD: {mdd_bh:.1f}%")
        print(f"     Stratégie RF: {total_strat:+.1f}%  |  "
              f"Sharpe: {sharpe_strat:.2f}  |  Max DD: {mdd_strat:.1f}%")

        bt_results[name] = {
            "cum_bh":      cum_bh,
            "cum_strat":   cum_strat,
            "total_bh":    total_bh,
            "total_strat": total_strat,
            "sharpe_bh":   sharpe_bh,
            "sharpe_strat": sharpe_strat,
            "mdd_bh":      mdd_bh,
            "mdd_strat":   mdd_strat,
        }

    return bt_results


# ════════════════════════════════════════════════════════════════════
#  6. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════

def plot_all(results: dict, bt_results: dict, pred_df: pd.DataFrame,
             data: dict, save_dir: str = "."):
    """Génère toutes les figures et les sauvegarde."""
    plt.style.use("dark_background")
    colors_etf = ["#00D4FF", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA"]
    color_map  = {n: colors_etf[i] for i, n in enumerate(results.keys())}

    os.makedirs(save_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────
    # FIG 1 : Tableau de prédictions + Feature Importance
    # ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Titre principal
    fig.suptitle(
        f"PRÉDICTIONS ETF — Horizon 6 mois  |  Random Forest\n"
        f"Généré le {datetime.today().strftime('%d/%m/%Y')}",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    # ── Sous-figure : tableau résumé ─────────────────────────────
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")

    table_data = pred_df.values.tolist()
    col_labels = pred_df.columns.tolist()

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161B22")
        cell.set_edgecolor("#30363D")
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="#60A5FA", fontweight="bold")
        else:
            row_data = pred_df.iloc[r - 1]
            if "HAUSSE" in str(row_data["Direction"]):
                cell.set_text_props(color="#34D399" if c == 1 else "white")
            else:
                cell.set_text_props(color="#F87171" if c == 1 else "white")

    ax_table.set_title("📊 Tableau des Prédictions (Horizon ≈ 6 mois)",
                        color="#60A5FA", fontsize=11, pad=10)

    # ── Sous-figures : top features par ETF ─────────────────────
    etf_names = list(results.keys())
    positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

    for i, name in enumerate(etf_names):
        if i >= len(positions):
            break
        r, c = positions[i]
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#161B22")

        top = results[name]["top_features"]
        bars = ax.barh(range(len(top)), top.values,
                       color=color_map[name], alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=7, color="white")
        ax.set_title(f"Top Features — {name}", color=color_map[name],
                     fontsize=9, fontweight="bold")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#30363D")
        ax.set_xlabel("Importance", fontsize=7, color="gray")

    # ── Sous-figure : métriques comparatives ─────────────────────
    ax_metric = fig.add_subplot(gs[2, 2])
    ax_metric.set_facecolor("#161B22")

    names   = list(results.keys())
    accs    = [results[n]["acc_mean"] * 100 for n in names]
    x_pos   = np.arange(len(names))

    bars = ax_metric.bar(x_pos, accs,
                         color=[color_map[n] for n in names], alpha=0.8)
    ax_metric.axhline(50, color="red", linestyle="--", alpha=0.7,
                      label="Aléatoire (50%)")
    ax_metric.set_xticks(x_pos)
    ax_metric.set_xticklabels(names, rotation=15, fontsize=8, color="white")
    ax_metric.set_ylabel("Accuracy (%)", color="gray", fontsize=8)
    ax_metric.set_title("Accuracy (Classification)", color="white",
                         fontsize=9, fontweight="bold")
    ax_metric.legend(fontsize=7)
    ax_metric.tick_params(colors="gray")
    ax_metric.spines[:].set_color("#30363D")
    ax_metric.set_ylim(0, 100)

    for bar, acc in zip(bars, accs):
        ax_metric.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 1,
                       f"{acc:.1f}%", ha="center", va="bottom",
                       fontsize=7, color="white")

    fig.savefig(f"{save_dir}/fig1_predictions.png",
                dpi=150, bbox_inches="tight",
                facecolor="#0D1117")
    print(f"\n  💾  Sauvegardé : {save_dir}/fig1_predictions.png")

    # ────────────────────────────────────────────────────────────────
    # FIG 2 : Backtesting — Stratégie RF vs Buy & Hold
    # ────────────────────────────────────────────────────────────────
    if bt_results:
        n_etf = len(bt_results)
        fig2, axes = plt.subplots(n_etf, 1, figsize=(16, 4 * n_etf))
        fig2.patch.set_facecolor("#0D1117")
        if n_etf == 1:
            axes = [axes]

        fig2.suptitle("📈 BACKTESTING — Stratégie Random Forest vs Buy & Hold",
                      fontsize=14, fontweight="bold", color="white")

        for ax, (name, bt) in zip(axes, bt_results.items()):
            ax.set_facecolor("#161B22")

            cum_bh    = bt["cum_bh"].dropna()
            cum_strat = bt["cum_strat"].dropna()

            ax.plot(cum_bh.index, cum_bh.values,
                    color="#94A3B8", alpha=0.7, linewidth=1.2,
                    label=f"Buy & Hold  ({bt['total_bh']:+.1f}%)")
            ax.plot(cum_strat.index, cum_strat.values,
                    color=color_map.get(name, "#60A5FA"),
                    linewidth=1.8,
                    label=f"RF Signal   ({bt['total_strat']:+.1f}%)")
            ax.fill_between(cum_strat.index, 1, cum_strat.values,
                            alpha=0.1,
                            color=color_map.get(name, "#60A5FA"))

            ax.axhline(1, color="white", linestyle="--", alpha=0.3)
            ax.set_title(
                f"{name}  |  "
                f"Sharpe RF: {bt['sharpe_strat']:.2f}  vs  "
                f"Sharpe BH: {bt['sharpe_bh']:.2f}  |  "
                f"Max DD RF: {bt['mdd_strat']:.1f}%",
                color=color_map.get(name, "white"), fontsize=9
            )
            ax.legend(fontsize=8, loc="upper left")
            ax.tick_params(colors="gray")
            ax.spines[:].set_color("#30363D")
            ax.set_ylabel("Performance cumulée", color="gray", fontsize=8)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{(x-1)*100:+.0f}%"))

        fig2.tight_layout()
        fig2.savefig(f"{save_dir}/fig2_backtest.png",
                     dpi=150, bbox_inches="tight",
                     facecolor="#0D1117")
        print(f"  💾  Sauvegardé : {save_dir}/fig2_backtest.png")

    # ────────────────────────────────────────────────────────────────
    # FIG 3 : Historique des prix + signal de trading
    # ────────────────────────────────────────────────────────────────
    if data:
        n_etf = len(data)
        fig3, axes = plt.subplots(n_etf, 1, figsize=(16, 3.5 * n_etf))
        fig3.patch.set_facecolor("#0D1117")
        if n_etf == 1:
            axes = [axes]

        fig3.suptitle("📉 HISTORIQUE DES PRIX + Moyennes Mobiles",
                      fontsize=14, fontweight="bold", color="white")

        for ax, (name, df) in zip(axes, data.items()):
            ax.set_facecolor("#161B22")
            close = df["Close"]
            c     = color_map.get(name, "#60A5FA")

            ax.plot(close.index, close.values,
                    color=c, linewidth=1.2, alpha=0.9, label="Prix")
            ax.plot(close.index,
                    close.rolling(50).mean(),
                    color="#F59E0B", linewidth=1, alpha=0.7, label="MA50")
            ax.plot(close.index,
                    close.rolling(200).mean(),
                    color="#EF4444", linewidth=1, alpha=0.7, label="MA200")
            ax.fill_between(close.index,
                            close.rolling(20).mean() - 2*close.rolling(20).std(),
                            close.rolling(20).mean() + 2*close.rolling(20).std(),
                            alpha=0.05, color=c, label="Bollinger 2σ")

            ax.set_title(f"{name}", color=c, fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left")
            ax.tick_params(colors="gray")
            ax.spines[:].set_color("#30363D")
            ax.set_ylabel("Prix (€)", color="gray", fontsize=8)

        fig3.tight_layout()
        fig3.savefig(f"{save_dir}/fig3_prix.png",
                     dpi=150, bbox_inches="tight",
                     facecolor="#0D1117")
        print(f"  💾  Sauvegardé : {save_dir}/fig3_prix.png")

    plt.close("all")


# ════════════════════════════════════════════════════════════════════
#  7. SAUVEGARDE DES MODÈLES
# ════════════════════════════════════════════════════════════════════

def save_models(results: dict, save_dir: str = "models"):
    """Sauvegarde les modèles entraînés pour réutilisation."""
    os.makedirs(save_dir, exist_ok=True)
    for name, res in results.items():
        joblib.dump(res["clf"],
                    f"{save_dir}/{name}_classifier.pkl")
        joblib.dump(res["reg"],
                    f"{save_dir}/{name}_regressor.pkl")
        res["X"].iloc[[-1]].to_parquet(
            f"{save_dir}/{name}_last_features.parquet")
    print(f"\n  💾  Modèles sauvegardés dans ./{save_dir}/")


# ════════════════════════════════════════════════════════════════════
#  8. RAPPORT EXCEL
# ════════════════════════════════════════════════════════════════════

def export_excel(pred_df: pd.DataFrame, results: dict,
                 bt_results: dict, save_dir: str = "."):
    """Exporte un rapport Excel complet."""
    try:
        import openpyxl  # noqa
        path = f"{save_dir}/rapport_predictions.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pred_df.to_excel(writer, sheet_name="Prédictions", index=False)

            # Métriques de performance
            rows = []
            for name, res in results.items():
                rows.append({
                    "ETF":           name,
                    "Accuracy CV":   f"{res['acc_mean']:.1%}",
                    "MAE CV (%)":    f"{res['mae_mean']:.2f}",
                    "R² CV":         f"{res['r2_mean']:.3f}",
                    "Nb observations": len(res["X"]),
                    "Nb features":   res["X"].shape[1],
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name="Métriques",
                                        index=False)

            # Backtesting
            if bt_results:
                bt_rows = []
                for name, bt in bt_results.items():
                    bt_rows.append({
                        "ETF":             name,
                        "BH Total (%)":    f"{bt['total_bh']:.1f}",
                        "RF Total (%)":    f"{bt['total_strat']:.1f}",
                        "Sharpe BH":       f"{bt['sharpe_bh']:.2f}",
                        "Sharpe RF":       f"{bt['sharpe_strat']:.2f}",
                        "Max DD BH (%)":   f"{bt['mdd_bh']:.1f}",
                        "Max DD RF (%)":   f"{bt['mdd_strat']:.1f}",
                    })
                pd.DataFrame(bt_rows).to_excel(writer,
                                               sheet_name="Backtesting",
                                               index=False)

        print(f"  💾  Rapport Excel : {path}")
    except Exception as e:
        print(f"  ⚠️  Export Excel ignoré : {e}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   ALGORITHME PRÉDICTIF ETF — RANDOM FOREST              ║")
    print("║   Horizon : 6 mois | 5 ETFs PEA Euronext Paris          ║")
    print("╚" + "═"*58 + "╝")

    OUTPUT_DIR = "output_etf_predictor"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Données
    data = download_data(TICKERS, years=TRAIN_YEARS)

    if not data:
        print("\n❌  Aucune donnée disponible. Vérifiez votre connexion "
              "et les tickers.")
        return

    # 2. Features
    datasets = build_dataset(data, horizon=HORIZON_JOURS)

    # 3. Entraînement + évaluation
    results = train_and_evaluate(datasets)

    # 4. Prédictions actuelles
    pred_df = predict_current(results, data)

    # 5. Backtesting
    bt_results = backtest_strategy(results, data)

    # 6. Visualisations
    plot_all(results, bt_results, pred_df, data, save_dir=OUTPUT_DIR)

    # 7. Sauvegarde modèles
    save_models(results, save_dir=f"{OUTPUT_DIR}/models")

    # 8. Export Excel
    export_excel(pred_df, results, bt_results, save_dir=OUTPUT_DIR)

    # ── Résumé final ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  ✅  RÉSUMÉ FINAL")
    print("═"*60)
    print(pred_df.to_string(index=False))
    print("\n" + "═"*60)
    print(f"  📁  Tous les fichiers sont dans : ./{OUTPUT_DIR}/")
    print("  ⚠️  AVERTISSEMENT : Ce modèle est éducatif et ne constitue")
    print("      pas un conseil en investissement.")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
