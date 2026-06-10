import streamlit as st
import pandas as pd
import os
import importlib
import sys

# Configuration de la page Streamlit
st.set_page_config(page_title="Prédictions ETF", page_icon="🔮", layout="wide")
st.title("🔮 Algorithme Prédictif ETF (Random Forest)")
st.write("Ce module utilise une intelligence artificielle pour anticiper les tendances à 6 mois.")

# Force le rechargement du module à chaque exécution (évite le cache)
if "etf_predictor" in sys.modules:
    del sys.modules["etf_predictor"]

try:
    etf_predictor = importlib.import_module("etf_predictor")
    MODELS_OK = True
except ModuleNotFoundError:
    st.error("❌ Le fichier 'etf_predictor.py' est introuvable à la racine de ton GitHub.")
    MODELS_OK = False

if MODELS_OK:
    if st.button("🚀 Calculer et afficher les prédictions en direct"):
        with st.spinner("📥 Chargement des données et entraînement de l'IA... (environ 1 minute)"):
            try:
                TICKERS = etf_predictor.TICKERS
                HORIZON_JOURS = etf_predictor.HORIZON_JOURS
                TRAIN_YEARS = etf_predictor.TRAIN_YEARS
                OUTPUT_DIR = "output_etf_predictor"

                data = etf_predictor.download_data(TICKERS, years=TRAIN_YEARS)
                datasets = etf_predictor.build_dataset(data, horizon=HORIZON_JOURS)
                results = etf_predictor.train_and_evaluate(datasets)
                pred_df = etf_predictor.predict_current(results, data)
                bt_results = etf_predictor.backtest_strategy(results, data)
                etf_predictor.plot_all(results, bt_results, pred_df, data, save_dir=OUTPUT_DIR)

                st.success("✅ Analyses complétées avec succès !")

                st.write("---")
                st.subheader("📊 Tableau Synthétique des Prédictions (Horizon ~6 mois)")

                def color_direction(val):
                    if val == "HAUSSE":
                        return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
                    elif val == "BAISSE":
                        return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
                    return ""

                st.dataframe(
                    pred_df.style.map(color_direction, subset=["Direction"]),
                    use_container_width=True
                )

                st.write("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Backtesting de la Stratégie")
                    path_fig2 = os.path.join(OUTPUT_DIR, "fig2_backtest.png")
                    if os.path.exists(path_fig2):
                        st.image(path_fig2, caption="Stratégie RF vs Buy & Hold")

                with col2:
                    st.subheader("📉 Analyse Technique")
                    path_fig3 = os.path.join(OUTPUT_DIR, "fig3_prix.png")
                    if os.path.exists(path_fig3):
                        st.image(path_fig3, caption="Historique des cours et Bandes de Bollinger")

                # Graphiques supplémentaires
                st.write("---")
                col3, col4 = st.columns(2)

                with col3:
                    path_fig1 = os.path.join(OUTPUT_DIR, "fig1_predictions.png")
                    if os.path.exists(path_fig1):
                        st.image(path_fig1, caption="Prédictions détaillées + Feature Importance")

                with col4:
                    path_fig4 = os.path.join(OUTPUT_DIR, "fig4_correlations.png")
                    if os.path.exists(path_fig4):
                        st.image(path_fig4, caption="Matrice de corrélation entre ETFs")

                # Note sur les données
                st.info("ℹ️ Si Yahoo Finance est indisponible, les prédictions sont basées sur des données simulées réalistes.")

            except Exception as e:
                st.error(f"❌ Une erreur technique est survenue : {e}")
                st.exception(e)  # Affiche la trace complète pour debug
