import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import date
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# 1. Configuration de la page
st.set_page_config(page_title="Dashboard PEA", page_icon="📈", layout="wide")

# 2. Menu de navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisir une page :", ["📊 Mon Tableau de Bord", "🔮 Prédictions ETF"])

# 3. Module de prédictions
if page == "🔮 Prédictions ETF":
    import importlib
    st.title("🔮 Algorithme Prédictif ETF (Random Forest)")
    st.write("Ce module utilise une intelligence artificielle pour anticiper les tendances à 6 mois.")
    
    if st.button("🚀 Calculer et afficher les prédictions en direct"):
        with st.spinner("📥 Téléchargement des données Yahoo Finance et entraînement de l'IA..."):
            try:
                etf_predictor = importlib.import_module("etf_predictor")
                OUTPUT_DIR = "output_etf_predictor"
                
                data = etf_predictor.download_data(etf_predictor.TICKERS, years=etf_predictor.TRAIN_YEARS)
                datasets = etf_predictor.build_dataset(data, horizon=etf_predictor.HORIZON_JOURS)
                results = etf_predictor.train_and_evaluate(datasets)
                pred_df = etf_predictor.predict_current(results, data)
                bt_results = etf_predictor.backtest_strategy(results, data)
                etf_predictor.plot_all(results, bt_results, pred_df, data, save_dir=OUTPUT_DIR)
                
                st.success("✅ Analyses complétées avec succès !")
                st.dataframe(pred_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                if os.path.exists(f"{OUTPUT_DIR}/fig2_backtest.png"):
                    col1.image(f"{OUTPUT_DIR}/fig2_backtest.png", caption="Stratégie RF vs Buy & Hold")
                if os.path.exists(f"{OUTPUT_DIR}/fig3_prix.png"):
                    col2.image(f"{OUTPUT_DIR}/fig3_prix.png", caption="Historique et Bandes de Bollinger")
            except Exception as e:
