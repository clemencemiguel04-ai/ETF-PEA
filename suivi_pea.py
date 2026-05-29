
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import date
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
 
# --- Configuration de la page ---
st.set_page_config(page_title="Dashboard PEA", page_icon="📈", layout="wide")
 
FICHIER_EXCEL = "suivi_pea_detail.xlsx"
 
# ⏱️ Refresh automatique toutes les 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="autorefresh")
 
# --- Tickers Yahoo Finance ---
YAHOO_TICKERS = {
    "PAASI":  "PAASI.PA",
    "PINDIA": "PINR.PA",
    "PUST":   "PUST.PA",
    "EFENSE": "GUARD.PA",
    "IWSC":   "WPEA.PA",
}
 
@st.cache_data(ttl=300)  # cache 5 minutes
def get_prix_actuels():
    prix = {}
    for nom, ticker in YAHOO_TICKERS.items():
        try:
            data = yf.Ticker(ticker)
            prix[nom] = round(data.fast_info["last_price"], 3)
        except Exception:
            prix[nom] = None
    return prix
 
# --- Mes ETFs ---
MES_ETFS = {
    "PAASI – Amundi PEA MSCI Emerging Asia (FR0013412012)": "PAASI",
    "PINDIA – Amundi PEA Inde (FR0011869320)": "PINDIA",
    "PUST – Amundi PEA Nasdaq-100 UCITS ETF (FR0011871110)": "PUST",
    "EFENSE – BNP Paribas Easy Bloomberg Europe Defense (LU3047998896)": "EFENSE",
    "IWSC – iShares MSCI World Swap PEA UCITS ETF (IE0002XZSH01)": "IWSC",
}
 
QUANTITES_DEFAUT = {
    "PAASI": 4.0,
    "PINDIA": 3.0,
    "PUST": 1.0,
    "EFENSE": 3.0,
    "IWSC": 9.0,
}
 
# --- Fonctions de gestion des données ---
def charger_donnees():
    if os.path.exists(FICHIER_EXCEL):
        return pd.read_excel(FICHIER_EXCEL)
    return pd.DataFrame(columns=[
        "Date", "ETF", "Quantité", "PRU", "Prix Actuel",
        "Investi Ligne (€)", "Valeur Ligne (€)", "+/- Value Ligne (€)", "+/- Value Ligne (%)", "Espèces du PEA (€)"
    ])
 
def sauvegarder_donnees(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by="Date")
    df.to_excel(FICHIER_EXCEL, index=False)
 
df = charger_donnees()
 
# --- Récupération des prix en temps réel ---
prix_live = get_prix_actuels()
 
# --- Interface Utilisateur ---
st.title("📈 Tableau de Bord PEA")
 
# Indicateur de statut des prix live
with st.container():
    cols = st.columns(len(YAHOO_TICKERS))
    for i, (nom, prix) in enumerate(prix_live.items()):
        if prix:
            cols[i].metric(nom, f"{prix:.3f} €")
        else:
            cols[i].metric(nom, "N/A ⚠️")
 
# Formulaire d'ajout de données
with st.expander("➕ Saisir les données d'un ETF", expanded=df.empty):
    with st.form("ajout_donnees"):
        col1, col2, col3 = st.columns(3)
 
        with col1:
            saisie_date = st.date_input("Date du relevé", value=date.today())
            etf_selectionne = st.selectbox("ETF", options=list(MES_ETFS.keys()))
            nom_etf = MES_ETFS[etf_selectionne]
 
        with col2:
            quantite_defaut = QUANTITES_DEFAUT.get(nom_etf, 0.0)
            quantite = st.number_input("Quantité possédée", min_value=0.0, value=quantite_defaut, step=0.001, format="%0.3f")
            pru = st.number_input("PRU (Prix de Revient Unitaire en €)", min_value=0.0, step=0.001, format="%0.3f")
 
        with col3:
            prix_live_etf = prix_live.get(nom_etf) or 0.0
            prix_actuel = st.number_input("Prix actuel de l'ETF (€)", min_value=0.0, value=prix_live_etf, step=0.001, format="%0.3f")
            especes = st.number_input("Espèces sur le PEA (€)", min_value=0.0, step=0.01)
 
        soumis = st.form_submit_button("💾 Calculer et Enregistrer")
 
        if soumis:
            investi_ligne = quantite * pru
            valeur_ligne = quantite * prix_actuel
            plus_moins_value_euros = valeur_ligne - investi_ligne
            plus_moins_value_pct = (plus_moins_value_euros / investi_ligne * 100) if investi_ligne > 0 else 0
 
            nouvelle_ligne = pd.DataFrame([{
                "Date": saisie_date,
                "ETF": nom_etf,
                "Quantité": quantite,
                "PRU": pru,
                "Prix Actuel": prix_actuel,
                "Investi Ligne (€)": investi_ligne,
                "Valeur Ligne (€)": valeur_ligne,
                "+/- Value Ligne (€)": plus_moins_value_euros,
                "+/- Value Ligne (%)": plus_moins_value_pct,
                "Espèces du PEA (€)": especes
            }])
 
            df = pd.concat([df, nouvelle_ligne], ignore_index=True)
            sauvegarder_donnees(df)
            st.success(f"✅ Données pour {nom_etf} enregistrées !")
            st.rerun()
 
if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])
    date_la_plus_recente = df['Date'].max()
    df_recent = df[df['Date'] == date_la_plus_recente]
 
    # Calculs pour le Dashboard global
    evaluation_titres = df_recent['Valeur Ligne (€)'].sum()
    especes_actuelles = df_recent['Espèces du PEA (€)'].max()
    total_pea = evaluation_titres + especes_actuelles
 
    total_investi_titres = df_recent['Investi Ligne (€)'].sum()
    total_investi_pea = total_investi_titres + especes_actuelles
 
    plus_moins_value_cpt = total_pea - total_investi_pea
    pmv_cpt_pct = (plus_moins_value_cpt / total_investi_pea * 100) if total_investi_pea > 0 else 0
 
    # --- 1. DASHBOARD RÉSUMÉ ---
    st.header(f"💼 Mon Compte au {date_la_plus_recente.strftime('%d/%m/%Y')}")
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Solde Total PEA", f"{total_pea:,.2f} €".replace(',', ' '))
    m2.metric("Évaluation Titres", f"{evaluation_titres:,.2f} €".replace(',', ' '))
    m3.metric("Espèces (Cash)", f"{especes_actuelles:,.2f} €".replace(',', ' '))
    m4.metric("+/- Value CPT (Global)", f"{plus_moins_value_cpt:,.2f} €", f"{pmv_cpt_pct:.2f} %")
 
    st.divider()
 
    # --- 2. GRAPHIQUE ÉVOLUTION ---
    st.subheader("📊 Évolution Historique")
 
    df_grouped = df.groupby('Date').agg({
        'Investi Ligne (€)': 'sum', 'Valeur Ligne (€)': 'sum', 'Espèces du PEA (€)': 'max'
    }).reset_index()
 
    df_grouped['Total Investi (€)'] = df_grouped['Investi Ligne (€)'] + df_grouped['Espèces du PEA (€)']
    df_grouped['Total PEA (€)'] = df_grouped['Valeur Ligne (€)'] + df_grouped['Espèces du PEA (€)']
    df_grouped['+/- Value CPT (€)'] = df_grouped['Total PEA (€)'] - df_grouped['Total Investi (€)']
    df_grouped['+/- Value CPT (%)'] = df_grouped.apply(
        lambda r: (r['+/- Value CPT (€)'] / r['Total Investi (€)'] * 100) if r['Total Investi (€)'] > 0 else 0, axis=1
    )
 
    type_affichage = st.radio("Afficher :", ["+/- Value CPT (%)", "+/- Value CPT (€)", "Évolution du Capital"], horizontal=True)
 
    fig = go.Figure()
 
    if type_affichage == "+/- Value CPT (%)":
        fig.add_trace(go.Scatter(x=df_grouped["Date"], y=df_grouped["+/- Value CPT (%)"], mode='lines+markers', line=dict(color='#00FFAA', width=3)))
        fig.update_layout(yaxis_ticksuffix=" %")
    elif type_affichage == "+/- Value CPT (€)":
        fig.add_trace(go.Scatter(x=df_grouped["Date"], y=df_grouped["+/- Value CPT (€)"], mode='lines+markers', line=dict(color='#FF00AA', width=3)))
        fig.update_layout(yaxis_ticksuffix=" €")
    else:
        fig.add_trace(go.Scatter(x=df_grouped["Date"], y=df_grouped["Total Investi (€)"], mode='lines', name='Total Investi', line=dict(color='#888888', dash='dash')))
        fig.add_trace(go.Scatter(x=df_grouped["Date"], y=df_grouped["Total PEA (€)"], mode='lines+markers', name='Valeur PEA', line=dict(color='#0088FF', width=3)))
        fig.update_layout(yaxis_ticksuffix=" €")
 
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # --- 3. DÉTAIL PAR LIGNE ---
    st.divider()
    st.subheader("🔎 Détail par ligne (Dernier pointage)")
 
    st.dataframe(df_recent[[
        "ETF", "Quantité", "PRU", "Prix Actuel", "Investi Ligne (€)", "Valeur Ligne (€)", "+/- Value Ligne (€)", "+/- Value Ligne (%)"
    ]].style.format({
        "Quantité": "{:.3f}", "PRU": "{:.3f} €", "Prix Actuel": "{:.3f} €",
        "Investi Ligne (€)": "{:.2f} €", "Valeur Ligne (€)": "{:.2f} €",
        "+/- Value Ligne (€)": "{:+.2f} €", "+/- Value Ligne (%)": "{:+.2f} %"
    }).map(
        lambda x: 'color: #00FFAA' if isinstance(x, (float, int)) and x > 0 else ('color: #FF4444' if isinstance(x, (float, int)) and x < 0 else ''),
        subset=["+/- Value Ligne (€)", "+/- Value Ligne (%)"]
    ),
    use_container_width=True, hide_index=True)
 
    with st.expander("📂 Voir tout l'historique brut Excel"):
        st.dataframe(df)
 
else:
    st.info("👋 Sélectionne un ETF et entre tes données pour commencer !")
