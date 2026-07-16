import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import date
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# 1. Configuration de la page
st.set_page_config(page_title="Dashboard PEA", page_icon="📈", layout="wide")

# 2. Menu de navigation ultra-simple et sécurisé (sans page_link)
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisir une page :", ["📊 Mon Tableau de Bord", "🔮 Prédictions ETF"])

# 3. Si l'utilisateur choisit les prédictions, on affiche le module IA et on arrête le reste du script
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
                st.error(f"❌ Erreur : {e}")
    st.stop() # Arrête le script ici pour ne pas afficher le tableau de bord en dessous

# 4. LE RESTE DE TON CODE SOURCE INITIAL REPREND ICI AUTOMATIQUEMENT SI ON EST SUR LE TABLEAU DE BORD
FICHIER_EXCEL = "suivi_pea_detail.xlsx"

# ⏱️ Refresh toutes les 5 min uniquement pendant les heures de marché (9h-18h)
from datetime import datetime as _dt
_heure = _dt.now().hour
if 9 <= _heure < 18:
    st_autorefresh(interval=5 * 60 * 1000, key="autorefresh")

# --- Tickers Yahoo Finance ---
YAHOO_TICKERS = {
    "PAASI":  "PAASI.PA",
    "PINDIA": "PINR.PA",
    "PUST":   "PUST.PA",
    "EFENSE": "GUARD.PA",
    "IWSC":   "WPEA.PA",
}

@st.cache_data(ttl=300)
def get_prix_actuels():
    prix = {}
    erreurs = {}
    for nom, ticker in YAHOO_TICKERS.items():
        valeur = None
        data = None
        for tentative in range(2):  # 1 essai + 1 retry
            try:
                data = yf.Ticker(ticker)
                valeur = round(data.fast_info["last_price"], 3)
                break
            except Exception as e1:
                try:
                    if data is None:
                        data = yf.Ticker(ticker)
                    info = data.info
                    prix_fallback = info.get("regularMarketPrice") or info.get("previousClose")
                    if prix_fallback:
                        valeur = round(prix_fallback, 3)
                        break
                except Exception as e2:
                    erreurs[nom] = f"fast_info: {e1} | info: {e2}"
        prix[nom] = valeur
    st.session_state["erreurs_prix"] = erreurs
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

# Affichage des erreurs de récupération de prix, si présentes
if st.session_state.get("erreurs_prix"):
    with st.expander("⚠️ Détails des erreurs de récupération des prix"):
        st.json(st.session_state["erreurs_prix"])

# Indicateur de statut des prix live
with st.container():
    cols = st.columns(len(YAHOO_TICKERS))
    for i, (nom, prix) in enumerate(prix_live.items()):
        if prix:
            cols[i].metric(nom, f"{prix:.3f} €")
        else:
            cols[i].metric(nom, "N/A ⚠️")

# --- Formulaire d'ajout de données ETF (SANS champ espèces) ---
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

        soumis = st.form_submit_button("💾 Calculer et Enregistrer")

        if soumis:
            investi_ligne = quantite * pru
            valeur_ligne = quantite * prix_actuel
            plus_moins_value_euros = valeur_ligne - investi_ligne
            plus_moins_value_pct = (plus_moins_value_euros / investi_ligne * 100) if investi_ligne > 0 else 0

            # Récupérer les espèces de la date la plus récente existante
            especes_existantes = 0.0
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                date_recente_avant_ajout = df['Date'].max()
                especes_existantes = df[df['Date'] == date_recente_avant_ajout]['Espèces du PEA (€)'].max()
                if pd.isna(especes_existantes):
                    especes_existantes = 0.0

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
                "Espèces du PEA (€)": especes_existantes  # reprend les espèces existantes
            }])

            lignes_a_ajouter = [nouvelle_ligne]

            # 🔁 REPORT AUTOMATIQUE DES AUTRES ETF À CETTE MÊME DATE
            # Sans ça, les ETF non-resaisis "disparaissent" du dernier pointage,
            # car le dashboard n'affiche que les lignes de la date la plus récente.
            if not df.empty:
                saisie_date_ts = pd.to_datetime(saisie_date)
                autres_etfs = [e for e in MES_ETFS.values() if e != nom_etf]
                for autre_etf in autres_etfs:
                    deja_present = ((df['ETF'] == autre_etf) & (df['Date'] == saisie_date_ts)).any()
                    if not deja_present:
                        df_etf_hist = df[df['ETF'] == autre_etf].sort_values('Date')
                        if not df_etf_hist.empty:
                            derniere = df_etf_hist.iloc[-1]
                            q = derniere['Quantité']
                            p = derniere['PRU']
                            prix_a = prix_live.get(autre_etf) or derniere['Prix Actuel']
                            inv = q * p
                            val = q * prix_a
                            pmv_e = val - inv
                            pmv_p = (pmv_e / inv * 100) if inv > 0 else 0

                            ligne_reportee = pd.DataFrame([{
                                "Date": saisie_date,
                                "ETF": autre_etf,
                                "Quantité": q,
                                "PRU": p,
                                "Prix Actuel": prix_a,
                                "Investi Ligne (€)": inv,
                                "Valeur Ligne (€)": val,
                                "+/- Value Ligne (€)": pmv_e,
                                "+/- Value Ligne (%)": pmv_p,
                                "Espèces du PEA (€)": especes_existantes
                            }])
                            lignes_a_ajouter.append(ligne_reportee)

            df = pd.concat([df] + lignes_a_ajouter, ignore_index=True)
            sauvegarder_donnees(df)
            st.success(f"✅ Données pour {nom_etf} enregistrées ! (les autres ETF ont été reportés à cette date)")
            st.rerun()

# --- Formulaire SÉPARÉ pour mettre à jour les espèces ---
if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])
    date_la_plus_recente = df['Date'].max()
    df_recent = df[df['Date'] == date_la_plus_recente]
    especes_actuelles_val = float(df_recent['Espèces du PEA (€)'].max() or 0.0)

    with st.expander("💰 Mettre à jour les espèces du PEA"):
        with st.form("maj_especes"):
            nouvelles_especes = st.number_input(
                "Espèces actuelles sur le PEA (€)",
                min_value=0.0,
                value=especes_actuelles_val,
                step=0.01,
                format="%0.2f"
            )
            if st.form_submit_button("💾 Mettre à jour les espèces"):
                # Met à jour TOUTES les lignes de la date la plus récente, sans créer de nouvelle ligne
                df.loc[df['Date'] == date_la_plus_recente, 'Espèces du PEA (€)'] = nouvelles_especes
                sauvegarder_donnees(df)
                st.success("✅ Espèces mises à jour !")
                st.rerun()

if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])
    date_la_plus_recente = df['Date'].max()
    df_recent = df[df['Date'] == date_la_plus_recente].copy()

    # ✅ Recalcul en temps réel avec les prix live
    for etf, prix in prix_live.items():
        if prix is not None:
            mask = df_recent['ETF'] == etf
            df_recent.loc[mask, 'Prix Actuel'] = prix
            df_recent.loc[mask, 'Valeur Ligne (€)'] = df_recent.loc[mask, 'Quantité'] * prix
            df_recent.loc[mask, '+/- Value Ligne (€)'] = df_recent.loc[mask, 'Valeur Ligne (€)'] - df_recent.loc[mask, 'Investi Ligne (€)']
            df_recent.loc[mask, '+/- Value Ligne (%)'] = (
                df_recent.loc[mask, '+/- Value Ligne (€)'] / df_recent.loc[mask, 'Investi Ligne (€)'] * 100
            ).where(df_recent.loc[mask, 'Investi Ligne (€)'] > 0, 0)

    # Calculs pour le Dashboard global
    evaluation_titres = df_recent['Valeur Ligne (€)'].sum()
    especes_actuelles = df_recent['Espèces du PEA (€)'].max()
    if pd.isna(especes_actuelles):
        especes_actuelles = 0.0
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

    # Fusion des données historiques avec les valeurs live du jour
    df_pour_graphe = df[df['Date'] != date_la_plus_recente].copy()
    df_pour_graphe = pd.concat([df_pour_graphe, df_recent], ignore_index=True)

    df_grouped = df_pour_graphe.groupby('Date').agg({
        'Investi Ligne (€)': 'sum',
        'Valeur Ligne (€)': 'sum',
        'Espèces du PEA (€)': 'max'
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

    # --- 3. GRAPHIQUE ÉVOLUTION PAR ETF ---
    st.divider()
    st.subheader("📈 Évolution par ETF")

    COULEURS_ETF = {
        "PAASI":  "#3B82F6",
        "PINDIA": "#F59E0B",
        "PUST":   "#10B981",
        "EFENSE": "#8B5CF6",
        "IWSC":   "#EF4444",
    }

    type_affichage_etf = st.radio("Afficher :", ["+/- Value (%)", "+/- Value (€)"], horizontal=True, key="radio_etf")

    fig_etf = go.Figure()

    for etf in df_pour_graphe['ETF'].unique():
        df_etf = df_pour_graphe[df_pour_graphe['ETF'] == etf].sort_values('Date')
        if type_affichage_etf == "+/- Value (%)":
            fig_etf.add_trace(go.Scatter(
                x=df_etf["Date"], y=df_etf["+/- Value Ligne (%)"],
                mode='lines+markers', name=etf,
                line=dict(color=COULEURS_ETF.get(etf, "#FFFFFF"), width=2)
            ))
            fig_etf.update_layout(yaxis_ticksuffix=" %")
        else:
            fig_etf.add_trace(go.Scatter(
                x=df_etf["Date"], y=df_etf["+/- Value Ligne (€)"],
                mode='lines+markers', name=etf,
                line=dict(color=COULEURS_ETF.get(etf, "#FFFFFF"), width=2)
            ))
            fig_etf.update_layout(yaxis_ticksuffix=" €")

    fig_etf.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_etf, use_container_width=True)

    # --- 4. GRAPHIQUE HISTORIQUE 5 ANS ---
    st.divider()
    st.subheader("🕰️ Historique 5 ans (base 100)")
    st.caption("Toutes les courbes sont normalisées à 100 au départ pour comparer les performances.")

    TICKERS_5ANS = {
        "PAASI":  "PAASI.PA",
        "PINDIA": "PINR.PA",
        "PUST":   "PUST.PA",
        "EFENSE": "GUARD.PA",
        "IWSC":   "WPEA.PA",
        "MSCI World (CW8)":     "CW8.PA",
        "S&P 500 (SP5)":        "SP5.PA",
        "Nasdaq-100 (UST)":     "UST.PA",
        "Émergents (PAEEM)":    "PAEEM.PA",
        "Nasdaq Amundi (PANX)": "PANX.PA",
        "Luxe (LOOKS)":         "LOOKS.PA",
        "MSCI World (indice)":  "URTH",
        "S&P 500 (indice)":     "^GSPC",
        "CAC 40 (indice)":      "^FCHI",
    }

    COULEURS_5ANS = {
        "PAASI":  "#3B82F6",
        "PINDIA": "#F59E0B",
        "PUST":   "#10B981",
        "EFENSE": "#8B5CF6",
        "IWSC":   "#EF4444",
        "MSCI World (CW8)":     "#93C5FD",
        "S&P 500 (SP5)":        "#FCD34D",
        "Nasdaq-100 (UST)":     "#6EE7B7",
        "Émergents (PAEEM)":    "#C4B5FD",
        "Nasdaq Amundi (PANX)": "#FCA5A5",
        "Luxe (LOOKS)":         "#F9A8D4",
        "MSCI World (indice)":  "#FFFFFF",
        "S&P 500 (indice)":     "#D1D5DB",
        "CAC 40 (indice)":      "#9CA3AF",
    }

    @st.cache_data(ttl=3600)
    def get_historique_5ans():
        from datetime import datetime, timedelta
        date_debut = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        historique = {}
        for nom, ticker in TICKERS_5ANS.items():
            try:
                data = yf.download(ticker, start=date_debut, progress=False, auto_adjust=True)
                if not data.empty:
                    serie = data['Close'].squeeze()
                    historique[nom] = (serie / serie.iloc[0]) * 100
            except Exception:
                pass
        return historique

    with st.spinner("Chargement de l'historique 5 ans..."):
        historique = get_historique_5ans()

    tous = list(historique.keys())
    mes_etf_defaut = [e for e in ["PAASI", "PINDIA", "PUST", "EFENSE", "IWSC"] if e in tous]
    selection = st.multiselect(
        "Sélectionner les courbes à afficher :",
        options=tous,
        default=mes_etf_defaut
    )

    fig_5ans = go.Figure()
    for nom in selection:
        if nom in historique:
            is_indice = "(indice)" in nom
            fig_5ans.add_trace(go.Scatter(
                x=historique[nom].index,
                y=historique[nom].values,
                mode='lines',
                name=nom,
                line=dict(
                    color=COULEURS_5ANS.get(nom, "#FFFFFF"),
                    width=1.5 if is_indice else 2,
                    dash='dot' if is_indice else 'solid'
                )
            ))

    fig_5ans.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_5ans, use_container_width=True)

    # --- 5. DÉTAIL PAR LIGNE ---
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

    with st.expander("🗑️ Supprimer une ligne"):
        df_affichage = df.copy()
        df_affichage.index = range(len(df_affichage))
        st.dataframe(df_affichage[["Date", "ETF", "Quantité", "PRU", "Espèces du PEA (€)"]], use_container_width=True)

        index_a_supprimer = st.number_input(
            "Numéro de la ligne à supprimer (voir index ci-dessus)",
            min_value=0, max_value=len(df_affichage)-1, step=1
        )

        if st.button("🗑️ Supprimer cette ligne"):
            df = df.drop(df.index[index_a_supprimer]).reset_index(drop=True)
            sauvegarder_donnees(df)
            st.success("✅ Ligne supprimée !")
            st.rerun()

else:
    st.info("👋 Sélectionne un ETF et entre tes données pour commencer !")
