import yfinance as yf
import pandas as pd
import os
from datetime import date

FICHIER_EXCEL = "suivi_pea_detail.xlsx"

# --- Tickers Yahoo Finance ---
YAHOO_TICKERS = {
    "PAASI":  "PAASI.PA",
    "PINDIA": "PINR.PA",
    "PUST":   "PUST.PA",
    "EFENSE": "GUARD.PA",
    "SOITEC": "SOI.PA",
    "IWSC":   "WPEA.PA",
}

# --- Quantités et PRU (relevés chez le courtier le 04/10/2026) ---
PORTEFEUILLE = {
    "PAASI":  {"quantite": 22.0, "pru": 36.956},
    "PINDIA": {"quantite": 3.0,  "pru": 21.890},
    "PUST":   {"quantite": 1.0,  "pru": 100.090},
    "EFENSE": {"quantite": 3.0,  "pru": 10.387},
    "SOITEC": {"quantite": 1.0,  "pru": 166.780},
    "IWSC":   {"quantite": 9.0,  "pru": 6.239},
}

ESPECES = 7.71  # ← Mets à jour si ça change

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

def run():
    aujourd_hui = date.today()
    print(f"📅 Pointage automatique du {aujourd_hui}")

    df = charger_donnees()

    # Vérifie qu'on n'a pas déjà pointé aujourd'hui
    df['Date'] = pd.to_datetime(df['Date'])
    if not df.empty and (df['Date'].dt.date == aujourd_hui).any():
        print("✅ Pointage déjà effectué aujourd'hui, on skip.")
        return

    nouvelles_lignes = []
    for nom, ticker in YAHOO_TICKERS.items():
        try:
            data = yf.Ticker(ticker)
            prix_actuel = round(data.fast_info["last_price"], 3)
            print(f"  {nom} ({ticker}) : {prix_actuel} €")
        except Exception as e:
            print(f"  ⚠️ Erreur pour {nom} : {e}")
            continue

        q = PORTEFEUILLE[nom]["quantite"]
        pru = PORTEFEUILLE[nom]["pru"]
        investi = round(q * pru, 2)
        valeur = round(q * prix_actuel, 2)
        plus_value = round(valeur - investi, 2)
        plus_value_pct = round((plus_value / investi * 100) if investi > 0 else 0, 2)

        nouvelles_lignes.append({
