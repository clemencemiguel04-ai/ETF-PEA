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
    "IWSC":   "WPEA.PA",
}

# --- Quantités et PRU ---
PORTEFEUILLE = {
    "PAASI":  {"quantite": 4.0,  "pru": 34.755},
    "PINDIA": {"quantite": 3.0,  "pru": 21.890},
    "PUST":   {"quantite": 1.0,  "pru": 100.090},
    "EFENSE": {"quantite": 3.0,  "pru": 10.387},
    "IWSC":   {"quantite": 9.0,  "pru": 6.239},
}

ESPECES = 7.91  # ← Mets à jour si ça change

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
            "Date": aujourd_hui,
            "ETF": nom,
            "Quantité": q,
            "PRU": pru,
            "Prix Actuel": prix_actuel,
            "Investi Ligne (€)": investi,
            "Valeur Ligne (€)": valeur,
            "+/- Value Ligne (€)": plus_value,
            "+/- Value Ligne (%)": plus_value_pct,
            "Espèces du PEA (€)": ESPECES
        })

    if nouvelles_lignes:
        df = pd.concat([df, pd.DataFrame(nouvelles_lignes)], ignore_index=True)
        sauvegarder_donnees(df)
        print(f"✅ {len(nouvelles_lignes)} lignes sauvegardées dans {FICHIER_EXCEL}")
    else:
        print("❌ Aucune donnée récupérée.")

if __name__ == "__main__":
    run()
