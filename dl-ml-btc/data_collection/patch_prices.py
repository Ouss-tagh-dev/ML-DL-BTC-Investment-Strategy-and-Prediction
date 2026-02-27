"""
╔══════════════════════════════════════════════════════════════╗
║  BTC PRICE PATCHER — Ajoute les prix au CSV existant        ║
║  Source: yfinance (BTC-USD, gratuit, sans clé, depuis 2010) ║
║  Usage: python patch_prices.py                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import pandas as pd
import numpy as np

# ── Chemin du CSV à patcher ───────────────────────────────────
CSV_PATH = r"C:\Users\15086\Desktop\data\raw\btc_news.csv"

# ─────────────────────────────────────────────────────────────
#  ÉTAPE 1 — Installer yfinance si nécessaire
# ─────────────────────────────────────────────────────────────
import sys
import subprocess

try:
    import yfinance as yf
    print("✅ yfinance déjà installé")
except ImportError:
    print("📦 Installation de yfinance...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
    import yfinance as yf
    print("✅ yfinance installé")


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 2 — Charger le CSV existant
# ─────────────────────────────────────────────────────────────
print(f"\n📂 Chargement de {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"   Articles: {len(df):,}")
print(f"   Période : {df['date'].min()} → {df['date'].max()}")
print(f"   Labels avant patch — UP: {(df['label']==1).sum()} | DOWN: {(df['label']==0).sum()}")


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 3 — Télécharger prix BTC via yfinance
# ─────────────────────────────────────────────────────────────
start_date = df['date'].min()
end_date   = df['date'].max()

print(f"\n📥 Téléchargement BTC-USD ({start_date} → {end_date})...")

btc = yf.download(
    "BTC-USD",
    start=start_date,
    end=end_date,
    interval="1d",
    progress=True,
    auto_adjust=True,
)

if btc.empty:
    print("❌ yfinance n'a rien retourné. Essaie avec une connexion sans VPN.")
    sys.exit(1)

btc = btc.reset_index()
btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]

price_df = pd.DataFrame({
    "date":  pd.to_datetime(btc["Date"]).dt.strftime("%Y-%m-%d"),
    "price": btc["Close"].values.flatten().astype(float),
})
price_df = price_df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)

# Calcul des colonnes J et J+1
price_df["price"]                 = price_df["price"].round(2)
price_df["price_change_24h"]      = (price_df["price"].pct_change() * 100).round(4)
price_df["price_change_next_day"] = price_df["price_change_24h"].shift(-1).round(4)
price_df["price_next_day"]        = price_df["price"].shift(-1).round(2)

print(f"✅ {len(price_df)} jours de prix récupérés")
print(f"   BTC min: ${price_df['price'].min():,.0f} | max: ${price_df['price'].max():,.0f}")
print(f"   Jours UP  : {(price_df['price_change_next_day'] > 0).sum()}")
print(f"   Jours DOWN: {(price_df['price_change_next_day'] < 0).sum()}")


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 4 — Merger les prix dans le CSV
# ─────────────────────────────────────────────────────────────
print("\n🔗 Merge des prix dans le CSV...")

# Supprimer les anciennes colonnes prix si elles existent
cols_to_drop = ["price", "price_next_day", "price_change_24h", "price_change_next_day", "label", "severity"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Merge
df = df.merge(
    price_df[["date", "price", "price_next_day", "price_change_24h", "price_change_next_day"]],
    on="date",
    how="left",
)

# Recalcul des labels J+1
df["label"] = df["price_change_next_day"].apply(
    lambda x: 1 if pd.notna(x) and x > 0 else 0
)
df["severity"] = df["price_change_next_day"].abs().apply(
    lambda x: min(10, max(1, int(abs(x)))) if pd.notna(x) else 1
)

n_with_price = df["price"].notna().sum()
n_up         = (df["label"] == 1).sum()
n_down       = (df["label"] == 0).sum()

print(f"   Articles avec prix  : {n_with_price:,} / {len(df):,}")
print(f"   Articles sans prix  : {df['price'].isna().sum():,}")
print(f"   Labels UP   (1)     : {n_up:,}  ({n_up/len(df)*100:.1f}%)")
print(f"   Labels DOWN (0)     : {n_down:,}  ({n_down/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 5 — Sauvegarde
# ─────────────────────────────────────────────────────────────

# Ordre colonnes final
cols = [
    "event_id", "timestamp", "date",
    "title", "text_clean",
    "source", "url", "category",
    "sentiment_score",
    "price", "price_next_day",
    "price_change_24h", "price_change_next_day",
    "label", "severity",
]
df = df[[c for c in cols if c in df.columns]]

df.to_csv(CSV_PATH, index=False, encoding="utf-8")
print(f"\n✅ CSV patché et sauvegardé: {CSV_PATH}")

# Backup séparé
backup_path = CSV_PATH.replace(".csv", "_with_prices.csv")
df.to_csv(backup_path, index=False, encoding="utf-8")
print(f"✅ Backup: {backup_path}")

# ─────────────────────────────────────────────────────────────
#  RAPPORT FINAL
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RAPPORT FINAL — CSV PATCHÉ")
print(f"{'='*55}")
print(f"  Total articles   : {len(df):,}")
print(f"  Période          : {df['date'].min()} → {df['date'].max()}")
print(f"  Jours couverts   : {df['date'].nunique()}")
print(f"  Labels UP (1)    : {n_up:,}  ({n_up/len(df)*100:.1f}%)")
print(f"  Labels DOWN (0)  : {n_down:,}  ({n_down/len(df)*100:.1f}%)")

if n_up == 0:
    print("\n⚠️  ENCORE 0 labels UP!")
    print("   Cause possible: VPN actif bloquant yfinance")
    print("   → Essaie de désactiver le VPN puis relance ce script")
else:
    print(f"\n🎉 SUCCESS! Labels OK — CSV prêt pour le notebook FinBERT!")
    print(f"   Lance maintenant BTC_News_FinBERT.ipynb")
print(f"{'='*55}")
