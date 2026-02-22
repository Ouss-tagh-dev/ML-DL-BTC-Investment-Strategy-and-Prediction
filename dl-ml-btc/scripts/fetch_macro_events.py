"""
Macro Events → BTC Impact  (format compatible avec historical_train.csv)
=========================================================================
Collecte des événements réels (guerres, crises éco, banques centrales, grèves…)
depuis l'API GDELT (gratuite, sans clé) + flux RSS globaux, puis corrèle chaque
article avec la variation du prix BTC sur 24h via l'API publique Kraken.

Schéma de sortie — identique à historical_train.csv :
    datetime  : date de l'événement (YYYY-MM-DD)
    text      : résumé/titre de l'article (texte brut, sans HTML)
    url       : URL source
    label     : 1 = BTC UP (hausse ≥ seuil), 0 = BTC DOWN (baisse ou neutre forcé)

But : enrichir le dataset d'entraînement BERT avec des événements macro réels,
pour que le modèle apprenne la causalité situation-du-monde → impact BTC.

Dépendances : pip install feedparser pandas requests numpy
"""

import feedparser
import pandas as pd
import requests
import numpy as np
import re
import os
import time
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "macro_events_btc_train.csv")

# Seuil en % pour décider UP (1) ou DOWN (0)
# Choix conservateur : ≥ +1% = UP, sinon DOWN (pas de NEUTRAL)
UP_THRESHOLD_PCT = 1.0

# ─────────────────────────────────────────────────────────────────
# SOURCES D'ÉVÉNEMENTS MACRO RÉELS
# ─────────────────────────────────────────────────────────────────

# A) Flux RSS de médias généralistes mondiaux
GLOBAL_RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.dw.com/rdf/rss-en-world",
    "https://rss.dw.com/rdf/rss-en-pol",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.federalreserve.gov/feeds/press_all.xml",   # FED officiel
    "https://www.bis.org/doclist/speeches.rss",             # BIS
]

# B) Mots-clés macro pour filtrer les articles pertinents
MACRO_KEYWORDS = [
    # Géopolitique / Conflits
    "war", "invasion", "conflict", "military", "attack", "bombing",
    "ceasefire", "sanctions", "troops", "nato", "ukraine", "gaza",
    "israel", "iran", "russia", "coup", "civil war",
    # Crises économiques
    "recession", "default", "bank run", "collapse", "bankruptcy",
    "financial crisis", "hyperinflation", "devaluation", "bail out",
    "debt crisis", "svb", "silicon valley bank",
    # Banques centrales
    "federal reserve", "fed", "ecb", "interest rate", "rate hike",
    "rate cut", "quantitative easing", "monetary policy", "cpi",
    "inflation data", "powell", "lagarde", "fomc", "pivot",
    # Régulation
    "regulation", "sec", "ban", "lawsuit", "legislation",
    "congress", "g7", "g20", "fatf", "bitcoin ban", "crypto law",
    # Grèves / Troubles sociaux
    "strike", "protest", "riots", "unrest", "work stoppage",
    "general strike", "union", "demonstration",
    # Pandémies / Santé
    "pandemic", "covid", "virus", "variant", "lockdown",
    "quarantine", "health crisis", "who", "epidemic",
    # Catastrophes naturelles
    "earthquake", "hurricane", "flood", "wildfire", "tsunami",
    # Macro data
    "gdp", "unemployment", "trade war", "tariff", "supply chain",
    "oil price", "opec", "energy crisis",
]


def is_macro_relevant(text: str) -> bool:
    """Retourne True si le texte contient au moins un mot-clé macro."""
    text_l = text.lower()
    return any(kw in text_l for kw in MACRO_KEYWORDS)


def clean_html(raw: str) -> str:
    return re.sub(re.compile(r'<.*?>'), '', raw or "").strip()


def clean_text(text: str) -> str:
    """Nettoyage léger du texte (comme le dataset historique)."""
    text = clean_html(text)
    # Supprimer caractères spéciaux superflus
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text[:600]   # limite raisonnable


# ─────────────────────────────────────────────────────────────────
# SOURCE A — RSS GLOBAUX
# ─────────────────────────────────────────────────────────────────

def fetch_from_rss() -> list:
    print(f"\n📡 [SOURCE A] Scanning {len(GLOBAL_RSS_FEEDS)} RSS feeds...")
    items = []

    for url in GLOBAL_RSS_FEEDS:
        try:
            print(f"  → {url}")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title   = entry.get('title', '')
                summary = clean_html(entry.get('summary', entry.get('description', '')))
                text    = f"{title} {summary}"

                if not is_macro_relevant(text):
                    continue

                # Date
                pub_date = None
                for field in ('published_parsed', 'updated_parsed'):
                    if field in entry:
                        try:
                            pub_date = pd.Timestamp(datetime(*entry[field][:6])).tz_localize('UTC')
                            break
                        except Exception:
                            pass
                if pub_date is None:
                    pub_date = pd.Timestamp.now(tz='UTC')

                items.append({
                    'timestamp': pub_date,
                    'text':      clean_text(text),
                    'url':       entry.get('link', url),
                })
        except Exception as e:
            print(f"  ❌ {url}: {e}")

    print(f"  ✔ {len(items)} macro items from RSS.")
    return items


# ─────────────────────────────────────────────────────────────────
# SOURCE B — GDELT API  (gratuite, sans clé, données depuis 2015)
# ─────────────────────────────────────────────────────────────────
#
# GDELT DOC 2.0 API — Articles anglophones liés à des événements réels
# Docs : https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
#
# On interroge un thème à la fois pour avoir de la diversité.

GDELT_THEMES = [
    "bitcoin",
    "war conflict military",
    "recession economic crisis",
    "federal reserve interest rate",
    "pandemic covid lockdown",
    "strike protest unrest",
    "oil price opec energy",
    "sanctions embargo",
    "inflation CPI monetary",
    "earthquake hurricane disaster",
    "tariff trade war",
    "bank collapse bankruptcy",
]

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_from_gdelt() -> list:
    print(f"\n🌍 [SOURCE B] Querying GDELT for {len(GDELT_THEMES)} themes...")
    items = []

    for theme in GDELT_THEMES:
        try:
            params = {
                'query':    theme,
                'mode':     'artlist',
                'maxrecords': 75,
                'format':   'json',
                'timespan': '18months',    # 18 derniers mois
                'sort':     'DateDesc',
                'sourcelang': 'english',
            }
            resp = requests.get(GDELT_BASE, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"  ⚠ GDELT status {resp.status_code} for '{theme}'")
                continue

            data = resp.json()
            articles = data.get('articles', [])
            print(f"  → '{theme}': {len(articles)} articles")

            for art in articles:
                title = art.get('title', '')
                url   = art.get('url', '')
                seendate = art.get('seendate', '')  # format: 20230115T120000Z

                # Parse date
                try:
                    dt = datetime.strptime(seendate[:15], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
                    ts = pd.Timestamp(dt)
                except Exception:
                    ts = pd.Timestamp.now(tz='UTC')

                text = clean_text(title)
                if not text or not is_macro_relevant(text):
                    continue

                items.append({
                    'timestamp': ts,
                    'text':      text,
                    'url':       url,
                })

            time.sleep(0.5)   # respect rate limit GDELT

        except Exception as e:
            print(f"  ❌ GDELT '{theme}': {e}")

    print(f"  ✔ {len(items)} articles from GDELT.")
    return items


# ─────────────────────────────────────────────────────────────────
# PRIX BTC — Kraken Public API (sans clé)
# ─────────────────────────────────────────────────────────────────

def fetch_btc_prices_kraken(start_dt: pd.Timestamp) -> pd.DataFrame:
    print(f"\n📈 Fetching BTC/USD hourly prices from Kraken (since {start_dt.date()})...")
    url     = "https://api.kraken.com/0/public/OHLC"
    klines  = []
    since   = int(start_dt.timestamp())
    end_ts  = int(datetime.now().timestamp())

    for _ in range(120):
        if since >= end_ts:
            break
        try:
            resp = requests.get(
                url,
                params={'pair': 'XBTUSD', 'interval': 60, 'since': since},
                timeout=12
            )
            data = resp.json()
            if data.get('error'):
                errs = data['error']
                print(f"  ⚠ Kraken: {errs}")
                if any("Rate limit" in e for e in errs):
                    time.sleep(10)
                    continue
                break
            result = data.get('result', {})
            keys   = [k for k in result if k != 'last']
            if not keys:
                break
            chunk = result[keys[0]]
            if not chunk:
                break
            klines.extend(chunk)
            last  = result.get('last')
            since = int(last) if last and int(last) > since else int(chunk[-1][0]) + 1
            print(f"  - chunk ending {datetime.fromtimestamp(int(chunk[-1][0]))}")
            time.sleep(1.5)
        except Exception as e:
            print(f"  ❌ {e}")
            break

    if not klines:
        return pd.DataFrame()

    rows = [{'timestamp': int(k[0]), 'open': float(k[1]), 'close': float(k[4])} for k in klines]
    df   = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"  ✔ {len(df)} hourly candles loaded.")
    return df


# ─────────────────────────────────────────────────────────────────
# LABELING BINAIRE (1 = UP, 0 = DOWN — jamais de NEUTRAL)
# ─────────────────────────────────────────────────────────────────

def label_binary(pct_change: float) -> int:
    """
    Retourne 1 (UP) si la variation est >= UP_THRESHOLD_PCT, sinon 0 (DOWN).
    Aucun label NEUTRAL : les mouvements faibles sont assignés DOWN par défaut.
    Compatible avec le schéma de historical_train.csv.
    """
    return 1 if pct_change >= UP_THRESHOLD_PCT else 0


def correlate_events_with_btc(events: list, prices_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n🔗 Correlating {len(events)} events with BTC prices...")

    price_times = prices_df['timestamp'].values   # numpy datetime64, UTC
    labeled = []

    for ev in events:
        ts = ev['timestamp']

        # Normaliser en UTC naive pour searchsorted
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        ts_naive = ts.replace(tzinfo=None)

        idx = np.searchsorted(price_times, np.datetime64(ts_naive, 'ns'))

        # Il faut au moins 24h de prix après l'événement
        if idx >= len(prices_df) - 25:
            continue

        price_open  = prices_df.iloc[idx]['open']
        idx_24h     = min(idx + 24, len(prices_df) - 1)
        price_close = prices_df.iloc[idx_24h]['close']

        if price_open <= 0:
            continue

        pct_24h = ((price_close - price_open) / price_open) * 100
        lbl     = label_binary(pct_24h)

        labeled.append({
            'datetime': ts.strftime('%Y-%m-%d'),   # format identique au CSV historique
            'text':     ev['text'],
            'url':      ev['url'],
            'label':    lbl,
            # colonnes bonus (peuvent être supprimées si non requises)
            '_btc_price':      round(price_open, 2),
            '_btc_change_24h': round(pct_24h, 2),
        })

    print(f"  ✔ {len(labeled)} events labeled.")
    df = pd.DataFrame(labeled)
    return df


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Collecte des événements macro ──────────────────────────
    rss_items   = fetch_from_rss()
    gdelt_items = fetch_from_gdelt()
    all_items   = rss_items + gdelt_items

    if not all_items:
        print("❌ No events collected.")
        return

    # ── 2. Déduplication & filtre temporel ───────────────────────
    df_events = pd.DataFrame(all_items)
    df_events.drop_duplicates(subset=['text'], inplace=True)

    # Garder les 18 derniers mois (aligné sur GDELT timespan)
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=548)
    df_events = df_events[df_events['timestamp'] >= cutoff]
    print(f"\n📅 Events after dedup + date filter: {len(df_events)}")

    if df_events.empty:
        print("❌ No events in range.")
        return

    events_list = df_events.to_dict('records')

    # ── 3. Prix BTC depuis Kraken ──────────────────────────────────
    oldest = df_events['timestamp'].min()
    df_prices = fetch_btc_prices_kraken(oldest)

    if df_prices.empty:
        print("❌ Could not fetch BTC prices. Saving events without labels...")
        df_events[['timestamp', 'text', 'url']].to_csv(
            OUTPUT_FILE.replace('.csv', '_unlabeled.csv'), index=False
        )
        return

    # ── 4. Labeling binaire ───────────────────────────────────────
    df_labeled = correlate_events_with_btc(events_list, df_prices)

    if df_labeled.empty:
        print("❌ Labeling failed (no matching price windows).")
        return

    # ── 5. Export au format historical_train.csv ──────────────────
    # Colonnes principales (compatibles avec l'entraînement BERT)
    df_out = df_labeled[['datetime', 'text', 'url', 'label']].copy()
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved {len(df_out)} labeled rows → {OUTPUT_FILE}")

    # ── 6. Statistiques ───────────────────────────────────────────
    n_up   = (df_out['label'] == 1).sum()
    n_down = (df_out['label'] == 0).sum()
    print(f"\n📊 Label distribution:")
    print(f"  UP   (1) = {n_up}  ({100*n_up/len(df_out):.1f}%)")
    print(f"  DOWN (0) = {n_down} ({100*n_down/len(df_out):.1f}%)")

    print(f"\n📌 Sample output:")
    print(df_out.head(10).to_string(index=False))

    # Export debug enrichi (avec prix BTC)
    debug_file = OUTPUT_FILE.replace('.csv', '_debug.csv')
    df_labeled.to_csv(debug_file, index=False)
    print(f"\n🔍 Debug file (with BTC prices) → {debug_file}")


if __name__ == "__main__":
    main()
