# @title Generate ETF Tracker Dashboard
# Run this cell. It will fetch real data and generate "ETF Tracker Dashboard".

!pip -q install yfinance plotly pandas

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import json
import warnings
import sys
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

print("🚀 Starting Dashboard Generation...")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

MIN_MACD_GAP_THRESHOLD = 0.001     # 0.10% Gap
MIN_MACD_DELTA_THRESHOLD = 0.001   # 0.10% Delta (Change)

MOMENTUM_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
SIGNAL_FRESH_DAYS = 10
BREAKOUT_WINDOW = 20
NOTIF_LOOKBACK_DAYS = 5
LOOKBACK_PERIOD = "2y"

universe_data = [
    # ===================== Equity (Select Sector SPDR) =====================
    ("Equity", "Technology", "XLK", "State Street® Technology Select Sector SPDR® ETF"),
    ("Equity", "Communication Services", "XLC", "State Street® Communication Services Select Sector SPDR® ETF"),
    ("Equity", "Consumer Discretionary", "XLY", "State Street® Consumer Discretionary Select Sector SPDR® ETF"),
    ("Equity", "Consumer Staples", "XLP", "State Street® Consumer Staples Select Sector SPDR® ETF"),
    ("Equity", "Health Care", "XLV", "State Street® Health Care Select Sector SPDR® ETF"),
    ("Equity", "Financials", "XLF", "State Street® Financial Select Sector SPDR® ETF"),
    ("Equity", "Industrials", "XLI", "State Street® Industrial Select Sector SPDR® ETF"),
    ("Equity", "Energy", "XLE", "State Street® Energy Select Sector SPDR® ETF"),
    ("Equity", "Materials", "XLB", "State Street® Materials Select Sector SPDR® ETF"),
    ("Equity", "Utilities", "XLU", "State Street® Utilities Select Sector SPDR® ETF"),
    ("Equity", "Real Estate", "XLRE", "State Street® Real Estate Select Sector SPDR® ETF"),
    ("Equity", "Pharmaceutical", "PPH", "VanEck® Pharmaceutical ETF"),

    # ===================== Theme =====================
    ("Theme", "Semiconductors", "SMH", "VanEck® Semiconductor ETF"),
    ("Theme", "Cybersecurity", "CIBR", "First Trust® Nasdaq Cybersecurity ETF"),
    ("Theme", "Clean Energy", "ICLN", "BlackRock iShares® Global Clean Energy ETF"),
    ("Theme", "Biotech", "XBI", "State Street® SPDR® S&P Biotech ETF"),
    ("Theme", "Defense", "ITA", "BlackRock iShares® U.S. Aerospace & Defense ETF"),
    ("Theme", "Tech-Software", "IGV", "BlackRock iShares® Expanded Tech-Software Sector ETF"),

    # ===================== Commodity =====================
    ("Commodity", "Gold", "GLD", "State Street® SPDR® Gold Shares"),
    ("Commodity", "Silver", "SLV", "BlackRock iShares® Silver Trust"),
    ("Commodity", "Gold Miners", "GDX", "VanEck® Gold Miners ETF"),
    ("Commodity", "Broad Commodities", "PDBC", "Invesco® Optimum Yield Diversified Commodity Strategy No K-1 ETF"),
    ("Commodity", "Uranium-Nuclear", "NLR", "VanEck® Uranium and Nuclear ETF"),

    # ===================== Region =====================
    ("Region", "Emerging Markets", "VWO", "Vanguard® FTSE Emerging Markets ETF"),
    # 3033.HK is CSOP Hang Seng TECH Index ETF (HKEX: 3033), NOT Hang Seng Tracker Fund
    ("Region", "Hong Kong", "3033.HK", "CSOP® Hang Seng TECH Index ETF"),
    ("Region", "China", "3188.HK", "ChinaAMC® CSI 300 Index ETF"),
    ("Region", "India", "INDA", "BlackRock iShares® MSCI India ETF"),
    ("Region", "Japan", "EWJ", "BlackRock iShares® MSCI Japan ETF"),
    ("Region", "Europe", "VGK", "Vanguard® FTSE Europe ETF"),
    ("Region", "South Korea", "EWY", "BlackRock iShares® MSCI South Korea ETF"),
    ("Region", "USA", "ITOT", "BlackRock iShares® Core S&P Total U.S. Stock Market ETF"),

    # ===================== Rates =====================
    ("Rates", "Cash (T-Bills)", "USFR", "WisdomTree® Floating Rate Treasury Fund"),
    ("Rates", "Aggregate Bond", "AGG", "BlackRock iShares® Core U.S. Aggregate Bond ETF"),
    ("Rates", "Short Treasuries", "SHY", "BlackRock iShares® 1-3 Year Treasury Bond ETF"),
    ("Rates", "Intermediate Treasuries", "IEF", "BlackRock iShares® 7-10 Year Treasury Bond ETF"),
    ("Rates", "Long Treasuries", "TLT", "BlackRock iShares® 20+ Year Treasury Bond ETF"),
    ("Rates", "TIPS", "TIP", "BlackRock iShares® TIPS Bond ETF"),
    ("Rates", "STRIPS", "GOVZ", "BlackRock iShares® 25+ Year Treasury STRIPS Bond ETF"),
    ("Rates", "Medium Investment Grade Credit", "LQD", "BlackRock iShares® iBoxx $ Investment Grade Corporate Bond ETF"),
    ("Rates", "High Yield Credit", "HYG", "BlackRock iShares® iBoxx $ High Yield Corporate Bond ETF"),
    ("Rates", "Short Investment Grade Credit", "VCSH", "Vanguard® Short-Term Corporate Bond ETF"),

    # ===================== UOBKH All-ETF Portfolio =====================
    ("UOBKH All-ETF Portfolio", "S&P 500", "SPY", "State Street® SPDR® S&P 500® ETF Trust"),
    ("UOBKH All-ETF Portfolio", "Energy", "XLE", "State Street® Energy Select Sector SPDR® ETF"),
    ("UOBKH All-ETF Portfolio", "Health Care", "XLV", "State Street® Health Care Select Sector SPDR® ETF"),
    ("UOBKH All-ETF Portfolio", "Dividends", "SCHD", "Schwab® U.S. Dividend Equity ETF"),
    ("UOBKH All-ETF Portfolio", "Developed Markets", "IDEV", "BlackRock iShares® Core MSCI International Developed Markets ETF"),
    ("UOBKH All-ETF Portfolio", "Emerging Markets ex China", "EMXC", "BlackRock iShares® MSCI Emerging Markets ex China ETF"),
    ("UOBKH All-ETF Portfolio", "China", "3188.HK", "ChinaAMC® CSI 300 Index ETF"),
    ("UOBKH All-ETF Portfolio", "Hong Kong", "3033.HK", "CSOP® Hang Seng TECH Index ETF"),
    ("UOBKH All-ETF Portfolio", "Aggregate Bond", "AGG", "BlackRock iShares® Core U.S. Aggregate Bond ETF"),
    ("UOBKH All-ETF Portfolio", "Short Investment Grade Credit", "VCSH", "Vanguard® Short-Term Corporate Bond ETF"),
]


universe = pd.DataFrame(universe_data, columns=["bucket", "sector", "ticker", "name"])
TICKERS = sorted(universe["ticker"].unique().tolist())

# Create a mapping for Full Names (used in Chart Title)
TICKER_TO_NAME = dict(zip(universe["ticker"], universe["name"]))

# =============================================================================
# 2. CALCULATION LOGIC
# =============================================================================

def calculate_technicals(p: pd.Series):
    if len(p) < 50:
        return None

    last_px = p.iloc[-1]
    res = {"last_px": float(last_px)}

    # Momentum
    for lbl, d in MOMENTUM_WINDOWS.items():
        if len(p) > d:
            res[lbl] = (p.iloc[-1] / p.iloc[-d-1]) - 1.0
        else:
            res[lbl] = np.nan

    # EMAs
    ema10 = p.ewm(span=10, adjust=False).mean()
    ema20 = p.ewm(span=20, adjust=False).mean()
    ema50 = p.ewm(span=50, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()

    # Delta EMA (Distance)
    res["d_ema10"] = (last_px / ema10.iloc[-1]) - 1.0
    res["d_ema20"] = (last_px / ema20.iloc[-1]) - 1.0
    res["d_ema50"] = (last_px / ema50.iloc[-1]) - 1.0
    res["d_ema200"] = (last_px / ema200.iloc[-1]) - 1.0

    # Breakout
    prior_high = p.shift(1).rolling(BREAKOUT_WINDOW).max().iloc[-1]
    res["breakout"] = "↑" if last_px > prior_high else "-"

    # MACD signal
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()

    cross_up = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    cross_dn = (macd.shift(1) >= sig.shift(1)) & (macd < sig)

    if cross_up.tail(SIGNAL_FRESH_DAYS).any():
        res["macd_sig"] = "Fresh↑"
    elif cross_dn.tail(SIGNAL_FRESH_DAYS).any():
        res["macd_sig"] = "Fresh↓"
    else:
        res["macd_sig"] = "↑" if macd.iloc[-1] > sig.iloc[-1] else "↓"

    # --- CROSSOVER LOGIC HELPER ---
    def get_crossover_status(short_ma, long_ma, fresh_days=SIGNAL_FRESH_DAYS):
        up = (short_ma.shift(1) <= long_ma.shift(1)) & (short_ma > long_ma)
        dn = (short_ma.shift(1) >= long_ma.shift(1)) & (short_ma < long_ma)

        if up.tail(fresh_days).any(): return "Fresh↑"
        if dn.tail(fresh_days).any(): return "Fresh↓"
        return "↑" if short_ma.iloc[-1] > long_ma.iloc[-1] else "↓"

    # 10/20
    res["sig_10_20"] = get_crossover_status(ema10, ema20)
    # 20/50
    res["sig_20_50"] = get_crossover_status(ema20, ema50)
    # 50/200 (Replaces Stack)
    res["sig_50_200"] = get_crossover_status(ema50, ema200)

    return res

def run_event_engine(p: pd.Series, ticker: str):
    """
    Events are detected over last NOTIF_LOOKBACK_DAYS.
    """
    if len(p) < NOTIF_LOOKBACK_DAYS + 30:
        return []

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig

    # Gap series = hist / price
    gap_series = hist / p

    # Δ Gap over last 5 trading days
    if len(gap_series) <= NOTIF_LOOKBACK_DAYS:
        return []

    gap_now = gap_series.iloc[-1]
    gap_prev = gap_series.iloc[-(NOTIF_LOOKBACK_DAYS+1)]
    delta_5d = gap_now - gap_prev

    # Detect flips within last 5 days
    cross_up = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    cross_dn = (macd.shift(1) >= sig.shift(1)) & (macd < sig)

    recent_up = cross_up.tail(NOTIF_LOOKBACK_DAYS)
    recent_dn = cross_dn.tail(NOTIF_LOOKBACK_DAYS)

    last_up = recent_up[recent_up].index.max() if recent_up.any() else None
    last_dn = recent_dn[recent_dn].index.max() if recent_dn.any() else None

    events = []

    # --- Flip event ---
    if last_up or last_dn:
        if last_up and (not last_dn or last_up > last_dn):
            direction = "Bullish"
            evt_date = last_up
        else:
            direction = "Bearish"
            evt_date = last_dn

        if pd.notna(gap_now) and pd.notna(delta_5d):
            if (abs(gap_now) >= MIN_MACD_GAP_THRESHOLD) and (abs(delta_5d) >= MIN_MACD_DELTA_THRESHOLD):
                events.append({
                    "ticker": ticker,
                    "type": "Flip",
                    "dir": direction,
                    "date": evt_date,
                    "gap": float(gap_now),
                    "delta": float(delta_5d)
                })

    # --- Momentum state today ---
    h_now = hist.iloc[-1]
    direction = None
    if h_now > 0 and delta_5d > 0:
        direction = "More Bullish"
    elif h_now > 0 and delta_5d < 0:
        direction = "Less Bullish"
    elif h_now < 0 and delta_5d < 0:
        direction = "More Bearish"
    elif h_now < 0 and delta_5d > 0:
        direction = "Less Bearish"

    if direction:
        if (abs(gap_now) >= MIN_MACD_GAP_THRESHOLD) and (abs(delta_5d) >= MIN_MACD_DELTA_THRESHOLD):
            events.append({
                "ticker": ticker,
                "type": "Momentum",
                "dir": direction,
                "date": p.index[-1],
                "gap": float(gap_now),
                "delta": float(delta_5d)
            })

    return events

# =============================================================================
# 3. DATA ACQUISITION (ETFs)
# =============================================================================
print(f"📥 Downloading Price History for {len(TICKERS)} ETFs...")
raw_data = yf.download(
    TICKERS, period=LOOKBACK_PERIOD, interval="1d",
    group_by='ticker', auto_adjust=True, progress=False
)

tracker_rows = []
update_events = []
etf_history_db = {}
valid_tickers = []

print("⚙️ Processing Price Data...")
for t in TICKERS:
    try:
        df = pd.DataFrame()
        if isinstance(raw_data.columns, pd.MultiIndex):
            if t in raw_data.columns.levels[0]:
                df = raw_data[t].copy()
        else:
            if len(TICKERS) == 1 and t == TICKERS[0]:
                df = raw_data.copy()

        if df.empty or 'Close' not in df.columns:
            continue

        p = df['Close'].dropna()
        if len(p) < 50:
            continue

        etf_history_db[t] = df
        valid_tickers.append(t)

        metrics = calculate_technicals(p)
        if metrics:
            metrics['ticker'] = t
            tracker_rows.append(metrics)

        update_events.extend(run_event_engine(p, t))

    except Exception as e:
        print(f"   Skipping {t}: {e}")

df_tracker = pd.DataFrame(tracker_rows).merge(universe, on="ticker", how="left")
df_updates = pd.DataFrame(update_events).merge(
    universe[['ticker', 'bucket']], on="ticker", how="left"
) if update_events else pd.DataFrame()

# =============================================================================
# 4. HOLDINGS EXTRACTION
# =============================================================================
print("🔍 Fetching Top 10 Holdings...")
etf_holdings_map = {}
count = 0

for t in TICKERS:
    count += 1
    sys.stdout.write(f"\r   Fetching {t} ({count}/{len(TICKERS)})")
    sys.stdout.flush()
    try:
        tk = yf.Ticker(t)
        if hasattr(tk, 'funds_data') and tk.funds_data:
            h = tk.funds_data.top_holdings
            if h is not None and not h.empty:
                h = h.reset_index()
                top_10 = []
                for _, r in h.head(10).iterrows():
                    top_10.append({
                        'symbol': str(r.get('Symbol', '')).strip(),
                        'name': str(r.get('Name', '')).strip(),
                        'weight': float(r.get('Holding Percent', np.nan))
                    })
                etf_holdings_map[t] = top_10
    except:
        etf_holdings_map[t] = []

print("\n✅ Holdings fetched.")

# =============================================================================
# 4B. HOLDINGS TECHNICALS
# =============================================================================
def normalize_holding_symbol(sym: str, parent_etf: str) -> str:
    if not sym:
        return sym
    s = sym.strip()
    if parent_etf.endswith(".HK"):
        if re.fullmatch(r"\d{3,5}", s):
            return f"{s}.HK"
    return s

all_holdings = []
for etf_t, holdings in etf_holdings_map.items():
    for h in holdings:
        norm = normalize_holding_symbol(h.get("symbol", ""), etf_t)
        if norm:
            all_holdings.append(norm)

all_holdings = sorted(list(set(all_holdings)))
holding_metrics_map = {}

if all_holdings:
    print(f"📥 Downloading Price History for {len(all_holdings)} unique holding tickers...")
    raw_hold = yf.download(
        all_holdings, period=LOOKBACK_PERIOD, interval="1d",
        group_by='ticker', auto_adjust=True, progress=False
    )

    def extract_series(raw, ticker):
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.levels[0]:
                d = raw[ticker].copy()
                if 'Close' in d.columns:
                    return d['Close'].dropna()
        else:
            if len(all_holdings) == 1 and ticker == all_holdings[0]:
                if 'Close' in raw.columns:
                    return raw['Close'].dropna()
        return None

    print("⚙️ Computing holding technicals...")
    for ht in all_holdings:
        try:
            p = extract_series(raw_hold, ht)
            if p is None or len(p) < 50:
                continue
            m = calculate_technicals(p)
            if m:
                holding_metrics_map[ht] = m
        except:
            continue

# =============================================================================
# 5. HTML HELPERS
# =============================================================================
COLORS = {'green': '#047857', 'red': '#b91c1c', 'gray': '#6b7280', 'blue': '#1d4ed8'}

def style_val(val, type='pct', colored=True):
    if pd.isna(val):
        return "-"
    txt = f"{val:.1%}" if type == 'pct' else f"{val:.2f}"
    if not colored:
        return txt
    c = COLORS['gray']
    if val > 0:
        c = COLORS['green']
    elif val < 0:
        c = COLORS['red']
    return f'<span style="color:{c}">{txt}</span>'

def style_sig(val):
    base_style = "display:inline-block; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:11px;"
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '<span style="color:#d1d5db">-</span>'
    val = str(val)
    if "Fresh" in val:
        if "↑" in val:
            return f'<span style="{base_style} background:#d1fae5; color:#065f46;">Fresh↑</span>'
        if "↓" in val:
            return f'<span style="{base_style} background:#fee2e2; color:#991b1b;">Fresh↓</span>'
    if "↑" in val:
        return f'<span style="{base_style} background:#dcfce7; color:#166534;">↑</span>'
    if "↓" in val:
        return f'<span style="{base_style} background:#fce7f3; color:#9d174d;">↓</span>'
    return '<span style="color:#d1d5db">-</span>'

def make_ticker_link(ticker):
    """Creates a clickable link that switches to the chart tab and selects the ticker."""
    return f'<a href="#" onclick="goToChart(\'{ticker}\'); return false;" style="color:#1d4ed8; text-decoration:none; border-bottom:1px dotted #1d4ed8;">{ticker}</a>'

# Updated ColGroup: Replaced Stack with 50/200 width
BUCKET_COLGROUP = """
<colgroup>
  <col style="width:72px;">   <col style="width:170px;">  <col style="width:70px;">   <col style="width:70px;">   <col style="width:70px;">   <col style="width:70px;">   <col style="width:80px;">   <col style="width:80px;">   <col style="width:80px;">   <col style="width:90px;">   <col style="width:70px;">   <col style="width:70px;">   <col style="width:70px;">   <col style="width:70px;">   <col style="width:60px;">   </colgroup>
"""

BUCKET_FOOTNOTE = (
    "* ∆EMA = (Price / EMA) − 1 | Fresh↑ / Fresh↓ = crossover within last 10 trading days | "
    "50/200 = EMA50 vs EMA200 (Golden/Death Cross) | MACD = MACD(12,26,9) line vs signal | Breakout = close > prior 20D high."
)

UPDATES_FOOTNOTE = (
    "*MACD Gap (%Px) = (MACD line − Signal line) ÷ Price | "
    "∆MACD Gap (%Px) = How much MACD Gap widened/narrowed over the last 5 trading days | "
    "Px = ETF Price"
)

# =============================================================================
# 6. HTML: TRACKER + UPDATES + HOLDINGS
# =============================================================================
tracker_html = ""
buckets = ["Equity", "Theme", "Region", "Rates", "Commodity", "UOBKH All-ETF Portfolio"]

for bucket in buckets:
    sub = df_tracker[df_tracker['bucket'] == bucket]
    if sub.empty:
        continue

    tracker_html += f"<div class='bucket-title'>{bucket}</div>"
    tracker_html += f"<table class='data-table bucket-table'>{BUCKET_COLGROUP}<thead><tr>"

    tracker_html += "<th>Ticker</th><th>Sector</th>"
    tracker_html += "<th class='vdiv'>1M</th><th>3M</th><th>6M</th><th>12M</th>"
    tracker_html += "<th class='vdiv'>ΔEMA10</th><th>ΔEMA20</th><th>ΔEMA50</th><th>ΔEMA200</th>"
    tracker_html += "<th class='vdiv'>10/20</th><th>20/50</th><th>50/200</th><th>MACD</th><th>BO</th>"
    tracker_html += "</tr></thead><tbody>"

    for _, r in sub.iterrows():
        tracker_html += "<tr>"
        tracker_html += f"<td><b>{make_ticker_link(r['ticker'])}</b></td>"
        tracker_html += f"<td>{r['sector']}</td>"

        tracker_html += f"<td class='vdiv'>{style_val(r.get('1M'), 'pct')}</td>"
        tracker_html += f"<td>{style_val(r.get('3M'), 'pct')}</td>"
        tracker_html += f"<td>{style_val(r.get('6M'), 'pct')}</td>"
        tracker_html += f"<td>{style_val(r.get('12M'), 'pct')}</td>"

        tracker_html += f"<td class='vdiv'>{style_val(r.get('d_ema10'), 'pct')}</td>"
        tracker_html += f"<td>{style_val(r.get('d_ema20'), 'pct')}</td>"
        tracker_html += f"<td>{style_val(r.get('d_ema50'), 'pct')}</td>"
        tracker_html += f"<td>{style_val(r.get('d_ema200'), 'pct')}</td>"

        tracker_html += f"<td class='vdiv' style='text-align:center'>{style_sig(r.get('sig_10_20','-'))}</td>"
        tracker_html += f"<td style='text-align:center'>{style_sig(r.get('sig_20_50','-'))}</td>"
        tracker_html += f"<td style='text-align:center'>{style_sig(r.get('sig_50_200','-'))}</td>"
        tracker_html += f"<td style='text-align:center'>{style_sig(r.get('macd_sig','-'))}</td>"
        tracker_html += f"<td style='text-align:center'>{style_sig(r.get('breakout','-'))}</td>"
        tracker_html += "</tr>"

    tracker_html += "</tbody></table>"
    tracker_html += f"<div class='footnote'>{BUCKET_FOOTNOTE}</div>"

updates_html = (
    f"<h3 style='margin-top:0'>MACD Event Updates Over The Last 5 Trading Days "
    f"(Gap &gt; {MIN_MACD_GAP_THRESHOLD:.1%} AND ΔGap &gt; {MIN_MACD_DELTA_THRESHOLD:.1%})</h3>"
)

if not df_updates.empty:
    updates_html += """
    <table class='update-table'>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Bucket</th>
                <th style='width:40%'>Update</th>
                <th style='text-align:right'>MACD Gap (%Px)</th>
                <th style='text-align:right'>ΔMACD Gap (%Px)</th>
            </tr>
        </thead>
        <tbody>
    """

    df_updates['abs_gap'] = df_updates['gap'].abs()

    def format_desc(row):
        c = "#047857" if "Bullish" in str(row['dir']) else "#b91c1c"
        d_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        if row['type'] == 'Flip':
            return f"MACD Flipped <b style='color:{c}'>{row['dir']}</b> on {d_str}"
        else:
            return f"Momentum <b style='color:{c}'>{row['dir']}</b> on {d_str}"

    df_updates['desc_html'] = df_updates.apply(format_desc, axis=1)

    df_grouped = df_updates.groupby('ticker', as_index=False).agg({
        'bucket': 'first',
        'gap': 'first',
        'delta': 'first',
        'abs_gap': 'first',
        'desc_html': lambda x: '<br>'.join(x)
    })

    df_grouped = df_grouped.sort_values('abs_gap', ascending=False)

    for _, r in df_grouped.iterrows():
        gap_val = float(r['gap'])
        gap_color = "#047857" if gap_val > 0 else "#b91c1c"
        delta_val = float(r['delta'])

        updates_html += f"""
        <tr>
            <td><b>{make_ticker_link(r['ticker'])}</b></td>
            <td>{r['bucket']}</td>
            <td>{r['desc_html']}</td>
            <td style='text-align:right; font-weight:bold; color:{gap_color}'>{gap_val:.2%}</td>
            <td style='text-align:right'>{delta_val:.2%}</td>
        </tr>
        """

    updates_html += "</tbody></table>"
else:
    updates_html += "<p>No events found matching strict criteria.</p>"

updates_html += f"<div class='footnote'>{UPDATES_FOOTNOTE}</div><br>"

# --- HOLDINGS TABLE GENERATION ---
holdings_html_map = {}

def holding_row_html(ticker, name, weight, m):
    w_disp = f"{weight*100:.2f}%" if pd.notna(weight) else "-"
    if not m:
        return f"""
        <tr>
            <td><b>{ticker}</b></td>
            <td>{name}</td>
            <td style="text-align:right">{w_disp}</td>
            <td class="vdiv">-</td><td>-</td><td>-</td><td>-</td>
            <td class="vdiv">-</td><td>-</td><td>-</td><td>-</td><td>-</td>
        </tr>
        """

    return f"""
    <tr>
        <td><b>{ticker}</b></td>
        <td>{name}</td>
        <td style="text-align:right">{w_disp}</td>

        <td class="vdiv">{style_val(m.get('1M'), 'pct')}</td>
        <td>{style_val(m.get('3M'), 'pct')}</td>
        <td>{style_val(m.get('6M'), 'pct')}</td>
        <td>{style_val(m.get('12M'), 'pct')}</td>

        <td class="vdiv" style="text-align:center">{style_sig(m.get('sig_10_20','-'))}</td>
        <td style="text-align:center">{style_sig(m.get('sig_20_50','-'))}</td>
        <td style="text-align:center">{style_sig(m.get('sig_50_200','-'))}</td>
        <td style="text-align:center">{style_sig(m.get('macd_sig','-'))}</td>
        <td style="text-align:center">{style_sig(m.get('breakout','-'))}</td>
    </tr>
    """

# --- CUSTOMIZABLE COLUMN WIDTHS (in pixels) ---
COL_W_SYMBOL = "85px"
COL_W_NAME   = "160px"  # Reduced to ensure fit (truncates if too long)
COL_W_WEIGHT = "65px"
COL_W_MOM    = "50px"   # 1M, 3M, 6M, 12M
COL_W_TECH   = "60px"   # 10/20, 20/50, 50/200, MACD
COL_W_BO     = "50px"   # Breakout

HOLDINGS_COLGROUP = f"""
<colgroup>
    <col style="width:{COL_W_SYMBOL};">
    <col style="width:{COL_W_NAME};">
    <col style="width:{COL_W_WEIGHT};">
    <col style="width:{COL_W_MOM};"><col style="width:{COL_W_MOM};"><col style="width:{COL_W_MOM};"><col style="width:{COL_W_MOM};">
    <col style="width:{COL_W_TECH};"><col style="width:{COL_W_TECH};"><col style="width:{COL_W_TECH};"><col style="width:{COL_W_TECH};">
    <col style="width:{COL_W_BO};">
</colgroup>
"""

for etf_ticker in TICKERS:
    holdings = etf_holdings_map.get(etf_ticker, [])
    h_html = f"<div class='holdings-header'>Top 10 Holdings: <b>{etf_ticker}</b></div>"

    if not holdings:
        h_html += "<p style='color:gray; font-size:12px; padding:10px;'><i>Holdings data unavailable via API.</i></p>"
    else:
        h_html += f"""
        <table class='data-table' style='font-size:12px; margin-top:5px; table-layout:fixed; width:100%;'>
            {HOLDINGS_COLGROUP}
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Name</th>
                    <th style="text-align:right;">Weight</th>
                    <th class="vdiv">1M</th><th>3M</th><th>6M</th><th>12M</th>
                    <th class="vdiv" style="text-align:center">10/20</th>
                    <th style="text-align:center">20/50</th>
                    <th style="text-align:center">50/200</th>
                    <th style="text-align:center">MACD</th>
                    <th style="text-align:center">BO</th>
                </tr>
            </thead>
            <tbody>
        """
        for item in holdings:
            raw_sym = item.get('symbol', '')
            sym = normalize_holding_symbol(raw_sym, etf_ticker)
            nm = item.get('name', '')
            wt = item.get('weight', np.nan)
            m = holding_metrics_map.get(sym)
            h_html += holding_row_html(sym, nm, wt, m)

        h_html += "</tbody></table>"

    holdings_html_map[etf_ticker] = h_html

# =============================================================================
# 7. CHART GENERATION
# =============================================================================
print("📈 Generating Interactive Chart...")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)

TRACES_PER_ETF = 8

for i, t in enumerate(valid_tickers):
    df = etf_history_db.get(t, pd.DataFrame())
    if df.empty:
        for _ in range(TRACES_PER_ETF):
            fig.add_trace(go.Scatter(x=[], y=[]))
        continue

    ema10 = df['Close'].ewm(span=10).mean()
    ema20 = df['Close'].ewm(span=20).mean()
    ema50 = df['Close'].ewm(span=50).mean()
    ema200 = df['Close'].ewm(span=200).mean()

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    hist = macd - sig

    viz = (i == 0)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', visible=viz, showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=ema10, line=dict(color='#3b82f6', width=1),
                             name='EMA10', visible=viz, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema20, line=dict(color='#8b5cf6', width=1),
                             name='EMA20', visible=viz, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, line=dict(color='#f59e0b', width=1),
                             name='EMA50', visible=viz, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, line=dict(color='#ef4444', width=1),
                             name='EMA200', visible=viz, showlegend=True), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=np.where(hist>=0, '#22c55e', '#ef4444'),
                         name='Histogram', visible=viz, showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd, line=dict(color='#3b82f6', width=1),
                             name='MACD', visible=viz, showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sig, line=dict(color='#f59e0b', width=1),
                             name='Signal', visible=viz, showlegend=True), row=2, col=1)

fig.update_layout(
    height=650,
    margin=dict(l=40, r=20, t=20, b=20),
    template="plotly_white",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_layout(xaxis_rangeslider_visible=False)

chart_div = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='plotly_chart')

# =============================================================================
# 8. ASSEMBLE HTML
# =============================================================================
dropdown_options = ""
for t in valid_tickers:
    dropdown_options += f'<option value="{t}">{t}</option>'

holdings_json = json.dumps(holdings_html_map)
tickers_json = json.dumps(valid_tickers)
names_json = json.dumps(TICKER_TO_NAME)  # Pass full names to JS

final_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>ETF Strategy Dashboard</title>
<style>
    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f3f4f6; margin: 0; padding: 20px; color: #111827; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}

    .tab-header {{ display: flex; border-bottom: 2px solid #e5e7eb; margin-bottom: 20px; }}
    .tab-btn {{ background: none; border: none; padding: 12px 24px; font-size: 16px; font-weight: 600; color: #6b7280; cursor: pointer; transition: all 0.2s; }}
    .tab-btn:hover {{ color: #111827; background: #f9fafb; }}
    .tab-btn.active {{ color: #2563eb; border-bottom: 2px solid #2563eb; margin-bottom: -2px; }}
    .tab-content {{ display: none; animation: fadeIn 0.4s; }}
    .tab-content.active {{ display: block; }}

    .bucket-title {{ background: #eef2ff; color: #111827; font-weight: bold; font-size: 14px; padding: 8px 12px; margin-top: 15px; border-left: 4px solid #c7d2fe; }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 6px;
        table-layout: fixed;
    }}
    .data-table th {{
        background: #111827;
        color: white;
        padding: 8px;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .data-table td {{
        border-bottom: 1px solid #e5e7eb;
        padding: 6px 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .data-table tr:nth-child(even) {{ background: #f9fafb; }}

    .vdiv {{ border-left: 2px solid #e5e7eb !important; }}
    .data-table th.vdiv {{ border-left: 2px solid #334155 !important; }}

    .update-table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'Helvetica Neue', sans-serif; }}
    .update-table th {{ background-color: #0f172a; color: white; padding: 10px; text-align: left; font-weight: 600; }}
    .update-table td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }}
    .update-table tr:nth-child(even) {{ background-color: #f8fafc; }}
    .update-table tr:hover {{ background-color: #f1f5f9; }}

    .controls {{ margin-bottom: 10px; padding: 15px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: space-between; }}
    select {{ padding: 8px; border-radius: 4px; border: 1px solid #d1d5db; font-size: 14px; width: 200px; }}

    .holdings-header {{ margin-top: 10px; font-weight: 600; }}

    .footnote {{
        font-size: 11px;
        color: #6b7280;
        margin: 4px 2px 12px 2px;
    }}

    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
</style>
</head>
<body>

<div class="container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 style="margin:0">📊 ETF Strategic Dashboard</h2>
        <span style="color:#6b7280; font-size:12px">Generated: {date.today()}</span>
    </div>
    <br>

    <div class="tab-header">
        <button id="btn-tracker" class="tab-btn active" onclick="openTab('tracker')">📋 ETF Tracker</button>
        <button id="btn-analyzer" class="tab-btn" onclick="openTab('analyzer')">📈 Chart Analyser</button>
    </div>

    <div id="tracker" class="tab-content active">
        {updates_html}
        {tracker_html}
    </div>

    <div id="analyzer" class="tab-content">
        <div class="controls">
            <div class="select-wrapper">
                <label>Select ETF:</label>
                <select id="etfSelector" onchange="updateChart()">{dropdown_options}</select>
            </div>
            <div style="font-size:12px; color:gray;">Updates Chart & Holdings</div>
        </div>
        <h3 id="chartTitle" style="margin-left:40px; margin-bottom:0;">Technical Analysis</h3>
        {chart_div}
        <hr style="margin: 20px 0; border:0; border-top:1px solid #e5e7eb;">
        <div id="holdingsContainer"></div>
    </div>
</div>

<script>
    var holdingsData = {holdings_json};
    var tickers = {tickers_json};
    var etfNames = {names_json};
    var tracesPerItem = {TRACES_PER_ETF};

    function openTab(tabName) {{
        var i, x, tablinks;
        x = document.getElementsByClassName("tab-content");
        for (i = 0; i < x.length; i++) {{ x[i].className = x[i].className.replace(" active", ""); }}
        tablinks = document.getElementsByClassName("tab-btn");
        for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}

        document.getElementById(tabName).className += " active";
        document.getElementById("btn-tracker").className = (tabName === 'tracker') ? "tab-btn active" : "tab-btn";
        document.getElementById("btn-analyzer").className = (tabName === 'analyzer') ? "tab-btn active" : "tab-btn";

        window.dispatchEvent(new Event('resize'));
    }}

    function goToChart(ticker) {{
        // 1. Switch Tab
        openTab('analyzer');

        // 2. Set Dropdown Value
        var selector = document.getElementById("etfSelector");
        selector.value = ticker;

        // 3. Update Chart
        updateChart();

        // 4. Scroll to top of analyzer
        document.getElementById("analyzer").scrollIntoView({{behavior: 'smooth'}});
    }}

    function updateChart() {{
        var selector = document.getElementById("etfSelector");
        var selectedTicker = selector.value;
        var selectedIndex = tickers.indexOf(selectedTicker);

        // Update Title with Full Name
        var fullName = etfNames[selectedTicker] || selectedTicker;
        document.getElementById("chartTitle").innerText = fullName + " Technical Analysis";

        // Update Holdings Table
        var container = document.getElementById("holdingsContainer");
        container.innerHTML = holdingsData[selectedTicker] || "<p>No holdings data.</p>";

        // Update Plotly Trace Visibility
        var totalTraces = tickers.length * tracesPerItem;
        var visibilityArray = new Array(totalTraces).fill(false);

        var startIdx = selectedIndex * tracesPerItem;
        for (var i = 0; i < tracesPerItem; i++) {{
            visibilityArray[startIdx + i] = true;
        }}

        var plotDiv = document.getElementById('plotly_chart');
        Plotly.restyle(plotDiv, {{'visible': visibilityArray}});
    }}

    window.onload = function() {{
        // Initialize with the first ticker in the list
        var first = tickers[0];
        if(first) {{
            document.getElementById("etfSelector").value = first;
            updateChart();
        }}
    }};
</script>
</body>
</html>
"""

filename = "ETF_Tracker_Dashboard.html"
with open(filename, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"✅ Success! Dashboard saved as: {filename}")

try:
    from google.colab import files
    files.download(filename)
except:
    pass
