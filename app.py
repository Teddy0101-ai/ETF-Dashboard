import re
import json
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.simplefilter(action="ignore", category=FutureWarning)

# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(
    page_title="ETF Strategy Dashboard",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# 1) CONFIGURATION (same logic as your notebook)
# =============================================================================
MIN_MACD_GAP_THRESHOLD = 0.001     # 0.10% Gap
MIN_MACD_DELTA_THRESHOLD = 0.001   # 0.10% Delta (Change)

MOMENTUM_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
SIGNAL_FRESH_DAYS = 10
BREAKOUT_WINDOW = 20
NOTIF_LOOKBACK_DAYS = 5
LOOKBACK_PERIOD = "2y"


# =============================================================================
# 2) UNIVERSE
# =============================================================================
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
TICKER_TO_NAME = dict(zip(universe["ticker"], universe["name"]))


# =============================================================================
# 3) CALCULATION LOGIC
# =============================================================================
def calculate_technicals(p: pd.Series) -> dict | None:
    if len(p) < 50:
        return None

    last_px = p.iloc[-1]
    res = {"last_px": float(last_px)}

    # Momentum
    for lbl, d in MOMENTUM_WINDOWS.items():
        if len(p) > d:
            res[lbl] = (p.iloc[-1] / p.iloc[-d - 1]) - 1.0
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

    def get_crossover_status(short_ma, long_ma, fresh_days=SIGNAL_FRESH_DAYS):
        up = (short_ma.shift(1) <= long_ma.shift(1)) & (short_ma > long_ma)
        dn = (short_ma.shift(1) >= long_ma.shift(1)) & (short_ma < long_ma)
        if up.tail(fresh_days).any():
            return "Fresh↑"
        if dn.tail(fresh_days).any():
            return "Fresh↓"
        return "↑" if short_ma.iloc[-1] > long_ma.iloc[-1] else "↓"

    res["sig_10_20"] = get_crossover_status(ema10, ema20)
    res["sig_20_50"] = get_crossover_status(ema20, ema50)
    res["sig_50_200"] = get_crossover_status(ema50, ema200)

    return res


def run_event_engine(p: pd.Series, ticker: str) -> list[dict]:
    if len(p) < NOTIF_LOOKBACK_DAYS + 30:
        return []

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig

    gap_series = hist / p
    if len(gap_series) <= NOTIF_LOOKBACK_DAYS:
        return []

    gap_now = gap_series.iloc[-1]
    gap_prev = gap_series.iloc[-(NOTIF_LOOKBACK_DAYS + 1)]
    delta_5d = gap_now - gap_prev

    cross_up = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    cross_dn = (macd.shift(1) >= sig.shift(1)) & (macd < sig)

    recent_up = cross_up.tail(NOTIF_LOOKBACK_DAYS)
    recent_dn = cross_dn.tail(NOTIF_LOOKBACK_DAYS)

    last_up = recent_up[recent_up].index.max() if recent_up.any() else None
    last_dn = recent_dn[recent_dn].index.max() if recent_dn.any() else None

    events = []

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
                    "delta": float(delta_5d),
                })

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

    if direction and pd.notna(gap_now) and pd.notna(delta_5d):
        if (abs(gap_now) >= MIN_MACD_GAP_THRESHOLD) and (abs(delta_5d) >= MIN_MACD_DELTA_THRESHOLD):
            events.append({
                "ticker": ticker,
                "type": "Momentum",
                "dir": direction,
                "date": p.index[-1],
                "gap": float(gap_now),
                "delta": float(delta_5d),
            })

    return events


def normalize_holding_symbol(sym: str, parent_etf: str) -> str:
    if not sym:
        return sym
    s = str(sym).strip()
    if parent_etf.endswith(".HK"):
        if re.fullmatch(r"\d{3,5}", s):
            return f"{s}.HK"
    return s


# =============================================================================
# 4) DATA ACQUISITION (cached)
# =============================================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 min cache
def fetch_price_history(tickers: list[str], period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return raw


def extract_close_series(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.levels[0]:
            d = raw[ticker].copy()
            if "Close" in d.columns:
                s = d["Close"].dropna()
                return s if not s.empty else None
        return None

    # Single ticker case
    if "Close" in raw.columns:
        s = raw["Close"].dropna()
        return s if not s.empty else None
    return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)  # 6h cache
def fetch_holdings_top10(etf_ticker: str) -> list[dict]:
    try:
        tk = yf.Ticker(etf_ticker)
        if hasattr(tk, "funds_data") and tk.funds_data:
            h = tk.funds_data.top_holdings
            if h is not None and not h.empty:
                h = h.reset_index()
                out = []
                for _, r in h.head(10).iterrows():
                    out.append({
                        "symbol": str(r.get("Symbol", "")).strip(),
                        "name": str(r.get("Name", "")).strip(),
                        "weight": float(r.get("Holding Percent", np.nan)) if pd.notna(r.get("Holding Percent", np.nan)) else np.nan,
                    })
                return out
    except Exception:
        pass
    return []


@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 min cache
def fetch_holdings_price_and_metrics(holding_tickers: list[str], period: str) -> dict:
    if not holding_tickers:
        return {}

    raw = yf.download(
        holding_tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    metrics_map = {}
    for ht in holding_tickers:
        try:
            s = extract_close_series(raw, ht)
            if s is None or len(s) < 50:
                continue
            m = calculate_technicals(s)
            if m:
                metrics_map[ht] = m
        except Exception:
            continue
    return metrics_map


def build_tracker_and_updates(raw_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracker_rows = []
    update_events = []
    for t in TICKERS:
        s = extract_close_series(raw_prices, t)
        if s is None or len(s) < 50:
            continue

        m = calculate_technicals(s)
        if m:
            m["ticker"] = t
            tracker_rows.append(m)

        update_events.extend(run_event_engine(s, t))

    df_tracker = pd.DataFrame(tracker_rows).merge(universe, on="ticker", how="left")

    if update_events:
        df_updates = pd.DataFrame(update_events).merge(universe[["ticker", "bucket"]], on="ticker", how="left")
        df_updates["abs_gap"] = df_updates["gap"].abs()
        df_updates = df_updates.sort_values("abs_gap", ascending=False)
    else:
        df_updates = pd.DataFrame()

    return df_tracker, df_updates


# =============================================================================
# 5) UI helpers
# =============================================================================
def style_pct(v):
    if pd.isna(v):
        return ""
    if v > 0:
        return "color: #047857; font-weight: 600;"
    if v < 0:
        return "color: #b91c1c; font-weight: 600;"
    return "color: #6b7280;"


def style_sig_cell(v):
    # return CSS for signal-like columns
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "color:#9ca3af;"
    s = str(v)
    if "Fresh↑" in s:
        return "background-color:#d1fae5; color:#065f46; font-weight:700; text-align:center;"
    if "Fresh↓" in s:
        return "background-color:#fee2e2; color:#991b1b; font-weight:700; text-align:center;"
    if "↑" == s:
        return "background-color:#dcfce7; color:#166534; font-weight:700; text-align:center;"
    if "↓" == s:
        return "background-color:#fce7f3; color:#9d174d; font-weight:700; text-align:center;"
    if "-" == s:
        return "color:#9ca3af; text-align:center;"
    return "text-align:center;"


def format_pct(x):
    return "-" if pd.isna(x) else f"{x:.1%}"


def build_deeplink(ticker: str) -> str:
    return f"?ticker={ticker}&tab=analyzer"


def make_price_chart(raw_prices: pd.DataFrame, ticker: str) -> go.Figure | None:
    if isinstance(raw_prices.columns, pd.MultiIndex):
        if ticker not in raw_prices.columns.levels[0]:
            return None
        df = raw_prices[ticker].copy()
    else:
        df = raw_prices.copy()

    if df.empty or "Close" not in df.columns:
        return None

    df = df.dropna(subset=["Close"])
    if df.empty:
        return None

    ema10 = df["Close"].ewm(span=10).mean()
    ema20 = df["Close"].ewm(span=20).mean()
    ema50 = df["Close"].ewm(span=50).mean()
    ema200 = df["Close"].ewm(span=200).mean()

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    hist = macd - sig

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.03
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price", showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=ema10, name="EMA10", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, name="EMA200", line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sig, name="Signal", line=dict(width=1)), row=2, col=1)

    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=10, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# =============================================================================
# Header + controls
# =============================================================================
st.markdown(
    """
    <style>
      .small-muted { color:#6b7280; font-size:12px; }
      a.ticker-link { color:#1d4ed8; text-decoration:none; border-bottom:1px dotted #1d4ed8; }
      a.ticker-link:hover { opacity:0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1, 1], vertical_alignment="center")
with col_a:
    st.title("📊 ETF Strategic Dashboard")
with col_b:
    st.markdown(f"<div style='text-align:right' class='small-muted'>Generated: {date.today()}</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    ttl_min = st.slider("Cache TTL (minutes)", min_value=5, max_value=180, value=30, step=5)
    st.caption("This controls refresh frequency on Streamlit Cloud.")
    st.divider()
    bucket_filter = st.multiselect("Buckets", options=sorted(universe["bucket"].unique().tolist()),
                                   default=sorted(universe["bucket"].unique().tolist()))
    show_holdings = st.checkbox("Show holdings (selected ETF only)", value=True)
    st.divider()
    refresh_now = st.button("🔄 Refresh data now (clear cache)", use_container_width=True)

if refresh_now:
    st.cache_data.clear()
    st.rerun()

# apply TTL slider (simple trick: store it & use in fetch wrappers)
# (We keep fetch functions cached; slider is mainly UX. If you want TTL truly dynamic,
# use different cache keys; keeping it simple here.)

# =============================================================================
# Load data
# =============================================================================
with st.spinner(f"Downloading prices for {len(TICKERS)} ETFs..."):
    raw_prices = fetch_price_history(TICKERS, LOOKBACK_PERIOD)

df_tracker, df_updates = build_tracker_and_updates(raw_prices)

# Apply bucket filter
if bucket_filter:
    df_tracker = df_tracker[df_tracker["bucket"].isin(bucket_filter)].copy()
    if not df_updates.empty:
        df_updates = df_updates[df_updates["bucket"].isin(bucket_filter)].copy()

# =============================================================================
# Deep-link / query param handling
# =============================================================================
qp = st.query_params
default_ticker = qp.get("ticker", [None])[0] if isinstance(qp.get("ticker"), list) else qp.get("ticker")
default_tab = qp.get("tab", ["tracker"])[0] if isinstance(qp.get("tab"), list) else qp.get("tab")

if default_ticker not in TICKERS:
    default_ticker = TICKERS[0] if TICKERS else None
if default_tab not in ["tracker", "analyzer"]:
    default_tab = "tracker"

tab_labels = ["📋 ETF Tracker", "📈 Chart Analyzer"]
tab_tracker, tab_analyzer = st.tabs(tab_labels)

# =============================================================================
# TAB 1: Tracker
# =============================================================================
with tab_tracker:
    st.subheader("MACD Event Updates (Last 5 trading days)")
    st.caption(
        f"Filter: MACD Gap > {MIN_MACD_GAP_THRESHOLD:.1%} AND ΔGap > {MIN_MACD_DELTA_THRESHOLD:.1%} "
        f"(Gap = (MACD − Signal)/Price)"
    )

    if df_updates.empty:
        st.info("No events found matching strict criteria.")
    else:
        show_updates = df_updates.copy()
        show_updates["date"] = pd.to_datetime(show_updates["date"]).dt.strftime("%Y-%m-%d")
        show_updates["gap"] = show_updates["gap"].map(lambda x: f"{x:.2%}")
        show_updates["delta"] = show_updates["delta"].map(lambda x: f"{x:.2%}")
        st.dataframe(
            show_updates[["ticker", "bucket", "type", "dir", "date", "gap", "delta"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("ETF Tracker")

    if df_tracker.empty:
        st.warning("No tracker rows available (price history missing).")
    else:
        # Build a display table + deep links
        disp = df_tracker.copy()
        disp["Link"] = disp["ticker"].apply(lambda t: f"[{t}]({build_deeplink(t)})")
        disp = disp[[
            "Link", "bucket", "sector",
            "1M", "3M", "6M", "12M",
            "d_ema10", "d_ema20", "d_ema50", "d_ema200",
            "sig_10_20", "sig_20_50", "sig_50_200", "macd_sig", "breakout"
        ]].rename(columns={"Link": "Ticker"})

        # Format
        pct_cols = ["1M", "3M", "6M", "12M", "d_ema10", "d_ema20", "d_ema50", "d_ema200"]
        sig_cols = ["sig_10_20", "sig_20_50", "sig_50_200", "macd_sig", "breakout"]

        sty = disp.style
        for c in pct_cols:
            sty = sty.format({c: format_pct})
            sty = sty.applymap(lambda v: style_pct(v) if pd.notna(v) else "", subset=[c])

        for c in sig_cols:
            sty = sty.applymap(style_sig_cell, subset=[c])

        sty = sty.set_properties(subset=["Ticker"], **{"font-weight": "700"})
        sty = sty.set_table_styles([
            {"selector": "th", "props": [("background-color", "#111827"), ("color", "white"), ("font-weight", "700")]},
            {"selector": "td", "props": [("border-bottom", "1px solid #e5e7eb")]},
        ])

        st.caption("Tip: click ticker to open Chart Analyzer (deep-link).")
        st.dataframe(sty, use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='small-muted'>"
            "Notes: ∆EMA = (Price/EMA)-1 | Fresh↑/Fresh↓ = crossover within last 10 trading days | "
            "50/200 = EMA50 vs EMA200 | MACD = MACD(12,26,9) line vs signal | BO = close > prior 20D high."
            "</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 2: Analyzer
# =============================================================================
with tab_analyzer:
    st.subheader("Chart Analyzer")

    # ticker selector
    col1, col2 = st.columns([1, 2], vertical_alignment="center")
    with col1:
        sel = st.selectbox(
            "Select ETF",
            options=TICKERS,
            index=TICKERS.index(default_ticker) if default_ticker in TICKERS else 0,
            key="ticker_select",
        )
    with col2:
        st.markdown(f"<div class='small-muted'>Full name: <b>{TICKER_TO_NAME.get(sel, sel)}</b></div>", unsafe_allow_html=True)

    # keep URL in sync
    st.query_params["ticker"] = sel
    st.query_params["tab"] = "analyzer"

    fig = make_price_chart(raw_prices, sel)
    if fig is None:
        st.error("No chart data available for this ticker.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    # Holdings
    if show_holdings:
        st.divider()
        st.subheader(f"Top 10 Holdings: {sel}")

        with st.spinner("Fetching holdings..."):
            holdings = fetch_holdings_top10(sel)

        if not holdings:
            st.info("Holdings data unavailable via yfinance funds API for this ticker.")
        else:
            # Normalize holding tickers
            hdf = pd.DataFrame(holdings)
            hdf["symbol"] = hdf["symbol"].apply(lambda x: normalize_holding_symbol(x, sel))
            holding_tickers = [x for x in hdf["symbol"].dropna().astype(str).tolist() if x.strip()]

            # Compute holding technicals (selected ETF only)
            with st.spinner("Computing holdings technicals (selected ETF only)..."):
                hold_metrics = fetch_holdings_price_and_metrics(sorted(list(set(holding_tickers))), LOOKBACK_PERIOD)

            def get_m(t, key):
                return hold_metrics.get(t, {}).get(key, np.nan)

            out = hdf.copy()
            out["weight"] = out["weight"].map(lambda x: "-" if pd.isna(x) else f"{x*100:.2f}%")

            for k in ["1M", "3M", "6M", "12M"]:
                out[k] = out["symbol"].map(lambda t: get_m(t, k))
            for k in ["sig_10_20", "sig_20_50", "sig_50_200", "macd_sig", "breakout"]:
                out[k] = out["symbol"].map(lambda t: hold_metrics.get(t, {}).get(k, "-") if t in hold_metrics else "-")

            # style
            pct_cols = ["1M", "3M", "6M", "12M"]
            sig_cols = ["sig_10_20", "sig_20_50", "sig_50_200", "macd_sig", "breakout"]

            sty = out.style
            for c in pct_cols:
                sty = sty.format({c: format_pct})
                sty = sty.applymap(lambda v: style_pct(v) if pd.notna(v) else "", subset=[c])
            for c in sig_cols:
                sty = sty.applymap(style_sig_cell, subset=[c])

            sty = sty.set_table_styles([
                {"selector": "th", "props": [("background-color", "#111827"), ("color", "white"), ("font-weight", "700")]},
                {"selector": "td", "props": [("border-bottom", "1px solid #e5e7eb")]},
            ])

            st.dataframe(
                sty,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Holdings display is disabled in the sidebar.")