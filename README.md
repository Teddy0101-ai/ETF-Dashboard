# ETF Strategy Dashboard (Streamlit)

A Streamlit web app that tracks a curated ETF universe and provides:
- Momentum (1M/3M/6M/12M)
- EMA distances (ΔEMA10/20/50/200)
- Signal status (10/20, 20/50, 50/200, MACD, Breakout)
- MACD event updates (last 5 trading days)
- Chart analyzer (Candlestick + EMAs + MACD panel)
- Top-10 holdings (selected ETF only) + holdings technicals

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub
2. Go to Streamlit Community Cloud
4. Main file path: app.py
5. Deploy

## Deep links
- Open Analyzer: ?ticker=XLK&tab=analyzer
- Open Tracker: ?tab=tracker
