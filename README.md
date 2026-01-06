# RSI Discord Bot (Yahoo Finance / yfinance)

This project is a self-hosted Discord bot that calculates RSI (Relative Strength Index) and posts **oversold** / **overbought** alerts for stocks available on **Yahoo Finance**.

It was created because there are no reliable free alert systems for many **Nordic exchanges**, while Yahoo Finance often provides broad coverage (including Nordic tickers) through a single, commonly accessible data source.

## Why this exists

Most free stock alert tools either:
- do not cover Nordic tickers well,
- require paid subscriptions for alerts
- limit customization and automation.

This bot is designed to run 24/7 on your own hardware (for example a Raspberry Pi or a small VPS) and provide **custom, automated RSI alerts** without paying for a third-party alert platform.

## Data source: Yahoo Finance via `yfinance` (unofficial)

This bot uses the Python library **`yfinance`** to fetch historical price data from **Yahoo Finance**.

Important notes:
- Yahoo Finance does **not** provide an official free public API for this use case.
- `yfinance` acts as a practical API-like wrapper around Yahoo Finance’s publicly available endpoints.
- Because this is unofficial, data availability and formatting can change without notice. In rare cases, Yahoo may throttle or block requests.

In short: this project is built on a widely used community approach to access Yahoo Finance data, but reliability is ultimately dependent on Yahoo Finance and the `yfinance` project.

## Coverage and ticker formats

The bot can alert on **any stock that Yahoo Finance supports**, across many exchanges, as long as you use the correct Yahoo ticker format.

Examples:
- Oslo Børs: `EQNR.OL`
- Stockholm: `ERIC-B.ST`
- US (NASDAQ/NYSE): `AAPL`, `MSFT`

If a ticker is not available on Yahoo Finance, the bot cannot fetch data for it.

## Rate limiting and batch processing (to reduce blocking risk)

To reduce the chance of being rate-limited or flagged by Yahoo Finance, the bot intentionally uses:
- **Batching** (processing tickers in groups instead of all at once)
- **Delays between batches**
- (Optional) additional cooldown logic to avoid repeated alerts

These limits are a practical compromise:
- They keep the bot stable for larger watchlists (hundreds of tickers),
- while lowering the likelihood of throttling or temporary blocking.

Even with these safeguards, you should assume there is always some risk of throttling when using unofficial Yahoo Finance access. If you increase the frequency or the number of tickers significantly, adjust batch size and delays accordingly.

## Disclaimer

This project is not affiliated with Yahoo Finance, and it is not investment advice. Market data may be delayed, incomplete, or temporarily unavailable. Use responsibly and in accordance with the terms of your data source and Discord.
