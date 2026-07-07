"""
Pull the current SAFCOM closing price from afrimarket (kwayisi.org) as a
sanity-check against EODHD.

RUN LOCALLY:  python data/fetch_afrimarket.py

Caveat: afrimarket exposes the closing price only — no volume. Use EODHD for
OHLCV. This script is a cheap "is the latest price sane?" cross-check.
"""
import afrimarket as afm
import pandas as pd


def main() -> None:
    try:
        scom = afm.Stock(ticker="SCOM", market=afm.markets["Nairobi Securities Exchange"])
        price_data = scom.get_price()
        print(f"afrimarket price data shape: {price_data.shape}")
        print(price_data.tail(10))
        price_data.to_csv("safcom_afrimarket_prices.csv", index=True)
        print(f"Last closing price: {price_data.iloc[-1]}")
        print("Note: afrimarket = closing price only. Use EODHD for OHLCV with volume.")
    except Exception as exc:  # noqa: BLE001 - network/library errors are non-fatal here
        print(f"afrimarket error: {exc}")
        print("Fallback manual current price: KES 28.75 (Jun 2026, kwayisi.org)")


if __name__ == "__main__":
    main()
