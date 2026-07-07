"""
Fetch SAFCOM (SCOM.KE) end-of-day history from EODHD to extend the series past
the XLSX cut-off (2025-09-15) up to the Vodacom takeover close.

RUN LOCALLY:  python data/fetch_eodhd.py
The Claude Code sandbox cannot reach EODHD (outbound blocked); your machine can.
Output CSV is picked up automatically by merge_data.py on the next run.

Security note: the API key is read from the EODHD_API_KEY environment variable —
it is never hard-coded here. Set it locally before running (see the README).
"""
import os
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.getenv("EODHD_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Set the EODHD_API_KEY environment variable before running "
        "(e.g. setx EODHD_API_KEY \"your-key\" on Windows). See the README."
    )
TICKER = "SCOM.KE"
FROM_DATE = "2025-09-15"   # day the XLSX ends
TO_DATE = "2026-06-30"     # Vodacom takeover close
OUTPUT_PATH = Path(__file__).resolve().parent / "safcom_eodhd_2025_2026.csv"


def main() -> None:
    url = f"https://eodhd.com/api/eod/{TICKER}"
    params = {"from": FROM_DATE, "to": TO_DATE, "api_token": API_KEY, "fmt": "json"}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    if df.empty:
        print("EODHD returned no rows for this range/ticker — nothing written.")
        return

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Fetched {len(df)} rows: {df['date'].min().date()} -> {df['date'].max().date()}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
