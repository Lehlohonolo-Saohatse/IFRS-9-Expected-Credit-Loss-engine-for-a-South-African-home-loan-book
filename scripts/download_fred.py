"""Run LOCALLY (needs internet):  python scripts/download_fred.py
Downloads SA series from FRED (no API key needed for the CSV endpoint) into data/raw/ in the
`date,value` format that src/macro.py expects. Repo and prime must come from SARB (see README).
Only the unemployment ID was verified when this was written; check the others on fred.stlouisfed.org
(search the title) and fix the ID if FRED has renamed or discontinued it.
"""
import io, sys, urllib.request
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SERIES = {
    "unemployment.csv": "LRUNTTTTZAQ156S",   # OECD via FRED: unemployment rate, 15+, SA, quarterly (verified)
    "cpi_yoy.csv": "CPALTT01ZAM659N",        # CPI all items, growth same period previous year, monthly (verify)
    "gdp_yoy.csv": "NAEXKP01ZAQ657S",        # real GDP growth y/y, quarterly (verify)
    "hpi.csv": "QZAR628BIS",                 # BIS residential property prices, South Africa (verify; real vs nominal)
}
for fname, sid in SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
        df = pd.read_csv(io.StringIO(raw)); df.columns = ["date", "value"]
        df = df.replace(".", pd.NA).dropna(); df["value"] = df["value"].astype(float)
        df.to_csv(RAW / fname, index=False)
        print(f"ok   {fname:18s} {sid:18s} {len(df)} rows {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")
    except Exception as e:
        print(f"FAIL {fname:18s} {sid:18s} {e}", file=sys.stderr)
print("Now add data/raw/repo.csv (SARB) and re-run: python -m src.pipeline")
