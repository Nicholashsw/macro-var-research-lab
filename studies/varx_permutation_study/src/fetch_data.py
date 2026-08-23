"""
fetch_data.py

Download every public source series the study needs into ../data. No API keys
are required. FRED is pulled from its keyless CSV endpoint with curl, because the
urllib user agent is blocked by Akamai. Run once before the pipeline:

    python fetch_data.py

The ALFRED real-time vintages are fetched on demand by varx_realtime.py and
cached under data/vintages, so they are not downloaded here.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
IDENT = os.path.join(DATA, "identified")

FRED = ["GDPC1", "PCEPILFE", "FEDFUNDS", "AHETPI", "PAYEMS", "GS10", "TB3MS",
        "PERMIT", "WTISPLC", "CPIAUCSL", "GPDIC1", "TNWBSHNO", "TWEXBMTH", "DTWEXBGS", "GCEC1"]

HLW = ("https://www.newyorkfed.org/medialibrary/media/research/economists/"
       "williams/data/Holston_Laubach_Williams_current_estimates.xlsx")
KANZIG_ZIP = "https://codeload.github.com/dkaenzig/oilsupplynews/zip/refs/heads/master"
FRBSF_MPS = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"


def curl(url, out):
    subprocess.run(["curl", "-sL", "-o", out, url], check=True)
    return os.path.getsize(out)


def main():
    os.makedirs(IDENT, exist_ok=True)
    for s in FRED:
        n = curl(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", f"{DATA}/{s}.csv")
        print(f"FRED {s}: {n} bytes")
    print("HLW r-star:", curl(HLW, f"{DATA}/hlw.xlsx"), "bytes")

    # externally identified shocks
    zp = f"{IDENT}/oilsupplynews.zip"
    curl(KANZIG_ZIP, zp)
    subprocess.run(["unzip", "-oq", zp, "-d", IDENT], check=True)
    print("Kaenzig oil supply news: extracted to identified/oilsupplynews-master")
    print("Bauer-Swanson MPS:", curl(FRBSF_MPS, f"{IDENT}/mps.xlsx"), "bytes")


if __name__ == "__main__":
    main()
