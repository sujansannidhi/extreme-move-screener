"""Build the screening universe: liquid S&P 500 names in volatile sectors + curated
high-beta non-S&P names + sector/market ETFs used for industry-adjusted returns."""
import pandas as pd

sp = pd.read_csv("data/sp500_constituents.csv")
sp["Symbol"] = sp["Symbol"].str.replace(".", "-", regex=False)

SECTOR_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
    "Energy": "XLE", "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

full_sectors = {"Information Technology", "Communication Services", "Consumer Discretionary",
                "Energy", "Health Care"}
partial = {
    "Financials": ["COIN", "SCHW", "IBKR", "KKR", "BX", "APO", "ARES", "GS", "MS", "SYF", "ALLY",
                   "CPAY", "FI", "GPN", "PYPL", "MSCI", "ERIE", "BRO"],
    "Industrials": ["GEV", "PWR", "AXON", "UBER", "BA", "TDG", "HWM", "ETN", "VRT", "BLDR",
                    "DAL", "UAL", "URI", "GNRC", "DAY", "PAYC", "JBHT", "CAT", "DE", "LUV"],
    "Materials": ["ALB", "FCX", "NEM", "NUE", "STLD", "MOS", "CF", "CE", "DOW", "LYB"],
    "Utilities": ["VST", "CEG", "NRG", "AES"],
    "Consumer Staples": ["CELH", "WBA", "EL", "DG", "DLTR"],
    "Real Estate": ["EQIX", "DLR"],
}

rows = []
for _, r in sp.iterrows():
    sec = r["GICS Sector"]
    if sec in full_sectors or r["Symbol"] in partial.get(sec, []):
        rows.append(dict(symbol=r["Symbol"], sector=sec, subindustry=r["GICS Sub-Industry"],
                         group="sp500", sector_etf=SECTOR_ETF[sec]))
sp_syms = {r["symbol"] for r in rows}

# Curated liquid high-beta names (sector, sub-industry/peer group)
HB = {
    # AI infra / semis / hardware
    "Information Technology|Semiconductors": ["ARM", "SMCI", "ALAB", "CRDO", "AAOI", "LITE", "COHR",
        "SITM", "NVTS", "INDI", "AEHR", "ACMR", "CAMT", "ASML", "TSM", "AMKR", "RMBS", "POWI", "ONTO",
        "FORM", "ICHR", "SOXL"],
    "Information Technology|Application Software": ["PLTR", "AI", "SOUN", "BBAI", "SNOW", "MDB", "NET",
        "DDOG", "CRWD", "ZS", "OKTA", "GTLB", "S", "APP", "U", "DUOL", "RBLX", "ASAN", "PATH", "TWLO",
        "SHOP", "TOST", "BILL", "HUBS", "CFLT", "ESTC", "DOCN", "IOT", "KVYO", "RDDT", "DJT", "TTD"],
    "Information Technology|Quantum Computing": ["IONQ", "RGTI", "QBTS", "QUBT", "ARQQ"],
    "Information Technology|AI Cloud / Crypto Miners": ["CRWV", "NBIS", "APLD", "IREN", "CIFR", "WULF",
        "HUT", "CORZ", "MARA", "RIOT", "CLSK", "BTDR", "HIVE", "BITF", "BTBT", "GLXY"],
    "Industrials|Space & Drones": ["ASTS", "RKLB", "LUNR", "RDW", "PL", "BKSY", "ACHR", "JOBY", "KTOS",
        "AVAV", "SPCE", "MNTS"],
    "Utilities|Nuclear & Power": ["OKLO", "SMR", "NNE", "LEU", "CCJ", "UEC", "TLN", "BWXT", "FLNC", "EOSE",
        "BE", "PLUG", "FCEL"],
    "Health Care|Biotechnology": ["NVAX", "BNTX", "VKTX", "SRPT", "ALNY", "INSM", "RARE", "IOVA", "CRSP",
        "NTLA", "BEAM", "EDIT", "VERV", "ARWR", "AXSM", "ACAD", "TGTX", "EXEL", "HALO", "SMMT", "KRYS",
        "CYTK", "MDGL", "RVMD", "RXRX", "TEM", "HIMS", "OSCR", "CLOV", "NUVL", "IMVT", "APLS", "ROIV",
        "BCRX", "DNLI", "PCVX", "RNA", "JANX", "KURA", "SAVA", "GERN", "ABCL", "NVCR", "TMDX"],
    "Financials|Crypto & Fintech": ["MSTR", "HOOD", "SOFI", "CRCL", "UPST", "AFRM", "LMND", "ROOT", "RKT",
        "UWMC", "NU", "FUTU", "TIGR", "LC", "DAVE", "BTCS", "SBET", "BMNR"],
    "Consumer Discretionary|EV & Autos": ["RIVN", "LCID", "NIO", "XPEV", "LI", "QS", "LAZR", "PSNY",
        "CHPT", "BLNK", "RUN", "ENPH", "SEDG", "ARRY", "NXT"],
    "Consumer Discretionary|Retail & Internet": ["GME", "AMC", "OPEN", "CVNA", "KSS", "SE", "MELI",
        "GRAB", "BABA", "PDD", "JD", "BIDU", "CPNG", "W", "ETSY", "CHWY", "PTON", "FIVE", "ELF", "DKNG",
        "ROKU", "SNAP", "PINS", "SPOT", "NFLX"],
    "Energy|Energy Services & LNG": ["NFE", "TPL", "AR", "RRC", "EQT", "CHRD", "VAL", "RIG", "NE",
        "BTU", "AMR", "HCC"],
    "Materials|Metals & Mining": ["MP", "USAR", "CENX", "AA", "CLF", "X", "HL", "CDE", "AG", "SILV",
        "PAAS", "AGI", "KGC", "GOLD", "FNV"],
    "Industrials|Shipping & Transport": ["ZIM", "FRO", "STNG", "GLNG", "SBLK", "AAL", "JBLU", "SAVE"],
}
# sub-industry -> sector ETF for the curated groups
for key, syms in HB.items():
    sec, sub = key.split("|")
    for s in syms:
        if s in sp_syms:
            continue
        rows.append(dict(symbol=s, sector=sec, subindustry=sub, group="highbeta",
                         sector_etf=SECTOR_ETF[sec]))

for etf in sorted(set(SECTOR_ETF.values())) + ["SPY", "QQQ", "IWM", "SMH", "XBI", "ARKK"]:
    rows.append(dict(symbol=etf, sector="ETF", subindustry="ETF", group="etf", sector_etf="SPY"))

u = pd.DataFrame(rows).drop_duplicates("symbol")
u.to_csv("data/universe.csv", index=False)
print(u.group.value_counts(), len(u))
