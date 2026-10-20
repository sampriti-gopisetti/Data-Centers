"""USDA 2020 RUCA tract file -> clean tract-level urban/rural features."""
import numpy as np
import pandas as pd


def _read_any(path: str) -> pd.DataFrame:
    """Read csv/xlsx and find the real header row (USDA files often have title rows on top)."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    else:
        sheets = {"csv": pd.read_csv(path, header=None, dtype=str, encoding="latin-1")}
    for raw in sheets.values():
        for i in range(min(25, len(raw))):
            # real header has both a primary and a secondary RUCA column; a title row has at most one
            if raw.iloc[i].astype(str).str.contains("ruca", case=False, na=False).sum() >= 2:
                df = raw.iloc[i + 1:].copy()
                df.columns = [str(c).strip() for c in raw.iloc[i]]
                return df.reset_index(drop=True)
    raise ValueError("No header row containing 'RUCA' found; open the file and check its layout.")


def _tract_col(df: pd.DataFrame) -> str:
    """The tract id column = the one whose values look like 11-digit FIPS codes."""
    for c in df.columns:
        s = df[c].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        if (s.str.len().isin([10, 11]) & s.str.isdigit()).mean() > 0.95:
            return c
    raise ValueError(f"Couldn't find a tract FIPS column. Columns: {list(df.columns)}")


def _ruca_col(df: pd.DataFrame, kind: str) -> str:
    hits = [c for c in df.columns if "ruca" in c.lower() and kind in c.lower()]
    if not hits:
        raise ValueError(f"No {kind} RUCA column. Columns: {list(df.columns)}")
    return hits[0]


def ruca_class(p: float) -> str | float:
    if pd.isna(p):
        return np.nan
    p = int(p)
    if p == 1:
        return "urban_core"
    if p in (2, 3):
        return "suburban"      # metro-commuting tracts: design choice, confirm with team
    if 4 <= p <= 9:
        return "small_city_town"
    return "rural"             # 10


def build_ruca_table(path: str) -> pd.DataFrame:
    df = _read_any(path)
    tcol, pcol, scol = _tract_col(df), _ruca_col(df, "primary"), _ruca_col(df, "secondary")
    out = pd.DataFrame({
        "tract_id": df[tcol].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(11),
        "ruca_primary": pd.to_numeric(df[pcol], errors="coerce"),
        "ruca_secondary": pd.to_numeric(df[scol], errors="coerce"),
    })
    out["ruca_uncoded"] = out["ruca_primary"].eq(99) | out["ruca_primary"].isna()  # water-only/zero-pop
    out.loc[out["ruca_uncoded"], ["ruca_primary", "ruca_secondary"]] = np.nan
    out["ruca_class"] = out["ruca_primary"].map(ruca_class)
    out["is_rural"] = out["ruca_primary"].eq(10).where(~out["ruca_uncoded"])
    return out.drop_duplicates("tract_id").reset_index(drop=True)