from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class SP500Universe:
    csv_path: str

    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        # Normalize columns from datasets/s-and-p-500-companies
        # Columns: Symbol, Name, Sector
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
        return df

    def tickers(self, *, max_n: Optional[int] = None) -> List[str]:
        syms = self.df()["Symbol"].tolist()
        if max_n is not None:
            syms = syms[:max_n]
        return syms

    def by_sector(self, *, max_n: Optional[int] = None) -> pd.DataFrame:
        df = self.df()
        if max_n is not None:
            df = df.head(max_n)
        return df.groupby("Sector")["Symbol"].count().sort_values(ascending=False).to_frame("count")
