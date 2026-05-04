from pathlib import Path
from datetime import datetime
import json

BASE_PLOTS_DIR   = Path("plots")
BASE_RESULTS_DIR = Path("results")

class RunDir:
    def __init__(self, name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name     = f"{ts}_{name}"
        self.plots    = BASE_PLOTS_DIR   / self.name
        self.results  = BASE_RESULTS_DIR / self.name
        self.plots.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def save_plot(self, fig, filename: str):
        path = self.plots / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot   → {path}")

    def save_result(self, df_or_dict, filename: str):
        path = self.results / filename
        if hasattr(df_or_dict, "write_csv"):   # polars df
            df_or_dict.write_csv(path)
        elif isinstance(df_or_dict, dict):      # params, metadata
            with open(path, "w") as f:
                json.dump(df_or_dict, f, indent=2)
        else:
            raise TypeError(f"Cannot save type {type(df_or_dict)}")
        print(f"Result → {path}")
