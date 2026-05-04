import polars as pl
from functools import reduce

def apply_colour_cuts(lf: pl.LazyFrame, cuts: list|pl.Expr) -> pl.LazyFrame:
    if isinstance(cuts, pl.Expr):
        cuts = [cuts]
    for expr in cuts:
        lf = lf.filter(expr)
    return lf

def all_pl(exprs:list):
    """AND-combine a list of Polars expressions."""
    return reduce(lambda a, b: a & b, exprs)

DESI_IS_NORTH = pl.col("dec_gal") > 32.375
DESI_IS_SOUTH = DESI_IS_NORTH.not_()
DESI_CUTS_N = [(pl.col("lsst_z") - pl.col("wise_w1_mag")) > 0.8 * (pl.col("lsst_r") - pl.col("lsst_z")) - 0.6, # (1b)
		(pl.col("lsst_g") - pl.col("wise_w1_mag") > 2.9) | (pl.col("lsst_r") - pl.col("wise_w1_mag") > 1.8), # (1c)
		((((pl.col("lsst_r") - pl.col("wise_w1_mag")) > 1.8 * (pl.col("wise_w1_mag") - 17.14)) & ((pl.col("lsst_r") - pl.col("wise_w1_mag")) > (pl.col("wise_w1_mag") - 16.33))) | ((pl.col("lsst_r") - pl.col("wise_w1_mag")) > 3.3)) # (1d) 
		]

DESI_CUTS_S = [(pl.col("lsst_z") - pl.col("wise_w1_mag")) > 0.8 * (pl.col("lsst_r") - pl.col("lsst_z")) - 0.6, # (2b)
                (pl.col("lsst_g") - pl.col("wise_w1_mag") > 2.97) | (pl.col("lsst_r") - pl.col("wise_w1_mag") > 1.8), # (2c)
		((((pl.col("lsst_r") - pl.col("wise_w1_mag")) > 1.83 * (pl.col("wise_w1_mag") - 17.13)) & ((pl.col("lsst_r") - pl.col("wise_w1_mag")) > (pl.col("wise_w1_mag") - 16.31))) | ((pl.col("lsst_r") - pl.col("wise_w1_mag")) > 3.4)) # (2d) 

		]

DESI_LRG_CUT = (
    (DESI_IS_NORTH & all_pl(DESI_CUTS_N)) |
    (DESI_IS_SOUTH & all_pl(DESI_CUTS_S))
)
