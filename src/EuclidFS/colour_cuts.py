import polars as pl
from functools import reduce

DESI_IS_NORTH = pl.col("dec_gal") > 32.375
DESI_IS_SOUTH = DESI_IS_NORTH.not_()
DESI_CUTS_N = [((pl.col("sdss_z_mag") - pl.col("wise_w1_mag")) > (0.8 * (pl.col("sdss_r_mag") - pl.col("sdss_z_mag")) - 0.6)), # (1b)
		((pl.col("sdss_g_mag") - pl.col("wise_w1_mag") > 2.97) | ((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > 1.8)), # (1c)
		((((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > (1.83 * (pl.col("wise_w1_mag") - 17.13))) & ((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > (pl.col("wise_w1_mag") - 16.31))) | ((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > 3.4)) # (1d) 
		]

DESI_CUTS_S = [(pl.col("sdss_z_mag") - pl.col("wise_w1_mag")) > 0.8 * (pl.col("sdss_r_mag") - pl.col("sdss_z_mag")) - 0.6, # (2b)
                (pl.col("sdss_g_mag") - pl.col("wise_w1_mag") > 2.9) | (pl.col("sdss_r_mag") - pl.col("wise_w1_mag") > 1.8), # (2c)
		((((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > 1.8 * (pl.col("wise_w1_mag") - 17.14)) & ((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > (pl.col("wise_w1_mag") - 16.33))) | ((pl.col("sdss_r_mag") - pl.col("wise_w1_mag")) > 3.3)) # (2d) 

		]

# DEPRECATED : concat was making polars panic
# def apply_lrg_cuts(ldf: pl.LazyFrame) -> pl.LazyFrame:
#     ldf_N = ldf.filter(DESI_IS_NORTH).filter(DESI_CUTS_N)
#     ldf_S = ldf.filter(DESI_IS_SOUTH).filter(DESI_CUTS_S)

#     res = pl.concat([ldf_N, ldf_S])
#     return res


def all_pl(exprs: list[pl.Expr]) -> pl.Expr:
    """AND-combine a list of Polars expressions into a single Expr."""
    return reduce(lambda a, b: a & b, exprs)

def apply_lrg_cuts(ldf: pl.LazyFrame) -> pl.LazyFrame:
    north_mask = DESI_IS_NORTH & all_pl(DESI_CUTS_N)
    south_mask = DESI_IS_SOUTH & all_pl(DESI_CUTS_S)
    return ldf.filter(north_mask | south_mask)

