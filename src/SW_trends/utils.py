import xarray as xr
import numpy as np


def rank_func(ensemble_vals, era_val):
    from scipy.stats import rankdata
    return rankdata(np.hstack((ensemble_vals, era_val)))[-1]


def get_rank_hist(ranks, mask, rank_max):
    vals = ranks.where(mask == 1).values.flatten()
    weights = (np.cos(np.deg2rad(ranks.lat))).expand_dims(dim={'lon': ranks.lon},
                                                          axis=-1).values.flatten()
    weights = weights[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    bin_edges = np.arange(0.5, rank_max + 1.5, 1)
    bin_middle = (bin_edges[1:] + bin_edges[:-1]) / 2
    hist_vals, _ = np.histogram(vals, bins=bin_edges, weights=weights, density=True)

    return hist_vals, bin_middle


def extract_years(filename):
    import re
    match = re.search(r'(\d{6})-(\d{6})', filename)  # Find YYYYMM-YYYYMM pattern
    if match:
        start_year = int(match.group(1)[:4])  # Extract first 4 digits as start year
        end_year = int(match.group(2)[:4])    # Extract first 4 digits as end year
        return start_year, end_year
    return None


def get_analysis_mask(da_lsmask, landcut=0.5):
    """Return the relevant analysis mask (land, no Greenland, 60S-80N) given a land fraction"""
    from helpful_utilities.geom import get_regrid_country
    country_folder = '/home/data/geom/ne_110m_admin_0_countries/'  # outlines of countries
    is_land = (da_lsmask > landcut).squeeze()
    if 'latitude' in da_lsmask.dims:
        is_land = adjust_era5_da(is_land)
    da_greenland = get_regrid_country('Greenland', country_folder, is_land.lat, is_land.lon, dilate=True)
    analysis_mask = is_land & ~da_greenland & (is_land.lat > -60) & (is_land.lat < 80)
    return analysis_mask


def adjust_era5_da(da):
    """Rename ERA5 lat/lon, and sort by latitude"""
    da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = da.sortby('lat')
    return da


def regrid_to_shared_grid(da, shared_lats, shared_lons):
    # Regrid all to shared 1x1 grid, looping lons
    da_pad = xr.concat([
        da.sel(lon=da.lon[-1]),  # shift last point to before first
        da,
        da.sel(lon=da.lon[0])   # shift first point to after last
    ], dim='lon')

    # Update longitude coordinates accordingly
    da_pad['lon'] = xr.concat([
        da.lon[-1] - 360,
        da.lon,
        da.lon[0] + 360
    ], dim='lon')

    da_interp = da_pad.interp({'lat': shared_lats, 'lon': shared_lons})

    return da_interp
