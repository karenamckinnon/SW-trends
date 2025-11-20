# Load my packages
import SW_trends.utils as my_utils
from helpful_utilities.ncutils import lon_to_360
import helpful_utilities.xutils as my_xutils
import helpful_utilities.stats as my_stat_utils
from helpful_utilities.plotting import plot_global_discrete

# Load standard packages for analysis
import pandas as pd
import xarray as xr
from scipy.stats import linregress, gaussian_kde
import numpy as np

# Load packages for plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

# Load packages for file management and other misc
from glob import glob
import cftime
import os
import string


# Directory information (update to use this on other systems)
figdir = '/home/kmckinnon/SW-trends/figs'
procdir = '/home/data/projects/SW-trends/proc'

geba_dir = '/home/data/GEBA'

data_dirs = {
    'JRA3Q': '/home/data/JRA3Q/monthly',
    'CESM2': '/home/data/CESM2/LE/monthly',
    'GOGA': '/home/data/CESM2/GOGA/monthly',
    'CLARA': '/home/data/CMSAF/CLARA',
    'ERA5': '/home/data/ERA5/month',
    'GEWEX': '/home/data/GEWEX/Shortwave_monthly_utc_1',
    'CERES': '/home/data/CERES',
    'MERRA2': '/home/data/MERRA2/monthly/SW_down',
}

# Variable names by product
name_dict = {
    'MERRA2': 'SWGDN',
    'GEWEX': 'all_sw_dn_sfc',
    'CLARA': 'SIS',
    'ERA5': 'surface_solar_radiation_downwards',
    'CERES': 'sfc_sw_down_all_mon',
    'CESM2': 'FSDS',
    'GOGA': 'FSDS',
    'JRA3Q': 'dswrf1have-sfc-fc-gauss-mn'
}

reanalysis_names = 'ERA5', 'MERRA2', 'JRA3Q', 'CLARA', 'GEWEX'  # compare to reanalysis and other RS products


years_to_use = 1980, 2024
nyrs = years_to_use[1] - years_to_use[0] + 1

var_to_analyze = 'Downward surface shortwave'
short_name = {'Downward surface shortwave': 'SW_down'}
this_var = short_name[var_to_analyze]

letters = list(string.ascii_lowercase)

# Trend colorbar
vmin, vmax = -0.5, 0.5
step = 0.1
levels = np.arange(vmin, vmax + step, step)
norm = BoundaryNorm(levels, ncolors=256)
cmap = 'RdBu_r'

trend_cbar = {'levels': levels,
              'norm': norm,
              'cmap': cmap}

# Correlation colorbar
vmin, vmax = -1, 1
step = 0.2
levels = np.arange(vmin, vmax + step, step)
norm = BoundaryNorm(levels, ncolors=256)
cmap = 'RdBu_r'

corr_cbar = {'levels': levels,
             'norm': norm,
             'cmap': cmap}

# Positive correlation colorbar
vmin, vmax = 0, 1
step = 0.1
levels = np.arange(vmin, vmax + step, step)
norm = BoundaryNorm(levels, ncolors=256)
cmap = 'Reds'

pos_corr_cbar = {'levels': levels,
                 'norm': norm,
                 'cmap': cmap}

trend_str = r'SW$_\downarrow$ trend (W/m$^2$/year)'


# # Load land masks
# - also remove Greenland, and subset to -60, 80

# 1x1
f_lsmask_1x1 = '/home/data/ERA5/fx/era5_lsmask_1x1.nc'
da_lsmask_1x1 = xr.open_dataarray(f_lsmask_1x1)
analysis_mask = my_utils.get_analysis_mask(da_lsmask_1x1)

# For regridding needs
shared_lats = analysis_mask.lat.values
shared_lons = analysis_mask.lon.values


# # Load all gridded data (satellite products, reanalyses, models)
all_sw = []
all_sw_native = []
name_list = []
name_list_native = []

names = name_dict.keys()
for name in names:
    this_dir = data_dirs[name]
    v = name_dict[name]
    print('%s, %s' % (name, v))
    if (name == 'CESM2') | (name == 'GOGA') | (name == 'ERA5') | (name == 'JRA3Q'):
        this_dir = this_dir + '/%s' % v

    if (name == 'CESM2') | (name == 'GOGA'):  # Ensembles
        if (name == 'CESM2'):
            # Load CESM2
            files = sorted(glob('%s/b.e21.B*smbb.f09_g17.*.*.h0.%s.??????-??????.nc' %
                                (this_dir, v)))
            ens_members = np.unique(np.array(['.'.join(f.split('.')[4:6]) for f in files]))
        else:
            # Load GOGA
            files = sorted(glob('%s/f.e21.F*.f09_f09.*goga*.cam.h0.%s*.??????-??????.nc' %
                                (this_dir, v)))
            ens_members = ['%02i' % n for n in range(1, 11)]

        files = [
            f for f in files
            if (years := my_utils.extract_years(f)) and (years[0] <= (years_to_use[-1] + 1)
                                                         and years[1] >= (years_to_use[0] + 1))
        ]

        da_sw = []

        # Load each ensemble member
        for ens_member in ens_members:
            use_files = [f for f in files if ens_member in f]
            da = xr.open_mfdataset(use_files)[v]

            # Round lat/lon since they can slightly differ between CESM products
            da['lat'] = np.round(da['lat'], 3).data.astype('float32')
            da['lon'] = np.round(da['lon'], 3).data.astype('float32')
            da_sw.append(da)

        da_sw = xr.concat(da_sw, dim='member')
        da_sw['member'] = ens_members
        da_sw = da_sw.rename(v)

        # Move time by one month because of CESM2 timestamp issues for monthly data
        # (the monthly averages are saved with the next month's time)
        new_time = [cftime.DatetimeNoLeap(t.year, t.month - 1, t.day) if t.month > 1
                    else cftime.DatetimeNoLeap(t.year - 1, 12, t.day)
                    for t in da_sw['time'].values]
        da_sw['time'] = new_time

    else:
        varname = name_dict[name]
        files = sorted(glob('%s/*.nc' % this_dir))
        if name == 'GEWEX':
            files = sorted(glob('%s/*.nc4' % this_dir))
        if name == 'ERA5':
            da_sw = xr.open_dataarray(files[0])  # ERA5 has single file
        else:
            da_sw = xr.open_mfdataset(files)[varname]

        if name == 'ERA5':
            da_sw = da_sw.rename({'valid_time': 'time'})
            adjusted_time = da_sw.time.dt.floor('D')
            da_sw['time'] = adjusted_time
            # Heating is in J/m2
            sec_per_day = 3600 * 24
            da_sw /= sec_per_day

        if name == 'LANDFLUX':  # mask to max total obs - some gridboxes on the edge have few measurements
            max_obs = (~np.isnan(da_sw)).sum('time').max()
            has_max = (~np.isnan(da_sw)).sum('time') == max_obs
            da_sw = da_sw.where(has_max)

    # Book-keeping: all to lat/lon, lon is 0-360, consistent names, same time period
    if 'latitude' in da_sw.dims:
        da_sw = da_sw.rename({'latitude': 'lat', 'longitude': 'lon'})
    da_sw = da_sw.sortby('lat').rename('%s_%s' % (name, short_name[var_to_analyze]))
    da_sw = lon_to_360(da_sw)

    # For reanalysis, for comparion to station data, keep native resolution as well
    if (name == 'ERA5') | (name == 'JRA3Q') | (name == 'MERRA2') | (name == 'CLARA') | (name == 'GEWEX'):
        all_sw_native.append(da_sw.copy())
        name_list_native.append(name)

    da_interp = my_utils.regrid_to_shared_grid(da_sw, shared_lats, shared_lons)

    name_list.append('%s_%s' % (name, short_name[var_to_analyze]))
    all_sw.append(da_interp.sel(time=slice('%04i' % years_to_use[0],
                                           '%04i' % years_to_use[1])).load())


# # Compute annual trends
trend_maps = []
start_year = []
end_year = []

for ct, da in enumerate(all_sw):

    print(da.name)

    # remove seasonal cycle to avoid some issues with preferential sampling across specific months
    # for datasets without complete coverage
    da_anom = da.groupby('time.month') - da.groupby('time.month').mean()
    ann_mean = my_xutils.compute_annual_mean_of_full_years(da_anom)
    ann_mean = ann_mean.sel(year=slice('%04i' % years_to_use[0], '%04i' % years_to_use[1]))
    trend = my_xutils.compute_linear_trend_per_year(ann_mean)

    trend_maps.append(trend)
    start_year.append(ann_mean.year[0])
    end_year.append(ann_mean.year[-1])


# # Load auxiliary datesets (ISCCP, CERES TOA, MERRA2 AOD, ERA5 clouds, ERA5 TOA )
# Get ISCCP data
isccp_dir = '/home/data/ISCCP/BASIC'
files = sorted(glob('%s/*.nc' % isccp_dir))
ds_isccp = xr.open_mfdataset(files)
da_cldamt_isccp = ds_isccp['cldamt_irtypes'].load()
del ds_isccp

# Get TOA CERES data
f_ceres_toa = '/home/data/CERES/CERES_EBAF-TOA_Ed4.2_Subset_200003-202407.nc'
ds_ceres_toa = xr.open_dataset(f_ceres_toa)
da_ceres_toa = ds_ceres_toa['solar_mon']
del ds_ceres_toa

# Get MERRA2 AOD data
aod_files = sorted(glob('/home/data/MERRA2/monthly/AOD/*.nc'))
da_aod = xr.open_mfdataset(aod_files)['AODANA']
da_aod = ((lon_to_360(da_aod.sortby('lat'))).interp(lat=shared_lats, lon=shared_lons)).load()

# Get ERA5 cloud data
savename = '%s/ds_clouds_era5_coarse.nc' % procdir
if os.path.isfile(savename):
    ds_clouds_era5_coarse = xr.open_dataset(savename)
else:
    era5_cloud_names = ['%s_cloud_cover' % c for c in list(('low', 'medium', 'high', 'total'))]
    ds_clouds_era5 = []
    for var in era5_cloud_names:
        this_dir = '%s/%s' % (data_dirs['ERA5'], var)
        da = xr.open_dataarray('%s/%s.nc' % (this_dir, var))
        da = da.rename({'valid_time': 'time'})
        adjusted_time = da.time.dt.floor('D')
        da['time'] = adjusted_time
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
        da = da.sortby('lat').rename(var)
        da = da.sel(time=slice('%04i' % years_to_use[0],
                               '%04i' % years_to_use[1]))
        ds_clouds_era5.append(da.load())
    ds_clouds_era5 = xr.merge(ds_clouds_era5)
    ds_clouds_era5_coarse = ds_clouds_era5.interp(lat=shared_lats, lon=shared_lons)
    del ds_clouds_era5
    ds_clouds_era5_coarse.to_netcdf(savename)

# Load ERA5 TOA
savename = '%s/da_era5_toa_coarse.nc' % procdir
if os.path.isfile(savename):
    da_era5_toa_coarse = xr.open_dataset(savename)
else:
    toa_var = 'toa_incident_solar_radiation'
    this_dir = '%s/%s' % (data_dirs['ERA5'], toa_var)
    da_era5_toa = xr.open_dataarray('%s/%s.nc' % (this_dir, toa_var))
    da_era5_toa = da_era5_toa.rename({'valid_time': 'time'}).drop('expver')
    adjusted_time = da_era5_toa.time.dt.floor('D')
    da_era5_toa['time'] = adjusted_time
    da_era5_toa = da_era5_toa.rename({'latitude': 'lat', 'longitude': 'lon'})
    da_era5_toa = da_era5_toa.sortby('lat').rename(toa_var)
    da_era5_toa = da_era5_toa.sel(time=slice('%04i' % years_to_use[0], '%04i' % years_to_use[1]))
    da_era5_toa_coarse = da_era5_toa.interp(lat=shared_lats, lon=shared_lons)
    del da_era5_toa
    da_era5_toa_coarse.to_netcdf(savename)


# # Reanalysis validation
# 1. CERES interannual variability
# 2. CERES trends
# 3. GEBA interannual (full period and pre-CERES)
# 4. GEBA trends

# ## CERES metrics
alpha_fdr = 0.01

idx_CERES = np.where(['CERES_%s' % (this_var) == n for n in name_list])[0][0]
ceres_start_year = start_year[idx_CERES]
ceres_end_year = end_year[idx_CERES]
ceres_ann = my_xutils.compute_annual_mean_of_full_years(all_sw[idx_CERES])
for rname in reanalysis_names:
    print(rname)

    savename = '%s/validation_metrics_%s.nc' % (procdir, rname)
    if os.path.isfile(savename):
        continue

    idx_r = np.where(['%s_%s' % (rname, this_var) == n for n in name_list])[0][0]

    # (1) correlation with CERES

    # Calculate annual mean of reanalysis
    sub_re = all_sw[idx_r].sel(time=slice('%04i' % ceres_start_year, '%04i' % ceres_end_year))
    reanalysis_ann = my_xutils.compute_annual_mean_of_full_years(sub_re)
    # Subset both to shared period
    this_start = np.max((ceres_start_year, start_year[idx_r]))
    this_end = np.min((ceres_end_year, end_year[idx_r]))

    reanalysis_ann = reanalysis_ann.sel(year=slice('%04i' % this_start, '%04i' % this_end))

    # Calculate correlation
    rho, pvals = my_xutils.pearsonr_xr(reanalysis_ann.where(analysis_mask),
                                       ceres_ann.where(analysis_mask).sel(year=slice('%04i' % this_start,
                                                                                     '%04i' % this_end)),
                                       dim='year')

    # Mask part 1: cases where there is not a significant correlation
    is_sig_map = my_stat_utils.fdr_da(pvals.where(analysis_mask), alpha_fdr=alpha_fdr)

    # (2) Assess whether the trends are different via significance of difference in datasets
    delta_da = reanalysis_ann - ceres_ann.sel(year=slice('%04i' % this_start, '%04i' % this_end))
    delta_da = delta_da.where(analysis_mask)
    _, delta_pval = my_xutils.xr_linregress_pval(delta_da)

    is_sig_map_trend_diff = my_stat_utils.fdr_da(delta_pval, alpha_fdr=alpha_fdr)

    r_quality_mask = (is_sig_map == 1).astype(int) & (is_sig_map_trend_diff == 0).astype(int)
    r_quality_mask = r_quality_mask.where(analysis_mask)

    # Get trend for subset period to compare to CERES
    r_trend_sub, _ = my_xutils.xr_linregress_pval(reanalysis_ann)

    ds_valid = xr.merge((is_sig_map.rename('sig_corr'),
                         is_sig_map_trend_diff.rename('sig_diff_trends'),
                         r_quality_mask.rename('quality_mask'),
                         rho.rename('correlation'),
                         r_trend_sub.rename('CERES_era_trend')))
    ds_valid.to_netcdf(savename)


# ## GEBA metrics
# First load station data from GEBA (annual mean)
files = glob('%s/*.csv' % geba_dir)
md = pd.read_csv(files[0])
geba_ann = pd.read_csv(files[-1])

# Will need to pull native reanalysis grids
all_sw_native_loaded = []
for r in reanalysis_names:
    idx_r = np.where([r == n for n in name_list_native])[0][0]
    all_sw_native_loaded.append(all_sw_native[idx_r].load())

# Pull two different sets of GEBA data:
# 0: 1980-2024, has at least 20 years of data
# 1: 1980-2000, has at least 10 years of data
savename_trends = '%s/geba_reanalysis_with_RS_trends.npz' % procdir

if not os.path.isfile(savename_trends):
    for geba_ct in range(2):

        # Define time thresholds
        if geba_ct == 0:  # Full period
            cutoff_year1 = 1980
            cutoff_year2 = 2024
            min_years = 20
            savestr = 'full_record_%iyearsmin' % min_years

        else:  # Before CERES but during reanalysis
            cutoff_year1 = 1980
            cutoff_year2 = 2000
            min_years = 10
            savestr = 'pre_ceres_%iyearsmin' % min_years

        savename_interannual = '%s/geba_reanalysis_with_RS_corr_%s.npz' % (procdir, savestr)
        print(savename_interannual)

        # Group by station
        grouped = geba_ann.groupby('tskey')

        valid_tskeys = []

        for tskey, group in grouped:
            # check that it is direct
            this_type = md.loc[md['tskey'] == tskey, 'ebcode']
            if this_type.values[0] != 'GLOBAL':
                continue
            group = group.dropna(subset=['converted_flux_avg'])

            # Subset to specific period
            subset = group.loc[(group['year'] >= cutoff_year1) & (group['year'] < cutoff_year2)]

            # Count obs
            N = (~np.isnan(subset['converted_flux_avg'])).sum()
            has_enough_data = N >= min_years

            if has_enough_data:
                valid_tskeys.append(tskey)

        # Final list of tskeys
        valid_tskeys = sorted(valid_tskeys)
        md_valid = md[md['tskey'].isin(valid_tskeys)]
        N = len(md_valid)
        geba_lats = md_valid['sgylat']
        geba_lons = md_valid['sgxlon']

        if geba_ct == 0:
            trends_save = np.nan * np.ones((len(reanalysis_names) + 2, N))
        rho_geba_reanalysis = np.nan * np.ones((len(reanalysis_names), N))

        for ct, key in enumerate(valid_tskeys):
            if (ct % 10) == 0:
                print('%i/%i' % (ct, len(valid_tskeys)))
            this_ts = geba_ann.loc[geba_ann['tskey'] == key]
            this_lat = md_valid.loc[md_valid['tskey'] == key, 'sgylat'].values[0]
            this_lon = md_valid.loc[md_valid['tskey'] == key, 'sgxlon'].values[0]

            # remove flagged and outlier values
            is_ok = (this_ts['computed_flag_avg'] == 8).values
            years = this_ts['year'][is_ok]
            vals = this_ts['converted_flux_avg'][is_ok]

            # remove outliers beyond 4sigma
            outlier = ((vals >= (4 * np.std(vals) + np.mean(vals))) |
                       (vals <= (-4 * np.std(vals) + np.mean(vals))))
#             if np.sum(outlier) > 0:
#                 fig, ax = plt.subplots(figsize=(5, 2))
#                 ax.plot(years, vals, 'ok')
#                 ax.plot(years[outlier], vals[outlier], 'sr')
#                 plt.show()
            years = years[~outlier]
            vals = vals[~outlier]

            # subset to reanalysis period
            reanalysis_period = (years >= years_to_use[0]) & (years <= years_to_use[1])
            years = np.array(years[reanalysis_period])  # GEBA years subset to reanalysis period
            vals = np.array(vals[reanalysis_period])

            if geba_ct == 0:  # trends during longer GEBA periods
                # remove mean during matching years
                sub_geba = vals.copy()
                mu_geba = np.mean(sub_geba)
                sub_geba_anom = sub_geba - mu_geba
                shared_years = np.intersect1d(years, np.arange(1980, 2025))
                X = shared_years - np.mean(shared_years)
                slope_geba, intercept_geba, r_value, _, _ = linregress(X, sub_geba_anom)
                trends_save[0, ct] = slope_geba

            for o_ct, rname in enumerate(reanalysis_names):

                # Get reanalysis
                this_ts_reanalysis = my_xutils.compute_annual_mean_of_full_years(
                    all_sw_native_loaded[o_ct].
                    sel(lat=this_lat, lon=this_lon, method='nearest')).squeeze()

                # Match reanalysis to GEBA
                shared_years = np.intersect1d(years, this_ts_reanalysis.year)
                X = shared_years - np.mean(shared_years)
                sub_reanalysis = this_ts_reanalysis.sel(year=shared_years)

                # Get idx to match GEBA to reanalysis (relevant for GEWEX only)
                idx_match = np.isin(years, sub_reanalysis.year)
                assert (years[idx_match] == sub_reanalysis.year).all()

                if (rname == 'GEWEX') & (geba_ct == 0):  # need to calculate trends using shorter period

                    sub_geba = vals[idx_match]
                    mu_geba = np.mean(sub_geba)
                    sub_geba_anom = sub_geba - mu_geba
                    slope_geba, intercept_geba, r_value, _, _ = linregress(X, sub_geba_anom)
                    trends_save[-1, ct] = slope_geba

                mu_reanalysis = sub_reanalysis.mean('year')
                sub_reanalysis_anom = sub_reanalysis - mu_reanalysis

                if geba_ct == 0:
                    slope_reanalysis, intercept_reanalysis, _, _, _ = linregress(X, sub_reanalysis_anom)
                    trends_save[o_ct + 1, ct] = slope_reanalysis

                # For all sets, calculate correlation
                rho_geba_reanalysis[o_ct, ct] = np.corrcoef(sub_reanalysis_anom.values,
                                                            vals[idx_match])[0, 1]

        # Save!
        if geba_ct == 0:
            np.savez(savename_trends,
                     lats=geba_lats, lons=geba_lons, trends_save=trends_save)
        np.savez(savename_interannual,
                 lats=geba_lats, lons=geba_lons, rho_geba_reanalysis=rho_geba_reanalysis)

# ## Figure 1 (ERA5) and Supplemental figures (JRA3Q, MERRA2, CLARA, GEWEX)
#
# 6 maps:
# - first row: interannual variability (one colorbar)
# - second row: trends in CERES and ERA5 (second colorbar)
# - third row: trends in GEBA and ERA5 (shared with second row)
geba_savename_trends = '%s/geba_reanalysis_with_RS_trends.npz' % procdir
geba_savename_interannual = '%s/geba_reanalysis_with_RS_corr_full_record_20yearsmin.npz' % procdir

for r_ct, rname in enumerate(reanalysis_names):

    figname = '%s/validate_%s.png' % (figdir, rname)

    ceres_savename = '%s/validation_metrics_%s.nc' % (procdir, rname)

    # Get relevant fields
    ds_ceres_validation = xr.open_dataset(ceres_savename)
    ds_geba_trends = np.load(geba_savename_trends)
    ds_geba_interannual = np.load(geba_savename_interannual)

    countries_for_plot = []
    is_china = []
    for this_lat, this_lon in zip(ds_geba_interannual['lats'], ds_geba_interannual['lons']):
        country = md.loc[(md['sgxlon'] == this_lon) & (md['sgylat'] == this_lat), 'affiliation']
        countries_for_plot.append(country.values[0])
        is_china.append('China' in country.values[0])

    # print out metrics for paper

    r_avg = my_xutils.area_weighted_average(ds_ceres_validation['correlation'].
                                            where(ds_ceres_validation['sig_corr']))
    print('Area-weighted avg corr, %s-CERES: %0.2f' % (rname, r_avg.values))

    print('Median corr with GEBA: %0.2f' % np.median(ds_geba_interannual['rho_geba_reanalysis'][r_ct, :]))
    corr_outside_china = ds_geba_interannual['rho_geba_reanalysis'][r_ct, ~(np.array(is_china).astype(bool))]
    print('Median corr without China: %0.2f' % np.median(corr_outside_china))
    corr_trends = np.corrcoef(ds_geba_trends['trends_save'][0, :],
                              ds_geba_trends['trends_save'][r_ct + 1, :])[0, 1]
    corr_trends_no_china = np.corrcoef(ds_geba_trends['trends_save'][0, ~(np.array(is_china).astype(bool))],
                                       ds_geba_trends['trends_save'][r_ct + 1,
                                                                     ~(np.array(is_china).astype(bool))])[0, 1]
    bias_trends = (- np.mean(ds_geba_trends['trends_save'][0, :])
                   + np.mean(ds_geba_trends['trends_save'][r_ct + 1, :]))
    print('correlation across stations of trends: %0.2f' % corr_trends)
    print('correation without China: %0.2f' % corr_trends_no_china)
    print('bias trends: %0.2f' % bias_trends)

    area_frac_mask = (my_xutils.area_weighted_average(ds_ceres_validation['quality_mask'] == 1) /
                      my_xutils.area_weighted_average(~np.isnan(ds_ceres_validation['quality_mask'])))
    print('Fraction of land area un-masked: %0.2f' % area_frac_mask)
    print('******')

    nrows = 3
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             gridspec_kw={'hspace': 0.02, 'wspace': 0.02})  # adjust spacing)

    fig_names = ['Corr(%s, CERES)' % rname,
                 'Corr(%s, GEBA)' % rname,
                 '%s trend subset to CERES period of record' % rname,
                 'CERES trend',
                 '%s trend subset to GEBA period of record' % rname,
                 'GEBA trend']

    text_ct = 0
    for row_ct in range(nrows):
        for col_ct in range(ncols):
            # First row: interannual variability
            if (row_ct == 0) & (col_ct == 0):
                to_plot = ds_ceres_validation['correlation']
                is_grid = True
            elif (row_ct == 0) & (col_ct == 1):
                to_plot = ds_geba_interannual['rho_geba_reanalysis'][r_ct, :]
                this_lats = ds_geba_interannual['lats']
                this_lons = ds_geba_interannual['lons']
                is_grid = False
            elif (row_ct == 1) & (col_ct == 0):
                to_plot = ds_ceres_validation['CERES_era_trend']  # reanalysis trend during CERES era
                is_grid = True
            elif (row_ct == 1) & (col_ct == 1):
                ceres_idx = np.where(['%s_%s' % ('CERES', this_var) == n for n in name_list])[0][0]
                to_plot = trend_maps[ceres_idx]  # CERES trend

                is_grid = True
            elif (row_ct == 2) & (col_ct == 0):  # reanalysis trends overlapping with GEBA
                to_plot = ds_geba_trends['trends_save'][r_ct + 1, :]
                this_lats = ds_geba_trends['lats']
                this_lons = ds_geba_trends['lons']
                is_grid = False
            elif (row_ct == 2) & (col_ct == 1):
                if rname == 'GEWEX':
                    to_plot = ds_geba_trends['trends_save'][-1, :]  # GEBA trends subset to GEWEX period
                else:
                    to_plot = ds_geba_trends['trends_save'][0, :]  # GEBA trends
                this_lats = ds_geba_trends['lats']
                this_lons = ds_geba_trends['lons']
                is_grid = False

            if row_ct == 0:
                cbar_dict = pos_corr_cbar
                to_hatch = ds_ceres_validation['sig_corr']
            else:
                cbar_dict = trend_cbar
                to_hatch = (ds_ceres_validation['sig_diff_trends'] == 0).astype(int)

            ax = axes[row_ct, col_ct]
            if is_grid:
                to_plot = to_plot.where(analysis_mask)
                im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'], add_colorbar=False)

            else:
                im = ax.scatter(this_lons, this_lats, c=to_plot, s=20, edgecolor='none', zorder=3,
                                norm=cbar_dict['norm'], cmap=cbar_dict['cmap'], transform=ccrs.PlateCarree())

            if to_hatch is not None:
                to_hatch = to_hatch.where(analysis_mask)
                contour = to_hatch.plot.contourf(ax=ax, levels=[-0.5, 0.5], hatches=['....', None],
                                                 colors='none', add_colorbar=False, zorder=4)

            ax.set_extent([-180, 180, -60, 80])

            # Add map features
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
            ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
            ax.coastlines(zorder=3, lw=1)
            ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
            ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
            ax.set_title('(%s) %s' % (letters[text_ct], fig_names[text_ct]))

            text_ct += 1

    # Get colorbar positions
    pos1 = (fig.axes[1]).get_position()
    cbar1_bottom = pos1.y0
    cbar1_height = pos1.y1 - pos1.y0

    pos2 = (fig.axes[3]).get_position()
    pos3 = (fig.axes[5]).get_position()
    cbar2_bottom = pos3.y0
    cbar2_height = pos2.y1 - pos3.y0

    # Add colorbars
    from matplotlib.cm import ScalarMappable

    # Colorbar 1: top row
    sm_top = ScalarMappable(cmap=pos_corr_cbar['cmap'], norm=pos_corr_cbar['norm'])
    sm_top.set_array([])

    cax_top = fig.add_axes([0.92, cbar1_bottom, 0.015, cbar1_height])  # [left, bottom, width, height]
    cbar_top = fig.colorbar(sm_top, cax=cax_top, orientation='vertical', extend='min')
    cbar_top.set_label('Correlation', fontsize=12)

    # Colorbar 2: bottom two rows
    sm_bot = ScalarMappable(cmap=trend_cbar['cmap'], norm=trend_cbar['norm'])
    sm_bot.set_array([])

    cax_bot = fig.add_axes([0.92, cbar2_bottom, 0.015, cbar2_height])
    cbar_bot = fig.colorbar(sm_bot, cax=cax_bot, orientation='vertical', extend='both')
    cbar_bot.set_label(trend_str, fontsize=12)

    plt.savefig(figname, dpi=200, bbox_inches='tight')


# # Figure 2: ERA5 trends, causes, and case studies
# (a) ERA5 trend with hatching
#
# (b) AOD trends
#
# (c) Total cloud trends
#
# (d)-(h) Case studies
era5_idx = np.where(['ERA5_%s' % (this_var) == n for n in name_list])[0][0]

# Load validation mask for ERA5
ceres_savename = '%s/validation_metrics_%s.nc' % (procdir, 'ERA5')
ds_validation = xr.open_dataset(ceres_savename)
quality_mask = ds_validation['quality_mask']

# Trends in AOD
da_aod_ann = (my_xutils.compute_annual_mean_of_full_years(da_aod)).sel(year=slice('%04i' % years_to_use[0],
                                                                                  '%04i' % years_to_use[1]))
beta_aod = my_xutils.compute_linear_trend_per_year(da_aod_ann)

# Trends in total cloud
cloud_file = '/home/data/ERA5/month/total_cloud_cover/total_cloud_cover.nc'
da_tc = xr.open_dataarray(cloud_file)
da_tc = da_tc.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'})
da_tc = ((da_tc.sortby('lat'))).interp(lat=shared_lats, lon=shared_lons).load()
da_tc_ann = (my_xutils.compute_annual_mean_of_full_years(da_tc)).sel(year=slice('%04i' % years_to_use[0],
                                                                                '%04i' % years_to_use[1]))
beta_tc = my_xutils.compute_linear_trend_per_year(da_tc_ann)

fig = plt.figure(figsize=(10, 8))
nrows = 6
ncols = 2

locs_case_studies = {'central/west US': [[30, 45], [235, 265]],
                     'central South America': [[-35, 0], [295, 320]],
                     'Europe': [[42, 57], [0, 32]],
                     'southeastern China': [[21, 38], [100, 120]],
                     'India': [[7, 29], [68, 90]]}

cbar_names = (trend_str,
              'AOD trend x 1000 ([]/year)',
              'Total cloud trend (%/year)')

# lefthand side: maps
# righthand side: time series
gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig,
                       width_ratios=[1.5, 1],
                       wspace=0.1, hspace=0.8)
proj = ccrs.PlateCarree()

# lefthand side
for ct in range(3):
    ax = fig.add_subplot(gs[(ct * 2):(ct * 2 + 2), 0], projection=proj)
    # cbar_ax = fig.add_subplot(gs[(ct * 3 + 2), 0])

    if ct == 0:  # ERA5 trends
        to_plot = trend_maps[era5_idx].where(analysis_mask).where(quality_mask)
        to_hatch = quality_mask
        cbar_dict = trend_cbar
        extend = 'both'

    elif ct == 1:  # AOD trends

        to_plot = 1000 * beta_aod.where(analysis_mask)  # just to make units on colorbar nicer
        levels = np.arange(-4, 4.1, 0.5)
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
        cbar_dict = {'levels': levels, 'norm': norm, 'cmap': 'Spectral_r'}
        extend = 'both'
        to_hatch = None

    else:  # Cloud trend
        to_plot = 100 * beta_tc  # 100 to switch from fraction to percent
        to_plot = to_plot.where(quality_mask).where(analysis_mask)
        to_hatch = quality_mask
        levels = np.arange(-.15, .16, 0.03)
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
        cbar_dict = {'levels': levels, 'norm': norm, 'cmap': 'RdBu'}
        extend = 'both'

    # Map look
    ax.set_extent([-180, 180, -60, 80])
    ax.coastlines(zorder=3, lw=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

    # Plot without auto colorbar
    im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'],
                      add_colorbar=False)

    # --- Horizontal colorbar under the map, same width ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='6%', pad=0.2, axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
    cb.set_label(cbar_names[ct], labelpad=0)
    cb.ax.tick_params(length=3)

    # Optional hatching
    if to_hatch is not None:
        to_hatch.plot.contourf(ax=ax, levels=[-0.5, 0.5],
                               hatches=['....', None], colors='none',
                               add_colorbar=False)
    ax.set_title('')
    ax.text(0.02, 0.1, '(%s)' % letters[ct], color='k', fontsize=12,
            ha='left', transform=ax.transAxes)

    for name, ((lat_min, lat_max), (lon_min, lon_max)) in locs_case_studies.items():
        width = lon_max - lon_min
        height = lat_max - lat_min
        rect = mpatches.Rectangle(
            (lon_min, lat_min), width, height,
            linewidth=1, edgecolor='tab:blue', facecolor='none',
            transform=ccrs.PlateCarree(), zorder=5
        )
        ax.add_patch(rect)


# righthand side
for ct_loc, this_loc in enumerate(locs_case_studies.keys()):
    these_coords = locs_case_studies[this_loc]

    vars_to_plot = 'SW', 'total cloud', 'AOD', 'ISCCP'

    all_ts = []

    for v in vars_to_plot:
        if v == 'SW':
            this_da = all_sw[era5_idx]
        elif v == 'total cloud':
            this_da = ds_clouds_era5_coarse['total_cloud_cover'] * 100  # cloud percentage
        elif v == 'AOD':
            this_da = 1000 * da_aod  # nicer units
        elif v == 'ISCCP':
            this_da = (da_cldamt_isccp.sum('cloud_irtype'))

        this_da = this_da.sel(time=slice('%04i' % years_to_use[0], '%04i' % years_to_use[1]))
        this_ts = this_da.sel(lat=slice(these_coords[0][0], these_coords[0][1]),
                              lon=slice(these_coords[1][0], these_coords[1][1]))
        this_ts = my_xutils.compute_annual_mean_of_full_years(this_ts.where(quality_mask == 1))
        this_ts = my_xutils.area_weighted_average(this_ts)
        all_ts.append(this_ts.load())

    ax = fig.add_subplot(gs[ct_loc, 1])

    # Main axis
    (all_ts[0] - all_ts[0].mean()).plot(label=r'ERA5 SW$_\downarrow$', color='k')

    # First twin axis (blue + purple)
    ax2 = ax.twinx()
    (all_ts[1] - all_ts[1].mean()).plot(ax=ax2, label='ERA5 total cloud (%)', color='tab:blue')
    if this_loc != 'India':
        (all_ts[3] - all_ts[3].mean()).plot(ax=ax2, label='ISCCP total cloud (%)', color='tab:blue',
                                            ls='--')
    else:
        sub_plot = all_ts[3].sel(year=slice('1999', None))
        (sub_plot - sub_plot.mean()).plot(ax=ax2, label='ISCCP total cloud (%)', color='tab:blue', ls='--')
    ax2.invert_yaxis()
    ax2.spines['right'].set_position(('axes', 1))  # offset to the right
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.spines['right'].set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:blue')

    # Second twin axis (orange)
    ax3 = ax.twinx()
    (all_ts[2] - all_ts[2].mean()).plot(ax=ax3, label='AOD x 1000', color='tab:orange')

    ax3.spines['right'].set_position(('axes', 1.15))  # further offset
    ax3.tick_params(axis='y', colors='tab:orange')
    ax3.spines['right'].set_color('tab:orange')
    ax3.yaxis.label.set_color('tab:orange')

    if ct_loc == 0:
        lines, labels = [], []
        for a in [ax, ax2, ax3]:
            lns, lbls = a.get_legend_handles_labels()
            lines.extend(lns)
            labels.extend(lbls)

        fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.77, 0.1))
    # --- Force all axes to align zero in the middle ---
    for a in [ax, ax2, ax3]:
        ymin, ymax = a.get_ylim()
        m = max(abs(ymin), abs(ymax))   # symmetric around 0
        a.set_ylim(-m, m)
        a.set_title('')
        a.set_xlabel('')
        a.set_xticks([])
        a.set_xlim(years_to_use[0] - 2, years_to_use[1] + 2)
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax.set_title('(%s) %s' % (letters[ct + ct_loc + 1], this_loc))
ax.set_xticks(np.arange(years_to_use[0], years_to_use[1] + 5, 10))

plt.savefig('%s/fig02.png' % figdir, dpi=200, bbox_inches='tight')


# # Figure 3
#
# 4 panels:
# - CMIP6 MMM
# - ranks of ERA5
# - histograms of ranks
# - similarity histograms

# Load CMIP6 trends and calculate rank of obs
# Calculated in a different notebook using Pangea data

cmip6_fname = 'CMIP6_rsds_historical-ssp370_%04i-%04i.nc' % (years_to_use[0], years_to_use[1])
cmip6_fname_ssp245 = 'CMIP6_rsds_historical-ssp245_1980-2024.nc'

cmip6_trends = xr.open_dataarray('%s/%s' % (procdir, cmip6_fname))
cmip6_trends_ssp245 = xr.open_dataarray('%s/%s' % (procdir, cmip6_fname_ssp245))

# For supplement: compare to using SSP2-4.5
if 'ssp2' in figdir:
    # Subset to shared models for consistency with main results
    shared_models = np.isin(cmip6_trends_ssp245.base_model, cmip6_trends.base_model)
    cmip6_trends = cmip6_trends_ssp245.isel(model=shared_models)
    cmip6_trends = cmip6_trends.drop('base_model')
    model_names = ([m.split('_')[0] for m in cmip6_trends.model.values])
    cmip6_trends = cmip6_trends.assign_coords(base_model=('model', model_names))

model_means = cmip6_trends.groupby('base_model').mean(dim='model')
MMM = model_means.mean(dim='base_model')
if 'ssp2' in figdir:
    ens_names = ['CMIP6']
else:
    ens_names = ['CESM2', 'CMIP6']

obs_trend = trend_maps[era5_idx]
obs_trend_masked = obs_trend.where(quality_mask).where(analysis_mask)

all_ranks = []

# Collect trends across ensembles
# And also calculate rank within each ensemble
all_members = []
for ens_name in ens_names:
    print(ens_name)
    if ens_name == 'CMIP6':
        ens_trends = cmip6_trends
        ens_dim_name = 'model'
    else:
        this_str = '%s_%s' % (ens_name, this_var)
        this_idx = np.where([this_str == n for n in name_list])[0][0]
        ens_trends = trend_maps[this_idx]
        ens_dim_name = 'member'

    rank_obs_with_ensemble = xr.apply_ufunc(
        my_utils.rank_func,
        ens_trends,
        obs_trend,
        input_core_dims=[[ens_dim_name], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    all_ranks.append(rank_obs_with_ensemble.rename('ERA5_in_%s' % ens_name))
    if ens_name == 'CMIP6':
        ens_trends = ens_trends.rename({'model': 'member'}).drop('base_model')
    all_members.append(ens_trends)
all_members = xr.concat(all_members, dim='member')

rank_obs_with_ensemble = xr.apply_ufunc(
    my_utils.rank_func,
    all_members,
    obs_trend,
    input_core_dims=[['member'], []],
    output_core_dims=[[]],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
)
all_ranks.append(rank_obs_with_ensemble.rename('ERA5_in_all_ensembles'))
all_ranks = xr.merge(all_ranks)

# For null hypothesis later, also rank each member of the super-ensemble against the obs
savename = '%s/null_ranks_%s_%04i-%04i.nc' % (procdir, '_'.join(ens_names), years_to_use[0], years_to_use[1])

if os.path.isfile(savename):  # slow to run
    null_ranks = xr.open_dataarray(savename)
else:
    null_ranks = []
    for this_member in all_members.member.values:
        sub_ens = all_members.drop_sel(member=this_member)
        sub_ens = xr.concat((sub_ens, obs_trend.expand_dims(member=['obs'])), dim='member')
        this_run = all_members.sel(member=this_member)
        rank_runs_with_ensemble = xr.apply_ufunc(
            my_utils.rank_func,
            sub_ens,
            this_run,
            input_core_dims=[['member'], []],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )
        null_ranks.append(rank_runs_with_ensemble)
    null_ranks = xr.concat(null_ranks, dim='member')
    null_ranks['member'] = all_members['member']
    null_ranks.to_netcdf(savename)

fig_names = 'CMIP6 MMM', 'Rank of ERA5 within %s' % '+'.join(ens_names)
cbar_names = trend_str, 'Rank'

nrows = 2
ncols = 2
fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
proj = ccrs.PlateCarree()
gs = gridspec.GridSpec(
    nrows=nrows, ncols=ncols, figure=fig,
    height_ratios=[1, 0.6],  # a bit more room for maps + colorbars
    hspace=0, wspace=0.18)

for ct in range(4):
    if ct < 2:  # first row, maps
        ax = fig.add_subplot(gs[0, ct], projection=proj)
    else:  # second row, histograms
        ax = fig.add_subplot(gs[1, ct % 2])

    if ct < 2:
        if ct == 0:  # CMIP trends
            to_plot = MMM.where(analysis_mask)
            cbar_dict = trend_cbar
            to_hatch = None
            extend = 'both'
        else:
            to_plot = all_ranks['ERA5_in_all_ensembles'].where(analysis_mask)
            to_plot = to_plot.where(quality_mask)
            rank_min, rank_max = int(to_plot.min()), int(to_plot.max())

            ranks_edge = int(np.ceil(0.025 * (rank_max)))
            levels = np.hstack((np.arange(rank_min - 0.5, rank_min + ranks_edge),
                                np.arange(rank_max + 0.5 - ranks_edge, rank_max + 1.5)))
            norm = mcolors.BoundaryNorm(levels, ncolors=256)
            cbar_dict = {'levels': levels, 'norm': norm, 'cmap': 'RdBu_r'}
            extend = 'neither'
            to_hatch = quality_mask

        ax.set_extent([-180, 180, -60, 80])
        ax.coastlines(zorder=3, lw=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
        ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

        im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'],
                          add_colorbar=False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.15, axes_class=plt.Axes)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
        cb.set_label(cbar_names[ct], labelpad=0)
        cb.ax.tick_params(length=3)
        if ct == 1:
            mid_bins = np.hstack(((np.ceil(levels)[:ranks_edge]).astype(int),
                                  (np.ceil(levels)[-(ranks_edge + 1):-1]).astype(int)))
            cb.set_ticks(mid_bins)

        # Optional hatching
        if to_hatch is not None:
            to_hatch.plot.contourf(ax=ax, levels=[-0.5, 0.5],
                                   hatches=['....', None], colors='none',
                                   add_colorbar=False)
        ax.set_title('(%s) %s' % (letters[ct], fig_names[ct]))

    if ct == 2:  # rank histogram

        tail_width = 6

        # Calculate histogram of ranks in non-masked regions
        hist_vals_era5, bin_middle = my_utils.get_rank_hist(all_ranks['ERA5_in_all_ensembles'],
                                                            quality_mask,
                                                            rank_max)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Area fraction')

        # plot null
        keep_max_null = 0
        max_idx = np.nan

        all_hist_vals_null = np.empty((len(null_ranks.member), len(bin_middle)))
        for kk in range(len(null_ranks.member)):
            hist_vals_null, _ = my_utils.get_rank_hist(null_ranks.isel(member=kk),
                                                       quality_mask,
                                                       rank_max)
            if hist_vals_null[-1] > keep_max_null:
                keep_max_null = hist_vals_null[-1]
                max_idx = kk
            all_hist_vals_null[kk, :] = hist_vals_null

            ax.plot(bin_middle, hist_vals_null, color='gray', lw=0.5, alpha=0.5)
        ax.plot(bin_middle, hist_vals_null, color='gray', lw=0.5, alpha=0.5, label='Null density samples')
        ax.plot(bin_middle[-1], keep_max_null, marker='*', color='k', markersize=5, label='Maximum null density')
        ax.plot(bin_middle, all_hist_vals_null[max_idx, :], color='k')
        ax.plot(bin_middle, hist_vals_era5, color='tab:red', label='ERA5 density')
        ax.legend(loc='upper left')
        ax.set_title('(%s) Rank histogram' % (letters[ct]))

        # Inset axis
        inset_ax = ax.inset_axes([0.63, 0.55, 0.3, 0.35])
        inset_ax.hist(all_hist_vals_null[:, bin_middle > (rank_max - tail_width)].sum(axis=1), color='gray')
        inset_ax.axvline(hist_vals_era5[bin_middle > (rank_max - tail_width)].sum(), color='tab:red')

        inset_ax.set_xlabel('Area fraction in\nmax %i ranks' % tail_width, fontsize=8)
        inset_ax.set_ylabel('Count', fontsize=8)
        inset_ax.tick_params(labelsize=8)

        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.8)

    elif ct == 3:  # correlation histograms plot
        nmembers = len(all_members.member)
        trends_masked = all_members.where(quality_mask).where(analysis_mask)

        savename = '%s/pattern_corr_ERA5_%s_%04i-%04i.npz' % (procdir, '_'.join(ens_names),
                                                              years_to_use[0], years_to_use[1])

        if os.path.isfile(savename):
            rho_load = np.load(savename)
            rho_member_trends = rho_load['rho_member_trends']
            rho_member_with_era5 = rho_load['rho_member_with_era5']
        else:

            rho_member_trends = np.nan * np.ones((nmembers, nmembers))
            rho_member_with_era5 = np.nan * np.ones((nmembers, ))
            for ct1, m1 in enumerate(trends_masked.member):

                rho_member_with_era5[ct1] = my_xutils.xr_weighted_corr(trends_masked.sel(member=m1),
                                                                       obs_trend_masked)

                for ct2, m2 in enumerate(trends_masked.member):
                    if ct2 <= ct1:
                        continue
                    this_rho = my_xutils.xr_weighted_corr(trends_masked.sel(member=m1),
                                                          trends_masked.sel(member=m2))
                    rho_member_trends[ct1, ct2] = this_rho

            np.savez(savename, rho_member_trends=rho_member_trends, rho_member_with_era5=rho_member_with_era5)

        print('Mean ERA5-model corr: %0.2f' % np.mean(rho_member_with_era5))
        print('Mean model-model corr: %0.2f' % np.nanmean(rho_member_trends.flatten()))
        bins = np.arange(0.2, 1, 0.02)
        ax.hist(rho_member_trends.flatten()[~np.isnan(rho_member_trends.flatten())], density=True,
                label='Within %s' % '+'.join(ens_names), color='k', bins=bins)
        ax.hist(rho_member_with_era5, density=True, label='ERA5-%s' % '+'.join(ens_names),
                alpha=0.8, color='tab:blue', bins=bins)
        ax.legend()
        ax.set_xlabel('Correlation')
        ax.set_title('(%s) Spatial correlation histograms' % (letters[ct]))

plt.savefig('%s/fig03.png' % figdir, dpi=200, bbox_inches='tight')


# # Supplemental figures

# ## Trends in all products without ocean masked
# ERA5, JRA-3Q, MERRA2, CLARA, GEWEX
# Plot
nrows, ncols = 3, 2
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(6 * ncols, 3 * nrows),
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
)

for ct, rname in enumerate(reanalysis_names):
    ax = axes.flatten()[ct]
    this_idx = np.where(['%s_%s' % (rname, this_var) == n for n in name_list])[0][0]
    to_plot = trend_maps[this_idx]
    this_start_year = start_year[this_idx]
    this_end_year = end_year[this_idx]
    cbar_dict = trend_cbar
    extend = 'both'

    # Map look
    ax.set_extent([-180, 180, -60, 80])
    ax.coastlines(zorder=3, lw=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

    im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'], add_colorbar=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='6%', pad=0.15, axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
    cb.set_label(trend_str, labelpad=0)
    cb.ax.tick_params(length=3)

    ax.set_title('(%s) %s (%i-%i)' % (letters[ct], rname, this_start_year, this_end_year))
fig.delaxes(axes.flatten()[-1])

plt.savefig('%s/SUPP_all_trends_with_ocean.png' % figdir, dpi=200, bbox_inches='tight')


# ## Histograms for GEBA correlations across different products
ds_geba_interannual_long = np.load('%s/geba_reanalysis_with_RS_corr_full_record_20yearsmin.npz' % procdir)
ds_geba_interannual_preceres = np.load('%s/geba_reanalysis_with_RS_corr_pre_ceres_10yearsmin.npz' % procdir)

bins = np.arange(-1, 1.01, 0.1)
fig, ax = plt.subplots(figsize=(7, 3), ncols=2, nrows=1, sharey=True)
fig.tight_layout()
colors = ['k', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
xmin, xmax = -1.0, 1.0
xx = np.linspace(xmin, xmax, 512)

for hist_ct in range(2):
    this_ax = ax[hist_ct]
    if hist_ct == 0:
        to_plot = ds_geba_interannual_long
        title = 'GEBA correlation\n20y+ records'
    else:
        to_plot = ds_geba_interannual_preceres
        title = 'GEBA correlation pre-CERES\n10y+ records'
    alpha = 1
    for o_ct, rname in enumerate(reanalysis_names):
        # pull values, drop NaNs
        x = np.asarray(to_plot['rho_geba_reanalysis'][o_ct, :]).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        kde = gaussian_kde(x, bw_method='scott')
        yy = kde(xx)

        # plot smooth density and its median
        this_ax.plot(xx, yy, lw=2, alpha=alpha, color=colors[o_ct], label=rname)
        med = np.median(x)
        print('%s, %0.2f' % (rname, med))
        this_ax.axvline(med, lw=2, alpha=alpha, color=colors[o_ct])

        # optional light fill under curve
        this_ax.fill_between(xx, 0, yy, alpha=0.1, color=colors[o_ct])

        alpha -= 0.1

    this_ax.set_xlim(xmin, xmax)
    this_ax.set_ylim(bottom=0)
    this_ax.set_title(title)
    this_ax.set_xlabel('Correlation')
    if hist_ct == 0:
        this_ax.set_ylabel('Density')

# one legend is enough
ax[0].legend(frameon=False, ncol=1, title=None)
plt.savefig('%s/SUPP_GEBA_corr_histograms.png' % figdir, dpi=200, bbox_inches='tight')


# ## ERA5 clearsky
# Load ERA5 clearsky, and show trends
savename = '%s/ERA5_clearsky_rg.nc' % procdir
if os.path.isfile(savename):
    da_era5_clearsky = xr.load_dataarray(savename)
else:
    csvar = 'surface_solar_radiation_downward_clear_sky'
    da_era5_clearsky = xr.open_dataarray('%s/%s/%s.nc' % (data_dirs['ERA5'], csvar, csvar))
    # Same processing as SW_down
    da_era5_clearsky = da_era5_clearsky.rename({'valid_time': 'time'})
    adjusted_time = da_era5_clearsky.time.dt.floor('D')
    da_era5_clearsky['time'] = adjusted_time
    # Heating is in J/m2
    da_era5_clearsky /= sec_per_day
    da_era5_clearsky = da_era5_clearsky.rename({'latitude': 'lat', 'longitude': 'lon'})
    da_era5_clearsky = da_era5_clearsky.sortby('lat')

    da_interp = my_utils.regrid_to_shared_grid(da_era5_clearsky, shared_lats, shared_lons)
    da_era5_clearsky = da_interp.sel(time=slice('%04i' % years_to_use[0], '%04i' % years_to_use[1]))
    del da_interp
    da_era5_clearsky.to_netcdf(savename)

beta_cs = my_xutils.compute_linear_trend_per_year(
    my_xutils.compute_annual_mean_of_full_years(da_era5_clearsky))
name = r'ERA5 SW$_\downarrow$ clearsky trend (W/m$^2$/yr)'
plot_global_discrete(beta_cs.sel(lat=slice(-60, 80)).rename(name), levels=trend_cbar['levels'] / 2)

corr_with_AOD = my_xutils.xr_weighted_corr(beta_cs.where(analysis_mask), beta_aod.where(analysis_mask))
print('Correlation with MERRA2 AOD trends (analysis region only): %0.2f' % corr_with_AOD)

plt.savefig('%s/SUPP_ERA5_clearsky_trends.png' % figdir, dpi=200, bbox_inches='tight')


# ## Correlation of SW at the surface and cloud fraction at different levels, ERA5 and CERES/ISCCP

# ### CERES / ISCCP
# Add total variable
da_cldamt_isccp_incl_total = xr.concat((da_cldamt_isccp,
                                        da_cldamt_isccp.sum('cloud_irtype').rename('total')),
                                       dim='cloud_irtype')
isccp_cloud_names = 'low', 'middle', 'high', 'total'

# Predict SW at surface using co-albedo estimate from clouds (scaling is irrelevant)
cloud_pred = da_ceres_toa * (1 - da_cldamt_isccp_incl_total / 100)
# Do comparison at the annual mean level
cloud_pred_ann = my_xutils.compute_annual_mean_of_full_years(cloud_pred)

ceres_idx = np.where(['%s_%s' % ('CERES', this_var) == n for n in name_list])[0][0]
rho_clouds = xr.corr(my_xutils.compute_annual_mean_of_full_years(all_sw[ceres_idx]),
                     cloud_pred_ann, dim='year')

# Make plot
nrows = 2
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 3 * nrows),
                         subplot_kw={'projection': ccrs.PlateCarree()})
levels = np.arange(0, 1.1, 0.1)
cmap = 'Reds'
norm = BoundaryNorm(levels, ncolors=256)

for ct, ax in enumerate(axes.flatten()):

    im = rho_clouds.isel(cloud_irtype=ct).sel(lat=slice(-60, 80)).plot(ax=ax,
                                                                       cmap=cmap,
                                                                       norm=norm,
                                                                       levels=levels,
                                                                       add_colorbar=False)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=1)
    ax.text(0.02, 0.1, '%s' % isccp_cloud_names[ct], color='k', fontsize=12, ha='left', transform=ax.transAxes)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(im, cax=cbar_ax, ticks=levels, extend='neither')
cb.set_label('Correlation', fontsize=12)

plt.savefig('%s/SUPP_CERES_ISCCP_cloud_SW_corr.png' % figdir, dpi=200, bbox_inches='tight')


# ### ERA5
#
# - Low cloud is a single level field calculated from cloud occurring on model levels with a pressure greater than 0.8
# times the surface pressure.
# - Medium cloud is a single level field calculated from cloud occurring on model levels with a pressure between 0.45
# and 0.8 times the surface pressure.
# - High cloud is a single level field calculated from cloud occurring on model levels with a pressure less than 0.45
# times the surface pressure.
era5_cloud_names = ['%s_cloud_cover' % c for c in list(('low', 'medium', 'high', 'total'))]
savename = '%s/rho_clouds_era5.nc' % procdir
if os.path.isfile(savename):
    rho_clouds_era5 = xr.open_dataarray(savename)
else:
    ds_clouds = []
    for var in era5_cloud_names:
        this_dir = '%s/%s' % (data_dirs['ERA5'], var)
        da = xr.open_dataarray('%s/%s.nc' % (this_dir, var))
        da = da.rename({'valid_time': 'time'})
        adjusted_time = da.time.dt.floor('D')
        da['time'] = adjusted_time
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
        da = da.sortby('lat').rename(var)
        da = da.sel(time=slice('%04i' % years_to_use[0],
                               '%04i' % years_to_use[1]))
        da = my_utils.regrid_to_shared_grid(da, shared_lats, shared_lons)
        ds_clouds.append(da.load())
    ds_clouds = xr.merge(ds_clouds)

    # Load TOA
    toa_var = 'toa_incident_solar_radiation'
    this_dir = '%s/%s' % (data_dirs['ERA5'], toa_var)
    da_toa = xr.open_dataarray('%s/%s.nc' % (this_dir, toa_var))
    da_toa = da_toa.rename({'valid_time': 'time'}).drop('expver')
    adjusted_time = da_toa.time.dt.floor('D')
    da_toa['time'] = adjusted_time
    da_toa = da_toa.rename({'latitude': 'lat', 'longitude': 'lon'})
    da_toa = da_toa.sortby('lat').rename(toa_var)
    da_toa = my_utils.regrid_to_shared_grid(da_toa, shared_lats, shared_lons)
    da_toa = da_toa.sel(time=slice('%04i' % years_to_use[0], '%04i' % years_to_use[1]))

    pred_surf_sw = (da_toa * (1 - ds_clouds)).to_array(dim='cloud_type')

    rho_clouds_era5 = xr.corr(my_xutils.compute_annual_mean_of_full_years(all_sw[era5_idx]),
                              my_xutils.compute_annual_mean_of_full_years(pred_surf_sw), dim='year')

    rho_clouds_era5.to_netcdf(savename)

# Make plot
nrows = 2
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 3 * nrows),
                         subplot_kw={'projection': ccrs.PlateCarree()})
levels = np.arange(0, 1.1, 0.1)
cmap = 'Reds'
norm = BoundaryNorm(levels, ncolors=256)

for ct, ax in enumerate(axes.flatten()):

    im = rho_clouds_era5.isel(cloud_type=ct).sel(lat=slice(-60, 80)).plot(ax=ax,
                                                                          cmap=cmap,
                                                                          norm=norm,
                                                                          levels=levels,
                                                                          add_colorbar=False)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=1)
    ax.text(0.02, 0.1, '%s' % era5_cloud_names[ct].split('_')[0], color='k', fontsize=12,
            ha='left', transform=ax.transAxes)
    ax.set_title('')

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(im, cax=cbar_ax, ticks=levels, extend='neither')
cb.set_label('Correlation', fontsize=12)
plt.savefig('%s/SUPP_ERA5_cloud_SW_corr.png' % figdir, dpi=200, bbox_inches='tight')


# ## Postage stamps for models
cmip6_trends_ens_mean = cmip6_trends.groupby('base_model').mean()
cmip6_trends_ens_count = cmip6_trends[:, 10, 10].groupby('base_model').count('model')

for model in cmip6_trends_ens_count.base_model:
    print('%s, %i' % (model.values, cmip6_trends_ens_count.sel(base_model=model)))

nmodels = len(cmip6_trends_ens_mean.base_model)

nrows = 12
ncols = 3

cmap = 'RdBu_r'

fig, axes = plt.subplots(figsize=(15, 25), nrows=nrows, ncols=ncols,
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for m_ct, model_name in enumerate(cmip6_trends_ens_mean.base_model):
    ax = axes[m_ct]

    if model_name == 'CESM2':  # use a bigger EM since we have it
        cesm2_idx = np.where(['CESM2_%s' % this_var == n for n in name_list])[0][0]
        idx = cmip6_trends['base_model'] == 'CESM2'
        this_ens = xr.concat((cmip6_trends.sel(model=idx).drop('base_model'),
                              trend_maps[cesm2_idx].rename({'member': 'model'})), dim='model')
        to_plot = this_ens.mean('model')
        this_count = len(this_ens['model'])
    else:
        to_plot = cmip6_trends_ens_mean.sel(base_model=model_name)
        this_count = cmip6_trends_ens_count.sel(base_model=model_name)
    im = (to_plot.sel(lat=slice(-60, 80))).plot(ax=ax, cmap=trend_cbar['cmap'],
                                                norm=trend_cbar['norm'],
                                                levels=trend_cbar['levels'],
                                                add_colorbar=False)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=1)

    ax.set_title('%s (%i)' % (str(model_name.values), int(this_count)))

# Shared colorbar
cbar_ax = fig.add_axes([0.95, 0.3, 0.015, 0.4])
cb = fig.colorbar(im, cax=cbar_ax, extend='both')
cb.set_label(trend_str, fontsize=12)
fig.delaxes(axes[-1])

plt.savefig('%s/SUPP_CMIP_postage_stamps.png' % figdir, dpi=200, bbox_inches='tight')


# ## Role of internal variability: S2N for CESM2 and CanESM2
cesm2_s2n = np.abs(trend_maps[cesm2_idx].mean('member')) / trend_maps[cesm2_idx].std('member')
da_canesm = cmip6_trends.sel(model=cmip6_trends['base_model'] == 'CanESM5')
canesm_s2n = np.abs(da_canesm.mean('model')) / da_canesm.std('model')

fig_names = 'CESM2 S2N', 'CanESM5 S2N'
nrows, ncols = 2, 1
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(7 * ncols, 3 * nrows),
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
)

vmin, vmax, step = 0, 3, 0.2
levels = np.arange(vmin, vmax + step, step)
norm = mcolors.BoundaryNorm(levels, ncolors=256)
cbar_dict = {'levels': levels, 'norm': norm, 'cmap': 'Purples'}
extend = 'max'

ims = []  # collect plotted images

for ct, ax in enumerate(axes.flatten()):
    if ct == 0:
        to_plot = cesm2_s2n
    else:
        to_plot = canesm_s2n

    to_plot = to_plot.where(analysis_mask)

    # Map look
    ax.set_extent([-180, 180, -60, 80])
    ax.coastlines(zorder=3, lw=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

    # Plot without auto colorbar
    im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'],
                      add_colorbar=False)
    ims.append(im)

    # Optional hatching
    if to_hatch is not None:
        to_hatch.plot.contour(ax=ax, levels=[-0.5, 0.5],
                              hatches=['....', None], colors='none',
                              add_colorbar=False)

    ax.set_title('(%s) %s' % (letters[ct], fig_names[ct]))

# Add a colorbar axis below the whole figure
cbar_ax = fig.add_axes([0.16, 0.05, 0.7, 0.02])

cb = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', extend=extend)
cb.set_label('Signal-to-noise ratio', labelpad=0)
cb.ax.tick_params(length=3)

plt.savefig('%s/SUPP_S2N.png' % figdir, dpi=200, bbox_inches='tight')


# ## GOGA trends
goga_idx = np.where(['%s_%s' % ('GOGA', this_var) == n for n in name_list])[0][0]
goga_mean_trend = trend_maps[goga_idx].mean('member')

name = r'GOGA SW$_\downarrow$ trend (W/m$^2$/yr), %i-%i' % (start_year[goga_idx], end_year[goga_idx])
plot_global_discrete(goga_mean_trend.sel(lat=slice(-60, 80)).rename(name), levels=trend_cbar['levels'])

plt.savefig('%s/SUPP_GOGA_trends.png' % figdir, dpi=200, bbox_inches='tight')


# ## Comparison of GOGA and ERA5 for 1985-2015: trend patterns, ranks
# calculate ERA5 and GOGA trends over this period
cesm2_idx = np.where(['%s_%s' % ('CESM2', this_var) == n for n in name_list])[0][0]
for period_ct in range(2):
    if period_ct == 0:
        goga_period = 1985, 2015
    else:
        goga_period = 1980, 2020

    short_trends = []
    for ct in range(3):
        if ct == 0:
            this_idx = goga_idx
            name = 'GOGA'
        elif ct == 1:
            this_idx = era5_idx
            name = 'ERA5'
        else:
            this_idx = cesm2_idx
            name = 'CESM2'

        da_anom = all_sw[this_idx].groupby('time.month') - all_sw[this_idx].groupby('time.month').mean()
        ann_mean = my_xutils.compute_annual_mean_of_full_years(da_anom)
        ann_mean = ann_mean.sel(year=slice('%04i' % goga_period[0], '%04i' % goga_period[1]))
        trend = my_xutils.compute_linear_trend_per_year(ann_mean)
        if name == 'CESM2':
            trend = trend.rename({'member': 'LE_member'})
        short_trends.append(trend.rename(name))

    short_trends = xr.merge(short_trends)

    rank_obs_with_goga = xr.apply_ufunc(
            my_utils.rank_func,
            short_trends['GOGA'],
            short_trends['ERA5'],
            input_core_dims=[['member'], []],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )

    # Plot
    # (a) ERA5 trend
    # (b) GOGA trend
    # (c) ERA5 rank in GOGA

    fig_names = ('ERA5 (%04i-%04i)' % (goga_period[0], goga_period[1]),
                 'CESM2-AMIP MMM (%04i-%04i)' % (goga_period[0], goga_period[1]),
                 'Rank of ERA5 within CESM2-AMIP (%04i-%04i)' % (goga_period[0], goga_period[1]),
                 'CESM2-LE MMM (%04i-%04i)' % (goga_period[0], goga_period[1]))
    cbar_names = trend_str, trend_str, 'Rank', trend_str

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(7 * ncols, 3 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
    )

    for ct, ax in enumerate(axes.flatten()):
        if ct == 0:
            to_plot = short_trends['ERA5']
            to_hatch = quality_mask
            cbar_dict = trend_cbar
            extend = 'both'
        elif ct == 1:
            to_plot = short_trends['GOGA'].mean('member')
            cbar_dict = trend_cbar
            extend = 'both'
            to_hatch = None
        elif ct == 2:
            to_plot = rank_obs_with_goga.where(quality_mask)
            to_hatch = quality_mask
            rank_min, rank_max = int(to_plot.min()), int(to_plot.max())
            ranks_edge = int(np.ceil(0.025 * (rank_max)))
            levels = np.hstack((np.arange(rank_min - 0.5, rank_min + ranks_edge),
                                np.arange(rank_max + 0.5 - ranks_edge, rank_max + 1.5)))
            levels = np.arange(0.5, 12, 1)
            norm = mcolors.BoundaryNorm(levels, ncolors=256)
            cbar_dict = {'levels': levels, 'norm': norm, 'cmap': 'RdBu_r'}
            extend = 'neither'
        else:
            to_plot = short_trends['CESM2'].mean('LE_member')
            cbar_dict = trend_cbar
            extend = 'both'
            to_hatch = None

        to_plot = to_plot.where(analysis_mask)

        # Map look
        ax.set_extent([-180, 180, -60, 80])
        ax.coastlines(zorder=3, lw=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
        ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

        im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'],
                          add_colorbar=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.15, axes_class=plt.Axes)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
        cb.set_label(cbar_names[ct], labelpad=0)
        cb.ax.tick_params(length=3)
        if ct == 2:
            mid_bins = (levels[:-1] + 0.5).astype(int)
            cb.set_ticks(mid_bins)

        # Optional hatching
        if to_hatch is not None:
            to_hatch.plot.contourf(ax=ax, levels=[-0.5, 0.5, 1],
                                   hatches=['....', None], colors='none',
                                   add_colorbar=False, zorder=5)

        ax.set_title('(%s) %s' % (letters[ct], fig_names[ct]))

    plt.savefig('%s/SUPP_GOGA_comparison_%04i-%04i.png' % (figdir, goga_period[0], goga_period[1]),
                dpi=200, bbox_inches='tight')


# ## Trend for model that looks most like ERA5
nrows = 3
ncols = 1
fig = plt.figure(figsize=(6 * ncols, 3.5 * nrows))
proj = ccrs.PlateCarree()
gs = gridspec.GridSpec(
    nrows=nrows, ncols=ncols, figure=fig,
    hspace=0.12, wspace=0.18)

for ct in range(3):

    ax = fig.add_subplot(gs[ct], projection=proj)
    if ct == 0:
        max_corr_with_era5 = np.argmax(rho_member_with_era5)
        to_plot = all_members.isel(member=max_corr_with_era5)
        name = to_plot.member.values
    else:
        min_corr_within_ensemble = np.min(rho_member_trends[~np.isnan(rho_member_trends)])
        idx = np.where(rho_member_trends == min_corr_within_ensemble)
        if ct == 1:
            to_plot = all_members.isel(member=idx[0][0])
        else:
            to_plot = all_members.isel(member=idx[1][0])
        name = to_plot.member.values

    to_plot = to_plot.where(analysis_mask)

    cbar_dict = trend_cbar
    extend = 'both'

    ax.set_extent([-180, 180, -60, 80])
    ax.coastlines(zorder=3, lw=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

    # Plot without auto colorbar
    im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'],
                      add_colorbar=False)

    # --- Horizontal colorbar under the map, same width ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='6%', pad=0.15, axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
    cb.set_label(r'SW$_\downarrow$ trend (W/m$^2$/year)', labelpad=0)
    cb.ax.tick_params(length=3)

    ax.set_title('(%s) %s trend' % (letters[ct], name))

plt.savefig('%s/SUPP_example_members.png' % figdir, dpi=200, bbox_inches='tight')


# ## Rank of obs within each ensemble for total clouds
# - CMIP6: percentage
# - CESM2: fraction
# Load CMIP6 trends for total cloud
da_cmip_clt_beta = xr.open_dataarray('%s/CMIP6_clt_historical-ssp370_1980-2024.nc' % procdir)

# Supplement with CESM2
cloud_var = 'CLDTOT'
files = sorted(glob('%s/%s/b.e21.B*smbb.f09_g17.*.*.h0.%s.??????-??????.nc' %
                    (data_dirs['CESM2'], cloud_var, cloud_var)))
ens_members = np.unique(np.array(['.'.join(f.split('.')[4:6]) for f in files]))

files = [
    f for f in files
    if (years := my_utils.extract_years(f)) and (years[0] <= (years_to_use[-1] + 1)
                                                 and years[1] >= (years_to_use[0] + 1))
]

da_clt_cesm2 = []

# Load each ensemble member
for ens_member in ens_members:
    use_files = [f for f in files if ens_member in f]
    da = xr.open_mfdataset(use_files)[cloud_var]
    da = my_utils.regrid_to_shared_grid(da, shared_lats, shared_lons)
    da_clt_cesm2.append(da.load())

da_clt_cesm2 = xr.concat(da_clt_cesm2, dim='member')
da_clt_cesm2['member'] = ens_members
da_clt_cesm2 = da_clt_cesm2.rename(cloud_var)

# Move time by one month because of CESM2 timestamp issues for monthly data
# (the monthly averages are saved with the next month's time)
new_time = [cftime.DatetimeNoLeap(t.year, t.month - 1, t.day) if t.month > 1
            else cftime.DatetimeNoLeap(t.year - 1, 12, t.day)
            for t in da_clt_cesm2['time'].values]
da_clt_cesm2['time'] = new_time
da_clt_cesm2 = da_clt_cesm2.sel(time=slice('%04i' % years_to_use[0], '%04i' % years_to_use[-1]))

da_clt_cesm2_beta = my_xutils.compute_linear_trend_per_year(
    my_xutils.compute_annual_mean_of_full_years(da_clt_cesm2))

# Rank obs within CESM2 + CMIP6
da_model_ensemble_tc = xr.concat((da_clt_cesm2_beta * 100,
                                  da_cmip_clt_beta.rename({'model': 'member'}).drop('base_model')),
                                 dim='member')

rank_obs_with_ensemble_tc = xr.apply_ufunc(
    my_utils.rank_func,
    da_model_ensemble_tc,
    beta_tc * 100,  # switch ERA5 to percentage as well
    input_core_dims=[['member'], []],
    output_core_dims=[[]],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
)

fig_names = 'CMIP6 MMM', 'Rank of ERA5 within CMIP6+CESM2'
cbar_names = trend_str, 'Rank'

nrows = 1
ncols = 1
fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
proj = ccrs.PlateCarree()
gs = gridspec.GridSpec(
    nrows=nrows, ncols=ncols, figure=fig)

ax = fig.add_subplot(gs[0], projection=proj)

to_plot = rank_obs_with_ensemble_tc.where(analysis_mask).where(quality_mask)

rank_min, rank_max = int(to_plot.min()), int(to_plot.max())
ranks_edge = int(np.ceil(0.025 * (rank_max)))
levels = np.hstack((np.arange(rank_min - 0.5, rank_min + ranks_edge),
                    np.arange(rank_max + 0.5 - ranks_edge, rank_max + 1.5)))
norm = mcolors.BoundaryNorm(levels, ncolors=256)
cbar_dict = {'levels': levels, 'norm': norm, 'cmap': 'RdBu_r'}
extend = 'neither'
to_hatch = quality_mask

# Map look
ax.set_extent([-180, 180, -60, 80])
ax.coastlines(zorder=3, lw=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
ax.add_feature(cfeature.LAND, color='darkgray', zorder=0)
ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=1)
ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

# Plot without auto colorbar
im = to_plot.plot(ax=ax, norm=cbar_dict['norm'], cmap=cbar_dict['cmap'],
                  add_colorbar=False)

# --- Horizontal colorbar under the map, same width ---
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='6%', pad=0.15, axes_class=plt.Axes)
cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
cb.set_label('Rank', labelpad=0)
cb.ax.tick_params(length=3)

mid_bins = np.hstack(((np.ceil(levels)[:ranks_edge]).astype(int),
                      (np.ceil(levels)[-(ranks_edge + 1):-1]).astype(int)))
cb.set_ticks(mid_bins)

# Optional hatching

to_hatch.plot.contourf(ax=ax, levels=[-0.5, 0.5],
                       hatches=['....', None], colors='none',
                       add_colorbar=False)
ax.set_title('ERA5 rank within CMIP6+CESM2 total cloud trends')

plt.savefig('%s/SUPP_total_cloud_ranks.png' % figdir, dpi=200, bbox_inches='tight')
