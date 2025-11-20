import numpy as np
import intake
import xarray as xr
from helpful_utilities.ncutils import lon_to_360
import helpful_utilities.xutils as my_xutils
from SW_trends.utils import regrid_to_shared_grid
from tqdm import tqdm
from subprocess import check_call


var_to_use = 'rsds'
exps = ['historical', 'ssp370']
procdir = '/home/data/projects/SW-trends/proc'
cmd = 'mkdir -p %s' % procdir
check_call(cmd.split())
start_year = 1980
end_year = 2024
annual = True

# Open catalog
col_url = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.json'
col = intake.open_esm_datastore(col_url)

# Search for the variable and experiments
cat = col.search(
    variable_id=var_to_use,
    table_id='Amon',
    experiment_id=exps
)

# Work with a local copy of the catalog dataframe
df = cat.df.dropna(subset=['zstore']).copy()

# Create a combined ID of model and member
df['source_member'] = df['source_id'] + '.' + df['member_id']

# Keep only model/member pairs that have both experiments
counts = df.groupby(['source_member', 'experiment_id']).size().unstack()
valid_sources = counts.dropna().index.tolist()
df = df[df['source_member'].isin(valid_sources)]

# View example keys (you can check one with df.iloc[0])
print(df[['source_id', 'member_id', 'experiment_id']].drop_duplicates())

# Group by model/member
grouped = df.groupby(['source_id', 'member_id'])

trend_dict = {}

for (model, member), group_df in tqdm(grouped, desc='Processing models'):
    try:
        # Get paths for historical and ssp585
        zstores = dict(zip(group_df['experiment_id'], group_df['zstore']))
        if not {exps[0], exps[1]}.issubset(zstores):
            continue  # skip incomplete

        # Load both parts
        ds_hist = xr.open_zarr(zstores[exps[0]], consolidated=True)
        ds_ssp = xr.open_zarr(zstores[exps[1]], consolidated=True)

        # Concatenate time series
        ds = xr.concat([ds_hist, ds_ssp], dim='time')
        ds = ds.sel(time=slice('%04i-01' % start_year, '%04i-12' % end_year))

        if annual:
            # Compute annual mean
            da_ann = my_xutils.annual_average_from_monthly(ds[var_to_use])
            slope = my_xutils.compute_linear_trend_per_year(da_ann)

        else:  # trends for each momth
            slope = []
            for mo in range(1, 13):
                da_mon = ds[var_to_use].sel(time=ds['time.month'] == mo)
                yrs = da_mon['time.year'].values
                da_mon = da_mon.rename({'time': 'year'})
                da_mon['year'] = yrs
                this_slope = my_xutils.compute_linear_trend_per_year(da_mon)

                slope.append(this_slope)
            slope = xr.concat(slope, dim='month')
            slope['month'] = np.arange(1, 13)

        # Store
        trend_dict[f'{model}_{member}'] = slope

    except Exception as e:
        print(f'Skipped {model}_{member}: {e}')

# Regrid all to shared 1x1 grid
target_lat = np.arange(-89.5, 90.1, 1.0)
target_lon = np.arange(0.5, 360.1, 1.0)
target_grid = {'lat': target_lat, 'lon': target_lon}

trend_interp_list = []

for key, da in trend_dict.items():
    print(key)
    try:
        # Check if latitude is decreasing (common in CMIP6)
        if da.lat[0] > da.lat[-1]:
            da = da.sortby('lat')
        # Check that longitude is zero to 360
        if da.lon.min() < 0:
            da = lon_to_360(da)
        da = da.sortby('lon')

        da_interp = regrid_to_shared_grid(da, target_lat, target_lon)
        trend_interp_list.append(da_interp.load())

    except Exception as e:
        print(f'Skipped {key} due to interpolation error: {e}')


trend_stack = xr.concat(trend_interp_list, dim='model', coords='minimal', compat='override')
trend_stack['model'] = list(trend_dict.keys())[:len(trend_interp_list)]

model_names = ([m.split('_')[0] for m in trend_stack.model.values])


trend_stack = trend_stack.assign_coords(base_model=('model', model_names))

if annual:
    trend_stack.to_netcdf('%s/CMIP6_%s_%s-%s_%i-%i.nc' % (procdir, var_to_use, exps[0], exps[1],
                                                          start_year, end_year))
else:
    trend_stack.to_netcdf('%s/CMIP6_seasonal_%s_%s-%s_%i-%i.nc' % (procdir, var_to_use, exps[0], exps[1],
                                                                   start_year, end_year))
