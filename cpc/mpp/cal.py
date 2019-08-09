# Built-in
from typing import Union, Optional

# Third-party
import numpy as np
from scipy.stats import norm
import xarray as xr
import matplotlib.pyplot as plt

# This application
from .plot_poe_charts import plot_debug_poe
import mpp_driver.data.common


class StatsError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def ensemble_regression(raw_fcst: Union[xr.Dataset, xr.DataArray],
                        stats: Union[xr.Dataset, xr.DataArray], method='ensemble',
                        ens_size_correction=False, ptiles=None, debug=False, variable=None,
                        debug_poe_latlon=None, climo=None, num_years=None,
                        num_stats_members=None, num_fcst_members=None):
    # ----------------------------------------------------------------------------------------------
    # Check parameters
    #
    # Check for required parameters
    for arg in ['ptiles', 'variable', 'num_years', 'num_stats_members', 'num_fcst_members']:
        if locals()[arg] is None:
            raise ValueError(f'{arg} is a required argument')
    # Make sure raw_fcst (stats) is an xarray.DataArray (xarray.Dataset) without a date (model)
    # dimension
    raw_fcst = mpp_driver.data.common.to_dataarray(raw_fcst).drop('model').squeeze()
    stats = mpp_driver.data.common.to_dataset(stats).drop('date').squeeze()
    # Make sure the forecast only contains a single model
    try:
        if len(raw_fcst.model) > 1:
            raise ValueError(f"The forecast data contains {len(raw_fcst.model)} models, ensemble "
                             f"regression can only be applied to a single mdoel.")
    except AttributeError:
        pass
    # Make sure the stats data contains all required stats
    required_stats = {'cov', 'es', 'xm', 'xv', 'ym', 'yv'}
    if set(list(stats.data_vars)) != required_stats:
        raise StatsError(f'Some stats are missing, required stats are: {required_stats}')

    # ----------------------------------------------------------------------------------------------
    # Set some config options
    #
    xv_min = 0.1    # min xv allowed
    rxy_min = 0.05  # min rxy allowed
    rxy_max = 0.90  # max rxy allowed

    # ----------------------------------------------------------------------------------------------
    # Extract fields from stats dict
    #
    cov = stats['cov']
    es = stats['es']
    xm = stats['xm']
    xv = stats['xv']
    ym = stats['ym']
    yv = stats['yv']

    # ----------------------------------------------------------------------------------------------
    # Set bounds for some stats
    #
    # Set lower bound for xv
    xv = xv.where(xv > xv_min, xv_min)
    # Set lower and upper bounds for yv
    yv_min = 0.1 * xv  # min yv allowed
    yv = yv.where(yv > yv_min, yv_min)
    yv_max = xv  # signal cannot be greater than obs variance
    yv = yv.where(yv < yv_max, yv_max)
    # Set lower and upper bounds for cov
    cov_min = 0  # ignore negative covariances and corelations
    cov = cov.where(cov > cov_min, cov_min)
    cov_max = xv  # Max covariance equal to variance of obs
    cov = cov.where(cov < cov_max, cov_max)

    # ----------------------------------------------------------------------------------------------
    # Adjust ym to look more like the 30-year climo
    #
    if 'adjust_model_climo' in stats:
        ym += stats['climo_mean'] - xm

    # ----------------------------------------------------------------------------------------------
    # Calculate correlation of obs and ens mean
    #
    rxy = cov / np.sqrt(xv * yv)
    # Limit values of rxy
    rxy = rxy.where(rxy < rxy_max, rxy_max)

    # ----------------------------------------------------------------------------------------------
    # Correct stats, compensating for the difference between the number of members used to
    # correct the stats, and the number of members in the real time forecast
    #
    if ens_size_correction:
        es = es * (num_stats_members / num_fcst_members) * \
             (num_fcst_members - 1) / (num_stats_members - 1)
        yv_uncorrected = yv
        yv = yv - (num_fcst_members - num_stats_members)
        rxy = rxy * np.sqrt(yv / yv_uncorrected)

    # ----------------------------------------------------------------------------------------------
    # Calculate correlation of obs and best member
    #
    rbest = rxy * np.sqrt(1 + es / yv)

    # ----------------------------------------------------------------------------------------------
    # Calculate fcst anomalies and mean
    #
    y_anom = raw_fcst - ym
    y_anom_mean = y_anom.mean(dim='member')

    # ----------------------------------------------------------------------------------------------
    # Calculate correction for over-dispersive model
    #
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.sqrt(yv / es * (num_fcst_members - 1) / num_fcst_members * (1 / rxy ** 2 - 1))
    k = k.where(k <= 1, 1)  # keep kn <= 1
    k = k.where(rbest <= 1, 1)  # set to 1 if rbest not > 1

    # ----------------------------------------------------------------------------------------------
    # Adjust fcst members to correct for over-dispersion
    #
    y_anom = y_anom_mean + k * (y_anom - y_anom_mean)

    # ----------------------------------------------------------------------------------------------
    # Make adjustments for cases where correlations are low (in this case, use climo)
    #
    rxy = rxy.where(rxy >= rxy_min, rxy_min)

    # ----------------------------------------------------------------------------------------------
    # Calculate regression coefficient and error of best member
    #
    a1 = rxy * np.sqrt(xv / yv)
    ebest = np.sqrt(num_years / (num_years - 2) * xv * (1 - rxy**2 * (1 + k**2 * es/yv)))
    emean = np.sqrt(num_years / (num_years - 2) * xv * (1 - rxy**2))
    # Set lower bound for ebest - if ebest is zero, then the scale parameter in norm.cdf will be
    # zero, resulting in the cdf being NaN - set to a very small number instead of zero
    ebest = ebest.where(ebest > 0, 0.00001)

    # ----------------------------------------------------------------------------------------------
    # Correct each member
    #
    norm_ptiles = norm.ppf(np.array(ptiles) / 100)
    poe_ens = y_anom.expand_dims({'norm_ptile': norm_ptiles}).copy(deep=True) * np.nan

    for norm_ptile in norm_ptiles:
        poe_ens.loc[dict(norm_ptile=norm_ptile)] = xr.apply_ufunc(norm.cdf, norm_ptile,
                                                                  a1 * y_anom / np.sqrt(xv),
                                                                  ebest / np.sqrt(xv))
    poe_ens_mean = (
        poe_ens.mean(dim='member')                    # Create ensemble mean
        .rename('poe')                                # Give the DataArray a name of 'poe'
        .rename({'norm_ptile': 'ptile'})              # Rename norm_ptile coordinate to ptile
        .assign_coords(ptile=list(reversed(ptiles)))  # Insert real ptile values (99-1)
        .sortby('ptile', ascending=True)              # Sort ptiles as ascending
    )

    # ----------------------------------------------------------------------------------------------
    # Plot stats for debugging
    #
    if debug:
        levels_dict = {
            'tmean': {
                'a1': np.arange(0.1, 1.3, 0.1),
                'ebest': np.arange(0.5, 6.5, 0.5),
                'emean': np.arange(0.5, 7.5, 0.5),
                'k': np.arange(0.1, 1.1, 0.1),
                'rxy': np.arange(0.1, 1.1, 0.1),
                'es': np.arange(3, 28, 3),
                'yv': np.arange(5, 56, 5),
                'xv': np.arange(5, 56, 5),
                'rbest': np.arange(0, 1.3, 0.1),
                'y_anom_mean': 'auto',
                'y_anom_single': np.arange(-10, 16, 1),
            },
            'precip': {
                'a1': np.arange(0.1, 1.3, 0.1),
                'ebest': np.arange(0.5, 3, 0.2),
                'emean': np.arange(0.5, 3, 0.2),
                'k': np.arange(0.1, 1.1, 0.1),
                'rxy': np.arange(0.1, 1.1, 0.1),
                'es': np.arange(1, 11, 1),
                'yv': np.arange(1, 11, 1),
                'xv': np.arange(1, 11, 1),
                'rbest': np.arange(0, 1.3, 0.1),
                'y_anom_mean': 'auto',
                'y_anom_single': 'auto',
            }
        }

        for stat in ['a1', 'ebest', 'emean', 'k', 'rxy', 'es', 'yv', 'rbest', 'xv', 'y_anom_mean']:
            try:
                levels = levels_dict[variable][stat]
            except KeyError:
                levels = 'auto'
            locals()[stat].plot()
            plt.savefig(f'{stat}.png', dpi=200)
            plt.close()

    if debug_poe_latlon:
        plot_debug_poe(raw_fcst, POE_ens_mean, debug_poe_latlon, geogrid, climo=climo,
                       var_info={'name': variable})


    # ----------------------------------------------------------------------------------------------
    # Return total POE
    #
    return poe_ens_mean
