import numpy as np
from scipy.stats import norm


class StatsError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def ensemble_regression(raw_fcst, stats, method='ensemble', ens_size_correction=False,
            ptiles=list([1, 2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 67, 75, 80, 85, 90, 95, 98, 99]),
            debug=False, variable='tmean'):
    # ----------------------------------------------------------------------------------------------
    # Check stats dict for all required stats
    #
    required_stats = ['cov', 'es', 'xm', 'xv', 'ym', 'yv', 'num_stats_members',
                      'num_fcst_members', 'num_years']
    for required_stat in required_stats:
        if required_stat not in stats.keys():
            raise StatsError('Stat {} not found in stats. Required stats are: {}'.format(
                required_stat, required_stats))

    # ----------------------------------------------------------------------------------------------
    # Set some config options
    #
    xv_min = 0.1  # min xv allowed
    rxy_min = 0.05  # min rxy allowed

    # ----------------------------------------------------------------------------------------------
    # Extract fields from stats dict
    #
    cov = stats['cov']
    es = stats['es']
    xm = stats['xm']
    xv = stats['xv']
    ym = stats['ym']
    yv = stats['yv']
    num_stats_members = stats['num_stats_members']
    num_fcst_members = stats['num_fcst_members']
    num_years = stats['num_years']

    # ----------------------------------------------------------------------------------------------
    # Set bounds for some stats
    #
    # Set lower bound for xv
    xv = np.where(xv < xv_min, xv_min, xv)
    # Set lower and upper bounds for yv
    yv_min = 0.1 * xv  # min yv allowed
    yv = np.where(yv < yv_min, yv_min, yv)
    yv_max = xv  # signal cannot be greater than obs variance
    yv = np.where(yv > yv_max, yv_max, yv)
    # Set lower and upper bounds for cov
    cov_min = 0  # ignore negative covariances and corelations
    cov = np.where(cov < cov_min, cov_min, cov)
    cov_max = xv  # Max covariance equal to variance of obs
    cov = np.where(cov > cov_max, cov_max, cov)

    # ----------------------------------------------------------------------------------------------
    # Calculate correlation of obs and ens mean
    #
    rxy = cov / np.sqrt(xv * yv)

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
    y_anom_mean = np.nanmean(y_anom, axis=0)

    # ----------------------------------------------------------------------------------------------
    # Calculate correction for over-dispersive model
    #
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.sqrt(yv / es * (num_fcst_members - 1) / num_fcst_members * (1 / rxy ** 2 - 1))
    k = np.where(k > 1, 1, k)  # keep kn <= 1
    k = np.where(rbest > 1, k, 1)  # set to 1 if rbest not > 1

    # ----------------------------------------------------------------------------------------------
    # Adjust fcst members to correct for over-dispersion
    #
    y_anom = y_anom_mean + k * (y_anom - y_anom_mean)

    # ----------------------------------------------------------------------------------------------
    # Make adjustments for cases where correlations are low (in this case, use climo)
    #
    rxy = np.where(rxy < rxy_min, 0, rxy)

    # ----------------------------------------------------------------------------------------------
    # Calculate regression coefficient and error of best member
    #
    a1 = rxy * np.sqrt(xv / yv)
    ebest = np.sqrt(num_years / (num_years - 2) * xv * (1 - rxy**2 * (1 + k**2 * es/yv)))
    emean = np.sqrt(num_years / (num_years - 2) * xv * (1 - rxy**2))
    # Set lower bound for ebest - if ebest is zero, then the scale parameter in norm.cdf will be
    # zero, resulting in the cdf being NaN - set to a very small number instead of zero
    ebest = np.where(ebest == 0, 0.00001, ebest)

    # ----------------------------------------------------------------------------------------------
    # Correct each member
    #
    norm_ptiles = norm.ppf(np.array(ptiles) / 100)
    POE_ens_mean = np.full((len(norm_ptiles), raw_fcst.shape[1]), np.nan)
    POE_ens = np.full((len(norm_ptiles), raw_fcst.shape[0], raw_fcst.shape[1]), np.nan)
    for p, norm_ptile in enumerate(norm_ptiles):
        for m in range(raw_fcst.shape[0]):
            with np.errstate(divide='ignore', invalid='ignore'):
                POE_ens[p, m] = 1 - norm.cdf(norm_ptile, a1 * y_anom[m] / np.sqrt(xv),
                                             ebest / np.sqrt(xv))
    POE_ens_mean = np.nanmean(POE_ens, axis=1)

    # ----------------------------------------------------------------------------------------------
    # Plot stats for debugging
    #
    if debug:
        from cpc.geogrids import Geogrid
        from cpc.geoplot import Geomap, Geofield

        geogrid = Geogrid('1deg-global')

        levels = {
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

        for stat in ['a1', 'ebest', 'emean', 'k', 'rxy', 'es', 'yv', 'rbest', 'xv']:
            geomap = Geomap()
            geofield = Geofield(locals()[stat], geogrid, levels=levels[variable][stat])
            geomap.plot(geofield)
            geomap.save('{}.png'.format(stat), dpi=200)

    # ----------------------------------------------------------------------------------------------
    # Return total POE
    #
    return POE_ens_mean
