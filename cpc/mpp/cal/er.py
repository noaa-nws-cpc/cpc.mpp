import numpy as np


class StatsError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def regress(raw_fcst, stats, method='ensemble', ens_size_correction=False,
            ptiles=list([1, 2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 67, 75, 80, 85, 90, 95, 98, 99])):
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
    with np.errstate(divide='ignore'):
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

