import numpy as np
from scipy import stats
import logging
import matplotlib.pyplot as plt


def _plot_correction(A_cdf, B_cdf, A_dist, B_dist, quantile_correction, C,
                     C_corrected, quantiles):
    # --------------------------------------------------------------------------
    # Plot correction in CDF space
    #
    plt.figure()
    # Figure settings
    linewidth = 2
    markeredgewidth = 2
    # Plot sample CDFs
    plt.plot(quantiles[1:], A_cdf, 'x', c='r', markersize=8,
             markeredgewidth=markeredgewidth, label='Fcst')
    plt.plot(quantiles[1:], B_cdf, '+', c='g', markersize=8,
             markeredgewidth=markeredgewidth, label='Obs')
    # Plot fitted CDFs
    plt.plot(quantiles[1:], A_dist.cdf(quantiles[1:]), '-', color='r',
             linewidth=linewidth, label='Gamma approx. of Fcst')
    plt.plot(quantiles[1:], B_dist.cdf(quantiles[1:]), '-', color='g',
             linewidth=linewidth, label='Gamma approx. of Obs')
    # Plot arrows indicating CDF correction
    for q in range(len(quantiles[1:])):
        if ~np.isinf(quantile_correction[q]) and quantile_correction[q] != 0:
            print('Correction at {}: {}'.format(quantiles[q],
                                                quantile_correction[q]))
            plt.arrow(
                quantiles[q + 1], A_dist.cdf(quantiles[q + 1]),
                quantile_correction[q], 0, length_includes_head=True,
                head_width=0.03, head_length=2, edgecolor='k', facecolor='k',
                overhang=1
            )
    # Add legend, title, and labels
    plt.legend(loc='lower right')
    plt.title('CDF Matching - Making Fcst Look Like Obs')
    plt.xlabel('Quantile Values (units of sample)')
    plt.ylabel('Probability Density')
    # plt.show()
    plt.savefig('stats1.png', dpi=200, bbox_inches='tight')
    # --------------------------------------------------------------------------
    # Plot correction in sample space
    #
    plt.figure()
    # Figure settings
    linewidth = 2
    markeredgewidth = 2
    # Plot C and C_corrected
    plt.plot(sorted(C), range(len(C)), 'r', linewidth=linewidth,
             label='Sample C')
    plt.plot(sorted(C_corrected), range(len(C)), 'g', linewidth=linewidth,
             label='Sample C (corrected)')
    # Add legend, title, and labels
    plt.legend(loc='lower right')
    plt.title('Sample C Before and After Correction')
    plt.xlabel('Sample value')
    plt.ylabel('Sample index')
    plt.savefig('stats2.png', dpi=200, bbox_inches='tight')


def cdf_correction(past_fcst_data, past_obs_data, curr_fcst_data, quantiles,
                   plot_point=None):
    # Make input NumPy arrays copies, so as not to modify the original arrays
    # in the calling function
    past_fcst_data = past_fcst_data.copy()
    past_obs_data = past_obs_data.copy()
    curr_fcst_data = curr_fcst_data.copy()
    # Initialize an array to store the corrected forecast data
    curr_fcst_data_corrected = np.empty(curr_fcst_data.shape) * np.nan
    # Flatten forecast array so all ensemble members and days are pooled
    # together. For example, for 45 days of past 1-deg forecasts, the array
    # shape would be (45 x 21 x 65160). We want to to flatten that to be
    # (945 x 65160).
    if past_fcst_data.ndim == 3:
        past_fcst_data = past_fcst_data.reshape(
            past_fcst_data.shape[0] * past_fcst_data.shape[1],
            past_fcst_data.shape[2])
    # Loop over all grid points
    for g in range(past_fcst_data.shape[1]):
        # ----------------------------------------------------------------------
        # QC data
        #
        qc_pcnt_data_req_at_gdpt = 90
        pcnt_data_found = 100 * np.count_nonzero(~np.isnan(past_obs_data[:, g])
                                                 ) / past_obs_data.shape[0]
        if pcnt_data_found < qc_pcnt_data_req_at_gdpt:
            continue
        pcnt_data_found = 100 * np.count_nonzero(~np.isnan(past_fcst_data[:, g])
                                                 ) / past_fcst_data.shape[0]
        if pcnt_data_found < qc_pcnt_data_req_at_gdpt:
            continue

        # ----------------------------------------------------------------------
        # Edge cases
        #
        # If obs are all zeros, set corrected forecast to zero
        if np.all(past_obs_data[:, g] == 0):
            curr_fcst_data_corrected[:, g] = 0
            continue
        # If obs are all the same, set the mean of the corrected forecasts to
        # that value
        if np.unique(past_obs_data[:, g]).size == 1:
            curr_fcst_data_corrected[:, g] = curr_fcst_data[:, g] + \
                (past_obs_data[0, g] - np.nanmean(past_fcst_data[:, g]))
            continue
        # If obs are all NaNs, set the corrected forecast to NaNs (can't
        # correct)
        if np.all(np.isnan(past_obs_data[:, g])):
            curr_fcst_data_corrected[:, g] = np.nan
            continue
        # If fcsts are all zero, set corrected forecast to zero
        if np.all(past_fcst_data[:, g] == 0):
            curr_fcst_data_corrected[:, g] = 0
            continue

        # ----------------------------------------------------------------------
        # Add a tiny amount so values are > 0
        #
        past_obs_data[:, g] = past_obs_data[:, g] + 0.001
        past_fcst_data[:, g] = past_fcst_data[:, g] + 0.001

        # ----------------------------------------------------------------------
        # Calculate PDFs and CDFs from the forecast and observation samples
        #
        past_fcst_pdf, edges = np.histogram(past_fcst_data[:, g], bins=quantiles)
        past_fcst_cdf = np.cumsum(past_fcst_pdf) / past_fcst_data.shape[0]
        past_obs_pdf, edges = np.histogram(past_obs_data[:, g], bins=quantiles)
        past_obs_cdf = np.cumsum(past_obs_pdf) / past_obs_data.shape[0]
        # Fit a gamma distribution
        try:
            past_fcst_dist = stats.gamma(
                *stats.gamma.fit(
                    past_fcst_data[np.nonzero(past_fcst_data[:, g]), g], floc=0
                )
            )
            past_obs_dist = stats.gamma(
                *stats.gamma.fit(
                    past_obs_data[np.nonzero(past_obs_data[:, g]), g], floc=0
                )
            )
        # If a ValueError is raised during curve-fitting, set the corrected
        # fcst data to np.nan
        except ValueError:
            curr_fcst_data_corrected[:, g] = np.nan
            continue
        # Calculate corrected quantiles
        corrected_quantiles = past_obs_dist.ppf(past_fcst_dist.cdf(quantiles[1:]))
        quantile_correction = corrected_quantiles - quantiles[1:]
        curr_fcst_data_corrected[:, g] = past_obs_dist.ppf(past_fcst_dist.cdf(
            curr_fcst_data[:, g]))
        # ----------------------------------------------------------------------
        # Plot
        #
        if plot_point and g == plot_point:
            try:
                _plot_correction(
                    past_fcst_cdf, past_obs_cdf, past_fcst_dist, past_obs_dist,
                    quantile_correction, curr_fcst_data[:, g],
                    curr_fcst_data_corrected[:, g], quantiles
                )
            except:
                pass

    # --------------------------------------------------------------------------
    # Fix any remaining issues with the corrected forecast
    #
    # Set all negative values to 0
    with np.errstate(invalid='ignore'):
        curr_fcst_data_corrected = np.where(curr_fcst_data_corrected < 0, 0,
                                        curr_fcst_data_corrected)

    return curr_fcst_data_corrected


def bias_correction(past_fcst_data, past_obs_data, curr_fcst_data, weights=None):

    # Calculate past ensemble mean fcst
    past_fcst_data_mean = np.nanmean(past_fcst_data, axis=1)

    # Calculate past bias
    bias = np.average(past_fcst_data_mean - past_obs_data, axis=0, weights=weights)

    # Correct current fcst, given the past bias
    corrected_curr_fcst_data = curr_fcst_data - bias

    return corrected_curr_fcst_data
