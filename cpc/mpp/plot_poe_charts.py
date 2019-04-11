import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import logging


logger = logging.getLogger(__name__)


def plot_debug_poe(raw_fcst, cal_fcst, latlon, geogrid=None, ptiles=None, climo=None,
                   var_info=None):
    logger.debug(f"Creating a POE ")
    if var_info is None:
        raise ValueError('Must include var_info parameter, see API docs')
    if ptiles is None:
        ptiles = [1, 2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 67, 75, 80, 85, 90, 95, 98, 99]
    num_members = raw_fcst.shape[0]
    num_lats = len(geogrid.lats)
    num_lons = len(geogrid.lons)
    plot_lat = latlon[0]
    plot_lon = latlon[1] if latlon[1] >= 0 else 360 + latlon[1]

    raw_fcst_da = xr.DataArray(raw_fcst.reshape(num_members, num_lats, num_lons),
                               dims=['member', 'lat', 'lon'],
                               coords={'member': list(range(51)), 'lat': geogrid.lats,
                                       'lon': geogrid.lons})
    cal_fcst_da = xr.DataArray(cal_fcst.reshape(len(ptiles), num_lats, num_lons),
                               dims=['ptile', 'lat', 'lon'],
                               coords={'ptile': ptiles, 'lat': geogrid.lats, 'lon': geogrid.lons})

    climo_da = xr.DataArray(climo.reshape(len(ptiles), num_lats, num_lons),
                               dims=['ptile', 'lat', 'lon'],
                               coords={'ptile': ptiles, 'lat': geogrid.lats, 'lon': geogrid.lons})

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_title(f"{var_info['name'].capitalize()} POE Forecast at ({plot_lat}N, {plot_lon}E)")
    ax.plot(np.exp(np.flip(np.sort(raw_fcst_da.loc[dict(lat=plot_lat, lon=plot_lon)])),),
            raw_fcst_da.member / (len(raw_fcst_da.member) - 1),
            label='Raw fcst')
    ax.set_xlabel(f"{var_info['name']}")
    ax.plot(climo_da.loc[dict(lat=plot_lat, lon=plot_lon)],
            cal_fcst_da.loc[dict(lat=plot_lat, lon=plot_lon)],
            label='Calibrated fcst')
    ax.set_ylabel('Forecast Probability')
    ax.set_ylim([0, 1])
    legend = ax.legend(loc='upper right')
    for i in range(len(climo_da.loc[dict(lat=plot_lat, lon=plot_lon)])):
        x = float(climo_da.loc[dict(lat=plot_lat, lon=plot_lon)][i])
        p = int(climo_da.loc[dict(lat=plot_lat, lon=plot_lon)].ptile[i])
        if p == 50:
            linewidth = 1
            linestyle = '-'
        else:
            linewidth = 0.3
            linestyle = '--'
        plt.axvline(x, linewidth=linewidth, linestyle=linestyle, color='red')
        ax.annotate(str(p), (x, -0.005), horizontalalignment='center', verticalalignment='top',
                    fontsize=6, fontweight='light', color='red', annotation_clip=False)

    # plt.tight_layout()
    plt.savefig('cal-poe.png', dpi=300)


if __name__ == '__main__':
    import pickle

    with open('/export/cpc-lw-mcharles/mcharles/pycharm-deployments/mpp-driver/mpp_driver/data.pkl', 'rb') as file:
        raw_fcst, cal_fcst, debug_poe_latlon, geogrid, climo = pickle.load(file)
        debug_poe_latlon = tuple([float(val) for val in debug_poe_latlon])

    plot_debug_poe(raw_fcst, cal_fcst, debug_poe_latlon, geogrid=geogrid, climo=climo,
                   var_info={'name': 'precip'})
