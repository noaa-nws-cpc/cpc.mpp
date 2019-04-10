import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from cpc.stats.stats import ptiles_to_full_fields, linear_interp_constant


def plot_debug_charts(raw_fcst, cal_fcst, latlon, geogrid=None,
                      ptiles=list([1, 2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 67, 75, 80, 85, 90,
                                   95, 98, 99]), climo=None):
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
    ax.set_title('TEST1')
    ax.plot(
        np.exp(
            np.flip(
                np.sort(raw_fcst_da.loc[dict(lat=plot_lat, lon=plot_lon)])
            ),
        ), raw_fcst_da.member / (len(raw_fcst_da.member) - 1)
    )
    plt.savefig('test1.png', dpi=200)

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_title('TEST2')
    ax.plot(climo_da.loc[dict(lat=plot_lat, lon=plot_lon)],
            cal_fcst_da.loc[dict(lat=plot_lat, lon=plot_lon)])
    plt.savefig('test2.png', dpi=200)

    print('asdf')


if __name__ == '__main__':
    import pickle

    with open('/export/cpc-lw-mcharles/mcharles/pycharm-deployments/mpp-driver/mpp_driver/data.pkl', 'rb') as file:
        raw_fcst, cal_fcst, debug_chart_latlon, geogrid, climo = pickle.load(file)
        debug_chart_latlon = tuple([float(val) for val in debug_chart_latlon])

    plot_debug_charts(raw_fcst, cal_fcst, debug_chart_latlon, geogrid=geogrid, climo=climo)
