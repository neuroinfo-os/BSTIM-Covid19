# -*- coding: utf-8 -*-
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np
import pandas as pd
import isoweek
import pickle as pkl
import datetime
import time
import numba
import os
from numba import njit
from collections import OrderedDict
from matplotlib import pyplot as pp
# BUG: may throw an error for flat RVs
theano.config.compute_test_value = 'off'


def uniform_times_by_week(weeks, n=500):
    """ Samples n random timepoints within a week, per week.
        converts times to datetime obj."""
    res = OrderedDict()
    for week in weeks:
        time_min = datetime.datetime.combine(
            isoweek.Week(*week).monday(), datetime.time.min)
        time_max = datetime.datetime.combine(
            isoweek.Week(*week).sunday(), datetime.time.max)
        res[week] = np.random.rand(n) * (time_max - time_min) + time_min
    return res


def uniform_times_by_day(days, rnd_tsel, n=10):
    """ Samples n random timepoints within a day, per day.
        converts pd.Timestamps to datetime obj."""
    res = OrderedDict()
    for day in days:
        time_min = datetime.datetime.combine(day, datetime.time.min)
        time_max = datetime.datetime.combine(day, datetime.time.max)
        res[day] = rnd_tsel.random(n) * (time_max - time_min) + time_min
    return res


def uniform_locations_by_county(counties, rnd_csel, n=5):
    res = OrderedDict()
    for (county_id, county) in counties.items():
        tp = county["testpoints"]
        if n == len(tp):
            res[county_id] = tp
        else:
            idx = rnd_csel.choice(tp.shape[0], n, replace=n > len(tp))
            res[county_id] = tp[idx]
    return res


def sample_time_and_space__once(times_by_day, locations_by_county):
    """
    Convert dictonarys to arrays for faster access in sample_time_and_space().

    Random access in times_by_day and locations_by_county are very costy.
    Hence they need to be converted to arrays and access must be done through
    indexes.
    """
    times_by_day_np = \
    pd.DataFrame.from_dict(times_by_day,orient='index').to_numpy(dtype='datetime64')

    t_convert_1 = np.frompyfunc(pd.Timestamp, 1, 1)
    times_by_day_np = t_convert_1(times_by_day_np)

    t_convert_2 = np.frompyfunc(datetime.datetime.timestamp, 1, 1)
    times_by_day_np = t_convert_2(times_by_day_np)
    times_by_day_np = np.array(times_by_day_np, np.float64)

    max_coords = 0
    for item in locations_by_county.items():
        max_coords = max( len(item[1]), max_coords)

    locations_by_county_np = \
    np.empty([len(locations_by_county.keys()), max_coords, 2], dtype='float64')
    for i,item in enumerate(locations_by_county.items()):
        locations_by_county_np[i][:] = item[1][:]

    return(times_by_day_np, locations_by_county_np)


def sample_time_and_space__prep(times_by_day,
                                locations_by_county,
                                times_by_day_np,
                                locations_by_county_np,
                                data,
                                idx):
    """
    Recalculations for a fixed dataframe sample_time_and_space().

    Calculation of helper arrays are very costy.
    If the dataframe does not change, precalculated values
    can be reused.
    """

    # subdata 'data' of 'indata' is likely to skip a few first days(rows)
    # in 'indata',
    # but as times_by_day_np represents the whole 'indata', an offsets
    # needs to be considered when accessing 'times_by_day_np'
    dayoffset = np.where(idx==True)[0][0]
    n_total = data.sum().sum()

    # get number of samples per county-day
    smpls_per_cntyday = np.array(data.values).flatten('F')

    ######## t_all ########

    # get list of day-ids for all county-days
    dayids = np.arange(len(data.index))
    day_of_cntyday = np.tile(dayids, len(data.columns))

    # get list of day-ids for all samples
    day_of_smpl = np.array([ day_of_cntyday[i] \
        for (i,smpls) in enumerate(smpls_per_cntyday) \
        for x in range(smpls) ])

    # get available times for each sample
    time_of_days = data.index.tolist() # needs to stay pandas format
    av_times_per_day = [len(times_by_day[d]) for d in time_of_days]
    av_times_per_smpl = [ av_times_per_day[day_of_cntyday[i]] \
        for (i,smpls) in enumerate(smpls_per_cntyday) \
        for x in range(smpls) ]

    ######## x_all ########

    # get list of county-ids for all county-days
    cntyids = np.arange(len(data.columns))
    cnty_of_cntyday = np.repeat(cntyids, len(data.index))

    # get list of county-ids for all samples
    cnty_of_smpl = np.array([ cnty_of_cntyday[i] \
        for (i,smpl) in enumerate(smpls_per_cntyday) \
        for x in range(smpl) ])

    # get available locations for each sample
    label_of_cntys = data.columns # list of countys labels
    av_locs_per_cnty = \
        [len(locations_by_county[c]) for c in label_of_cntys]

    av_locs_per_smpl = \
        [ av_locs_per_cnty[cnty_of_cntyday[i]] \
        for (i,smpls) in enumerate(smpls_per_cntyday) \
        for x in range(smpls) ]

    return (n_total, dayoffset,
            day_of_smpl, av_times_per_smpl,
            cnty_of_smpl, av_locs_per_smpl)


def sample_time_and_space__pred(n_days,
                                n_counties,
                                times_by_day_np,
                                locations_by_county_np,
                                d_offs,
                                c_offs,
                                num_tps,
                                av_times_per_smpl,
                                av_locs_per_smpl,
                                rnd_time,
                                rnd_loc):

    ######## t_all ########
    n_total = n_days * n_counties * num_tps

    rnd_timeid_per_smpl = np.floor(av_times_per_smpl * rnd_time.random(n_total)
                                   ).astype("int32")

    # collect times for each sample with its random time-id
    t_all = \
        [times_by_day_np[d_offs+i][rnd_timeid_per_smpl[ \
        (i*n_counties+j)*num_tps+x]] \
        for i in range(n_days) \
        for j in range(n_counties) \
        for x in range(num_tps) ]

    ######## x_all ########

    # calc random location-id for each sample
    rnd_locid_per_smpl = \
        np.floor( av_locs_per_smpl * \
        rnd_loc.random((n_total,)) ).astype("int32")

    # collect locations for each sample with its random loc-id
    x_all = \
        [locations_by_county_np[c_offs+j] \
        [rnd_locid_per_smpl[(i*n_counties+j)*num_tps+x]] \
        for i in range(n_days)
        for j in range(n_counties)
        for x in range(num_tps)]

    return t_all, x_all


@numba.jit(nopython=True, parallel=True, cache=False)
def sample_time_and_space_tx(n_total, n_counties, dayoffset,
                             day_of_smpl, rnd_timeid_per_smpl_all, times_by_day_np,
                             cnty_of_smpl, rnd_locid_per_smpl_all, locations_by_county_np):
    t_all_np = np.empty(day_of_smpl.size * n_counties, dtype=np.float64)
    for j in numba.prange(n_counties):
        for (i,day) in enumerate(day_of_smpl):
            t_all_np[day_of_smpl.size*j+i] = \
                times_by_day_np[day+dayoffset][rnd_timeid_per_smpl_all[j*n_total+i]]

    x_all_np = np.empty((cnty_of_smpl.size * n_counties, 2), dtype=np.float64)
    for j in numba.prange(n_counties):
        for (i,cnty) in enumerate(cnty_of_smpl):
            x_all_np[cnty_of_smpl.size*j+i] = \
                locations_by_county_np[cnty][rnd_locid_per_smpl_all[j*n_total+i]]

    return t_all_np, x_all_np

def sample_time_and_space(n_counties,
                          n_total,
                          dayoffset,
                          times_by_day_np,
                          locations_by_county_np,
                          day_of_smpl,
                          av_times_per_smpl,
                          cnty_of_smpl,
                          av_locs_per_smpl,
                          rnd_time,
                          rnd_loc):
    """
    Calculations samples in time and space.
    Calculation a hughe random number array use
    precalulated results to pick samples.
    """
    ######## t_all ########
    if n_total == 0:
        return np.empty((0,), dtype=np.float64), \
            np.empty((0, 2), dtype=np.float64)

    # calc random time-id for each sample
    n_all = n_total * n_counties

    av_times_per_smpl_all = np.tile(av_times_per_smpl, n_counties)

    rnd_timeid_per_smpl_all = \
        np.floor( av_times_per_smpl_all * rnd_time.random( (n_all,) ) ).astype("int32")

    ######## x_all ########

    # calc random location-id for each sample
    av_locs_per_smpl_all = np.tile(av_locs_per_smpl, n_counties)
    rnd_locid_per_smpl_all = \
        np.floor( av_locs_per_smpl_all * rnd_loc.random( (n_all,) ) ).astype("int32")

    t_all, x_all = sample_time_and_space_tx(n_total, n_counties,
                                dayoffset, day_of_smpl,
                                rnd_timeid_per_smpl_all, times_by_day_np,
                                cnty_of_smpl, rnd_locid_per_smpl_all,
                                locations_by_county_np)

    return t_all, x_all


def gaussian_bf(dx, σ):
    """ spatial basis function """
    σ = np.float32(σ)
    res = tt.zeros_like(dx)
    idx = (abs(dx) < np.float32(5) * σ)  # .nonzero()
    return tt.set_subtensor(res[idx], tt.exp(
        np.float32(-0.5 / (σ**2)) * (dx[idx])**2) / \
        np.float32(np.sqrt(2 * np.pi * σ**2)))


def bspline_bfs(x, knots, P):
    """ temporal basis function
            x: t-delta distance to last knot (horizon 5)
    """
    knots = knots.astype(np.float32)
    idx = ((x >= knots[0]) & (x < knots[-1]))  # .nonzero()
    xx = x[idx]

    N = {}
    for p in range(P + 1):
        for i in range(len(knots) - 1 - p):
            if p == 0:
                N[(i, p)] = tt.where((knots[i] <= xx)
                                     * (xx < knots[i + 1]), 1.0, 0.0)
            else:
                N[(i, p)] = \
                    (xx - knots[i]) / (knots[i + p] - knots[i]) * N[(i, p - 1)] + \
                    (knots[i + p + 1] - xx) / (knots[i + p + 1] - knots[i + 1]) * \
                    N[(i + 1, p - 1)]

    highest_level = []
    for i in range(len(knots) - 1 - P):
        res = tt.zeros_like(x)
        highest_level.append(tt.set_subtensor(res[idx], N[(i, P)]))
    return highest_level


def gaussian_gram(σ):
    return np.array([[np.power(2 * np.pi * (a**2 + b**2), -0.5)
                      for b in σ] for a in σ])


def temporal_bfs(x):
    return bspline_bfs(x, np.array([0, 0, 1, 2, 3, 4, 5]) * 24 * 3600.0, 2)


def spatial_bfs(x):
    return [gaussian_bf(x, σ) for σ in [6.25, 12.5, 25.0, 50.0]]


def jacobian_sq(latitude, R=6365.902):
    """
        jacobian_sq(latitude)

    Computes the "square root" (Cholesky factor) of the Jacobian of the cartesian
    projection from polar coordinates (in degrees longitude, latitude) onto
    cartesian coordinates (in km east/west, north/south) at a given latitude
    (the projection's Jacobian is invariante wrt. longitude).
    TODO: don't import jacobian_sq from geo_utils to remove potential conflicts
    """
    return R * (np.pi / 180.0) * (abs(tt.cos(tt.deg2rad(latitude))) *
                                  np.array([[1.0, 0.0], [0.0, 0.0]]) +
                                  np.array([[0.0, 0.0], [0.0, 1.0]]))


def build_ia_bfs(temporal_bfs, spatial_bfs, profile=False):
    x1 = tt.fmatrix("x1")
    t1 = tt.fvector("t1")
    # M = tt.fmatrix("M")
    x2 = tt.fmatrix("x2")
    t2 = tt.fvector("t2")

    lat = x1[:, 1].mean()
    M = jacobian_sq(lat)**2

    # (x1,t1) are the to-be-predicted points, (x2,t2) the historic cases

    # spatial distance btw. each points (defined with latitude,longitude) in x1 and x2 with gramian M
    # (a-b)^2 = a^2 + b^2 -2ab; with a,b=vectors
    dx = tt.sqrt(  (x1.dot(M) * x1).sum(axis=1).reshape((-1,  1)) # a^2
                 + (x2.dot(M) * x2).sum(axis=1).reshape(( 1, -1)) # b^2
                 - 2 * x1.dot(M).dot(x2.T) )                      # -2ab

    # temporal distance btw. each times in t1 and t2
    dt = t1.reshape((-1, 1)) - t2.reshape((1, -1))

    ft = tt.stack(temporal_bfs(dt.reshape((-1,))), axis=0) # cast to floats?
    fx = tt.stack(spatial_bfs(dx.reshape((-1,))), axis=0)

    # aggregate contributions of all cases
    contrib = ft.dot(fx.T).reshape((-1,)) / tt.cast(x1.shape[0], "float32")

    return theano.function( [t1, x1, t2, x2], contrib,
                            allow_input_downcast=True, profile=profile)


def iaeffect_sampler(   data, times_by_day, locations_by_county,
                        temporal_bfs, spatial_bfs, num_tps=5, time_horizon=5):
    rnd_time = np.random.Generator(np.random.PCG64())
    rnd_loc  = np.random.Generator(np.random.PCG64())
    rnd_time_pred = np.random.Generator(np.random.PCG64())
    rnd_loc_pred  = np.random.Generator(np.random.PCG64())

    number_of_threads = int(os.environ['OMP_NUM_THREADS'])
    numba.set_num_threads(number_of_threads)

    # Convert dictonarys to arrays for faster access in sample_time_and_space().
    (times_by_day_np, locations_by_county_np,) = \
        sample_time_and_space__once(times_by_day, locations_by_county)

    ia_bfs = build_ia_bfs(temporal_bfs, spatial_bfs)

    t_data_1 = []
    x_data_1 = []
    t_pred_1 = []
    x_pred_1 = []

    days = data.index
    counties = data.columns
    d_offs=0 # just to limit the time of test
    c_offs=0 # just to limit the time of test
    #days = data.index[d_offs:d_offs+50]
    #counties = data.columns[c_offs:c_offs+50]

    num_features = len(temporal_bfs(tt.fmatrix("tmp"))) * \
        len(spatial_bfs(tt.fmatrix("tmp")))
    res_1 = np.zeros((len(days), len(counties), num_features), dtype=np.float32)

    n_days = len(days)
    n_counties = len(counties)

    # create dataframe with 'num_tps' in each cell
    pred_data = pd.DataFrame(num_tps, index=days, columns=counties)
    idx = np.empty([len(data.index)], dtype='bool')
    idx.fill(True)

    # precalculate pediction values
    (n_total, dayoffset, day_of_smpl, av_times_per_smpl,
    cnty_of_smpl, av_locs_per_smpl,) = \
        sample_time_and_space__prep(times_by_day, locations_by_county,
                                    times_by_day_np, locations_by_county_np,
                                    pred_data, idx)
    (t_pred_all, x_pred_all,) = \
        sample_time_and_space__pred(n_days, n_counties, times_by_day_np,
                                    locations_by_county_np, d_offs, c_offs,
                                    num_tps, av_times_per_smpl, av_locs_per_smpl,
                                    rnd_time_pred, rnd_loc_pred)



    for i, day in enumerate(days):

        # calc which sub-table will be selected
        idx = ((day - pd.Timedelta(days=5)) <= data.index) * (data.index < day)
        subdata = data.iloc[idx]

        if subdata.size != 0:
            # Recalculations for a fixed dataframe sample_time_and_space().
            (n_total, dayoffset, day_of_smpl,
            av_times_per_smpl, cnty_of_smpl, av_locs_per_smpl,) = \
                sample_time_and_space__prep(times_by_day, locations_by_county,
                                            times_by_day_np, locations_by_county_np,
                                            subdata, idx)

            t_data_all, x_data_all = sample_time_and_space(n_counties,
                                                            n_total,
                                                            dayoffset,
                                                            times_by_day_np,
                                                            locations_by_county_np,
                                                            day_of_smpl,
                                                            av_times_per_smpl,
                                                            cnty_of_smpl,
                                                            av_locs_per_smpl,
                                                            rnd_time,
                                                            rnd_loc)

            for j, county in enumerate(counties):

                # calcs only for the single DataFrame.cell[day][county]
                offs = (i*n_counties+j)*num_tps
                t_pred = t_pred_all[offs:offs+num_tps]
                x_pred = x_pred_all[offs:offs+num_tps]

                # get subarray for county==j
                t_data = t_data_all[j*n_total:(j+1)*n_total] # [county][smpl]
                x_data = x_data_all[j*n_total:(j+1)*n_total] # [county][smpl]

                # use theano.function for day==i and county==j
                res_1[i, j, :] = ia_bfs(t_pred, x_pred, t_data, x_data)
    return res_1
