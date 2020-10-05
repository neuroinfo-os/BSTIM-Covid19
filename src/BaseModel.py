# from sampling_utils import *
from collections import OrderedDict
import theano
import re
import pandas as pd
import datetime
import numpy as np
import scipy as sp
import pymc3 as pm
import patsy as pt
import theano.tensor as tt
# BUG: may throw an error for flat RVs
theano.config.compute_test_value = 'off'


class SpatioTemporalFeature(object):
    def __init__(self):
        self._call_ = np.frompyfunc(self.call, 2, 1)

    def __call__(self, times, locations):
        _times = [pd.Timestamp(d) for d in times]
        return self._call_(np.asarray(_times).reshape(
            (-1, 1)), np.asarray(locations).reshape((1, -1))).astype(np.float32)


class SpatioTemporalYearlyDemographicsFeature(SpatioTemporalFeature):
    """ TODO:
    * county data must be updated to include 2019/2020 demographic data
      |> fix call
    """

    def __init__(self, county_dict, group, scale=1.0):
        self.dict = {
            (year, county): val * scale
            for county, values in county_dict.items()
            for (g, year), val in values["demographics"].items()
            if g == group
        }
        super().__init__()

    def call(self, yearweekday, county):
        # TODO: do this properly when data is available!
        return self.dict.get((2018, county))
        # return self.dict.get((yearweekday.year,county))


class SpatialEastWestFeature(SpatioTemporalFeature):
    def __init__(self, county_dict):
        self.dict = {
            county: 1.0 if "east" in values["region"] else (
                0.5 if "berlin" in values["region"] else 0.0) for county,
            values in county_dict.items()}
        super().__init__()

    def call(self, yearweekday, county):
        return self.dict.get(county)


class TemporalFourierFeature(SpatioTemporalFeature):
    def __init__(self, i, t0, scale):
        self.t0 = t0
        self.scale = scale
        self.τ = (i // 2 + 1) * 2 * np.pi
        self.fun = np.sin if (i % 2) == 0 else np.cos
        super().__init__()

    def call(self, t, x):
        return self.fun((t - self.t0) / self.scale * self.τ)


class TemporalPeriodicPolynomialFeature(SpatioTemporalFeature):
    def __init__(self, t0, period, order):
        self.t0 = t0
        self.period = period
        self.order = order
        super().__init__()

    def call(self, t, x):
        tdelta = (t - self.t0).days % self.period
        return (tdelta / self.period) ** self.order


class TemporalSigmoidFeature(SpatioTemporalFeature):
    def __init__(self, t0, scale):
        self.t0 = t0
        self.scale = scale
        super().__init__()

    def call(self, t, x):
        t_delta = (t - self.t0) / self.scale
        return sp.special.expit(t_delta.days + (t_delta.seconds / (3600 * 24)))


class TemporalPolynomialFeature(SpatioTemporalFeature):
    def __init__(self, t0, tmax, order):
        self.t0 = t0
        self.order = order
        self.scale = (tmax - t0).days
        super().__init__()

    def call(self, t, x):
        t_delta = (t - self.t0).days / self.scale
        return t_delta ** self.order


class ReportDelayPolynomialFeature(SpatioTemporalFeature):
    def __init__(self, t0, t_max, order):
        self.t0 = t0
        self.order = order
        self.scale = (t_max - t0).days
        super().__init__()

    def call(self, t, x):
        _t = 0 if t <= self.t0 else (t - self.t0).days / self.scale
        return _t ** self.order

class BaseModel(object):
    """
    Model for disease prediction.

    The model has 4 types of features (predictor variables):
    * temporal (functions of time)
    * spatial (functions of space, i.e. longitude, latitude)
    * county_specific (functions of time and space, i.e. longitude, latitude)
    """

    def __init__(
            self,
            trange,
            counties,
            model=None,
            include_report_delay=True,
            report_delay_order=4,
            include_demographics=True,
            include_temporal=True,
            trend_poly_order=4,
            include_periodic=True,
            periodic_poly_order=4,
            orthogonalize=False):

        self.county_info = counties
        self.include_report_delay = include_report_delay
        self.report_delay_order = report_delay_order
        self.include_demographics = include_demographics
        self.include_temporal = include_temporal
        self.trend_poly_order = trend_poly_order
        self.include_periodic = include_periodic
        self.periodic_poly_order = periodic_poly_order
        self.trange = trange  # 0 -> 28th of Jan; 1-> Last

        self.features = {
            "temporal_trend": {
                "temporal_polynomial_{}".format(i): TemporalPolynomialFeature(
                    trange[0], trange[1], i)
                for i in range(self.trend_poly_order+1)} if self.include_temporal else {},
            "temporal_seasonal": {
                "temporal_periodic_polynomial_{}".format(i): TemporalPeriodicPolynomialFeature(
                    trange[0], 7, i)
                for i in range(self.periodic_poly_order+1)} if self.include_periodic else {},
            "spatiotemporal": {
                "demographic_{}".format(group): SpatioTemporalYearlyDemographicsFeature(
                    self.county_info,
                    group) for group in [
                        "[0-5)",
                        "[5-20)",
                        "[20-65)"]} if self.include_demographics else {},
            "temporal_report_delay": {
                "report_delay": ReportDelayPolynomialFeature(
                    trange[1] - pd.Timedelta(days=5), trange[1], self.report_delay_order)}
                if self.include_report_delay else {}, # what is going in here?
            "exposure": {
                "exposure": SpatioTemporalYearlyDemographicsFeature(
                    self.county_info,
                    "total",
                    1.0 / 100000)}}

    def evaluate_features(self, days, counties):
        all_features = {}
        for group_name, features in self.features.items():
            group_features = {}
            for feature_name, feature in features.items():
                feature_matrix = feature(days, counties)
                group_features[feature_name] = pd.DataFrame(
                    feature_matrix[:, :], index=days, columns=counties).stack()
            all_features[group_name] = pd.DataFrame([], index=pd.MultiIndex.from_product(
                [days, counties]), columns=[]) if len(group_features) == 0 else pd.DataFrame(group_features)
        return all_features

    def init_model(self, target):
        days, counties = target.index, target.columns

        # extract features
        features = self.evaluate_features(days, counties)
        Y_obs = target.stack().values.astype(np.float32)
        T_S = features["temporal_seasonal"].values.astype(np.float32)
        T_T = features["temporal_trend"].values.astype(np.float32)
        T_D = features["temporal_report_delay"].values.astype(np.float32)
        TS = features["spatiotemporal"].values.astype(np.float32)

        log_exposure = np.log(
            features["exposure"].values.astype(np.float32).ravel())

        # extract dimensions
        num_obs = np.prod(target.shape)
        num_t_s = T_S.shape[1]
        num_t_t = T_T.shape[1]
        num_t_d = T_D.shape[1]
        num_ts = TS.shape[1]
        num_counties = len(counties)
        
        with pm.Model() as self.model:
            # priors
            # NOTE: Vary parameters over time -> W_ia dependent on time
            # δ = 1/√α
            δ = pm.HalfCauchy("δ", 10, testval=1.0)
            α = pm.Deterministic("α", np.float32(1.0) / δ)

            W_t_s = pm.Normal("W_t_s", mu=0, sd=10,
                              testval=np.zeros(num_t_s), shape=num_t_s)
            W_t_t = pm.Normal("W_t_t", mu=0, sd=10,
                              testval=np.zeros((num_counties, num_t_t)), shape=(num_counties, num_t_t))

            W_t_d = pm.Normal("W_t_d", mu=0, sd=10,
                              testval=np.zeros(num_t_d), shape=num_t_d)
            W_ts = pm.Normal("W_ts", mu=0, sd=10,
                             testval=np.zeros(num_ts), shape=num_ts)

            self.param_names = ["δ", "W_t_s", "W_t_t", "W_t_d", "W_ts"]
            self.params = [δ, W_t_s, W_t_t, W_t_d, W_ts]

            expanded_Wtt = tt.tile(W_t_t.reshape(shape=(1,num_counties,-1)), reps=(21, 1, 1))
            expanded_TT = np.reshape(T_T, newshape=(21,412,2))
            result_TT = tt.flatten(tt.sum(expanded_TT*expanded_Wtt,axis=-1))

            # calculate mean rates
            μ = pm.Deterministic(
                "μ",
                tt.exp(
                    tt.dot(T_S, W_t_s) +
                    result_TT + 
                    tt.dot(T_D, W_t_d) +
                    tt.dot(TS, W_ts)+
                    log_exposure
                    )
                  )
            # constrain to observations
            pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y_obs)

    def sample_parameters(
            self,
            target,
            n_init=100,
            samples=1000,
            chains=None,
            cores=8,
            init="advi",
            target_accept=0.8,
            max_treedepth=10,
            **kwargs):
        """
            sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

        Samples from the posterior parameter distribution, given a training dataset.
        The basis functions are designed to be causal, i.e. only data points strictly
        predating the predicted time points are used (this implies "one-step-ahead"-predictions).
        """

        self.init_model(target)

        if chains is None:
            chains = max(2, cores)

        with self.model:
            # run!
            nuts = pm.step_methods.NUTS(
                vars=self.params,
                target_accept=target_accept,
                max_treedepth=max_treedepth)
            trace = pm.sample(samples, nuts, chains=chains, cores=cores,
                              compute_convergence_checks=False, **kwargs)
        return trace

    def sample_predictions(
            self,
            target_days,
            target_counties,
            parameters,
            prediction_days,
            average_periodic_feature=False,
            average_all=False,
            init="auto"):

        all_days = pd.DatetimeIndex(
            [d for d in target_days] + [d for d in prediction_days])

        # extract features
        features = self.evaluate_features(all_days, target_counties)
        # num_counties = 412 #hardcoded; not needed?
        T_S = features["temporal_seasonal"].values
        T_T = features["temporal_trend"].values
        T_D = features["temporal_report_delay"].values
        TS = features["spatiotemporal"].values
        log_exposure = np.log(features["exposure"].values.ravel())

        
        if average_periodic_feature:
            T_S = np.reshape(T_S, newshape=(-1,412,5))
            mean = np.mean(T_S, axis=0, keepdims=True)
            T_S = np.reshape(np.tile(mean, reps=(T_S.shape[0],1,1)), (-1,5))          
        
        if average_all:
            T_S = np.reshape(T_S, newshape=(31,412,-1))
            mean = np.mean(T_S, axis=0, keepdims=True)
            T_S = np.reshape(np.tile(mean, reps=(31,1,1)), (-1,5))          
 
            TS = np.reshape(TS, newshape=(31,412,-1))
            mean = np.mean(TS, axis=0, keepdims=True)
            TS = np.reshape(np.tile(mean, reps=(31,1,1)),(-1,3)) 

            T_D = np.reshape(T_D, newshape=(31,412,-1))
            mean = np.mean(T_D, axis=0, keepdims=True)
            T_D = np.reshape(np.tile(mean, reps=(31,1,1)), (-1)) 

            log_exposure = np.reshape(log_exposure, newshape=(31,412))
            mean = np.mean(log_exposure, axis=0, keepdims=True)
            log_exposure = np.reshape(np.tile(mean, reps=(31,1)), (-1))

        # extract coefficient samples
        α = parameters["α"]
        W_t_s = parameters["W_t_s"]
        W_t_t = parameters["W_t_t"]
        W_t_d = parameters["W_t_d"]
        W_ts = parameters["W_ts"]

        num_predictions = len(target_days) * len(target_counties) + \
            len(prediction_days) * len(target_counties)
        num_parameter_samples = α.size
        y = np.zeros((num_parameter_samples, num_predictions), dtype=int)
        μ = np.zeros((num_parameter_samples, num_predictions),
                     dtype=np.float32)

        expanded_Wtt = np.tile(np.reshape(W_t_t, newshape=(-1,1,412,2)), reps=(1,31, 1, 1))
        expanded_TT = np.reshape(T_T, newshape=(1,31,412,2))
        result_TT = np.reshape(np.sum(expanded_TT*expanded_Wtt,axis=-1), newshape=(-1,31*412))
      
        for i in range(num_parameter_samples):
            μ[i, :] = np.exp(
                        np.dot(T_S, W_t_s[i]) +
                        result_TT[i] + 
                        np.dot(TS, W_ts[i]) +
                        np.dot(T_D, W_t_d[i]) + 
                        log_exposure)
            y[i, :] = pm.NegativeBinomial.dist(
                    mu=μ[i, :], alpha=α[i]).random()

        return {"y": y, "μ": μ, "α": α}
