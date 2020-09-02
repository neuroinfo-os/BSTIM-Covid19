from sampling_utils import *
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


class IAEffectLoader(object):
    generates_stats = False

    def __init__(self, var, filenames, days, counties, predict_for=None):
        self.vars = [var]
        self.samples = []
        i = 0
        for filename in filenames:
            try:
                with open(filename, "rb") as f:
                    tmp = pkl.load(f)
            except FileNotFoundError:
                print("Warning: File {} not found!".format(filename))
                pass
            except Exception as e:
                print(e)
            else:
                m = tmp["ia_effects"]
                ds = list(tmp["predicted day"])
                cs = list(tmp["predicted county"])
                d_idx = np.array([ds.index(d) for d in days]).reshape((-1, 1))
                print(i)
                i = i+1
                print("Days")
                print(days)
                print("ds")
                print(ds)
                for d in days:
                    print(d)
                    print(ds.index(d))
                c_idx = np.array([cs.index(c) for c in counties])

                # Simulate linear IA effects if predicting the future
                if predict_for is not None:
                    d1 = [ds.index(d) for d in days]
                    d2 = list(range(d1[-1], d1[-1]+len(predict_for)))
                    n_days_pred = len(d2)
                    # Repeat ia_effects for last day.
                    last = m[-1, :, :]
                    last = np.tile(last, (n_days_pred, 1, 1))
                    m = np.concatenate((m, last), axis=0)
                    # Update d_idx.
                    d_idx = np.array(d1 + d2).reshape(-1, 1)

                self.samples.append(np.moveaxis(
                    m[d_idx, c_idx, :], -1, 0).reshape((m.shape[-1], -1)).T)

    def step(self, point):
        new = point.copy()
        # res = new[self.vars[0].name]
        new_res = self.samples[np.random.choice(len(self.samples))]
        new[self.vars[0].name] = new_res
        # random choice; but block structure <-- this must have "design matrix" shape/content
        return new

    def stop_tuning(self, *args):
        pass

    @property
    def vars_shape_dtype(self):
        shape_dtypes = {}
        for var in self.vars:
            dtype = np.dtype(var.dtype)
            shape = var.dshape
            shape_dtypes[var.name] = (shape, dtype)
        return shape_dtypes


class BaseModel(object):
    """
    Model for disease prediction.

    The model has 4 types of features (predictor variables):
    * temporal (functions of time)
    * spatial (functions of space, i.e. longitude, latitude)
    * county_specific (functions of time and space, i.e. longitude, latitude)
    * interaction effects (functions of distance in time and space relative to each datapoint)
    """

    def __init__(
            self,
            trange,
            counties,
            ia_effect_filenames,
            num_ia=16,
            model=None,
            include_ia=True,
            include_report_delay=True,
            report_delay_order=4,
            include_demographics=True,
            include_temporal=True,
            trend_poly_order=4,
            include_periodic=True,
            periodic_poly_order=4,
            orthogonalize=False):

        self.county_info = counties
        self.ia_effect_filenames = ia_effect_filenames
        self.num_ia = num_ia if include_ia else 0
        self.include_ia = include_ia
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
            if self.include_report_delay else {},
            "exposure": {
                "exposure": SpatioTemporalYearlyDemographicsFeature(
                    self.county_info,
                    "total",
                    1.0 / 100000)}}

        # self.Q = np.eye(self.num_ia, dtype=np.float32)
        # if orthogonalize:
        #     # transformation to orthogonalize IA features
        #     T = np.linalg.inv(np.linalg.cholesky(
        #         gaussian_gram([6.25, 12.5, 25.0, 50.0]))).T
        #     for i in range(4):
        #         self.Q[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = T

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

    def init_model(self, target, window=False):
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
        

        if self.include_ia:

            with pm.Model() as self.model:
                # interaction effects are generated externally -> flat prior
                IA = pm.Flat("IA", testval=np.ones(
                    (num_obs, self.num_ia)), shape=(num_obs, self.num_ia))

                # priors
                # NOTE: Vary parameters over time -> W_ia dependent on time
                # δ = 1/√α
                δ = pm.HalfCauchy("δ", 10, testval=1.0)
                α = pm.Deterministic("α", np.float32(1.0) / δ)
                W_ia = pm.Normal("W_ia", mu=0, sd=10, testval=np.zeros(
                    self.num_ia), shape=self.num_ia)
                W_t_s = pm.Normal("W_t_s", mu=0, sd=10,
                                  testval=np.zeros(num_t_s), shape=num_t_s)
                if window:
                    # initialize W_t_t to have dimension (412,2)
                    W_t_t = pm.Normal("W_t_t", mu=0, sd=10,
                                      testval=np.zeros((num_counties, num_t_t)), shape=(num_counties, num_t_t))
                else:
                    W_t_t = pm.Normal("W_t_t", mu=0, sd=10,
                                        testval=np.zeros(num_t_t), shape=num_t_t)
                W_t_d = pm.Normal("W_t_d", mu=0, sd=10,
                                  testval=np.zeros(num_t_d), shape=num_t_d)
                W_ts = pm.Normal("W_ts", mu=0, sd=10,
                                 testval=np.zeros(num_ts), shape=num_ts)
                self.param_names = ["δ", "W_ia",
                                    "W_t_s", "W_t_t", "W_t_d", "W_ts"]
                self.params = [δ, W_ia, W_t_s, W_t_t, W_t_d, W_ts]

                # calculate interaction effect
                IA_ef = tt.dot(IA, W_ia)

                if window:
                    # possibly four weeks instead of three
                    expanded_Wtt = tt.tile(W_t_t.reshape(shape=(1,num_counties,-1)), reps=(21, 1, 1))
                    expanded_TT = np.reshape(T_T, newshape=(21,412,2))
                    result_TT = tt.flatten(tt.sum(expanded_TT*expanded_Wtt,axis=-1))
                else:
                    result_TT = tt.dot(T_T, W_t_t)

                # calculate mean rates
                μ = pm.Deterministic(
                    "μ",
                    tt.exp(
                        IA_ef +
                        tt.dot(T_S, W_t_s) +
                        result_TT + 
                        tt.dot(T_D, W_t_d) +
                        tt.dot(TS, W_ts)+
                        log_exposure
                        )
                      )
                # constrain to observations
                pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y_obs)

        else:
            # here the 3 week window prediction is not modeled yet
            with pm.Model() as self.model:
                # priors
                # δ = 1/√α
                δ = pm.HalfCauchy("δ", 10, testval=1.0)
                α = pm.Deterministic("α", np.float32(1.0) / δ)
                W_t_s = pm.Normal("W_t_s", mu=0, sd=10,
                                  testval=np.zeros(num_t_s), shape=num_t_s)
                W_t_t = pm.Normal("W_t_t", mu=0, sd=10,
                                  testval=np.zeros(num_t_t), shape=num_t_t)
                W_t_d = pm.Normal("W_t_d", mu=0, sd=10,
                                  testval=np.zeros(num_t_d), shape=num_t_d)
                W_ts = pm.Normal("W_ts", mu=0, sd=10,
                                 testval=np.zeros(num_ts), shape=num_ts)
                self.param_names = ["δ", "W_t_s", "W_t_t", "W_t_d", "W_ts"]
                self.params = [δ, W_t_s, W_t_t, W_t_d, W_ts]

                # calculate mean rates
                μ = pm.Deterministic(
                    "μ",
                    tt.exp(
                        tt.dot(T_S, W_t_s) +
                        tt.dot(T_T, W_t_t) +
                        tt.dot(T_D, W_t_d) +
                        tt.dot(TS, W_ts) +
                        log_exposure))

                # constrain to observations
                pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y_obs)

    def map_estimate():
        """ TODO Q: how to include IA?"""
        pass

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
            window=False,
            **kwargs):
        """
            sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

        Samples from the posterior parameter distribution, given a training dataset.
        The basis functions are designed to be causal, i.e. only data points strictly
        predating the predicted time points are used (this implies "one-step-ahead"-predictions).
        """
        # model = self.model(target)

        self.init_model(target,window=window)

        if chains is None:
            chains = max(2, cores)

        if self.include_ia:
            with self.model:
                # run!
                ia_effect_loader = IAEffectLoader(
                    self.model.IA,
                    self.ia_effect_filenames,
                    target.index,
                    target.columns)
                nuts = pm.step_methods.NUTS(
                    vars=self.params,
                    target_accept=target_accept,
                    max_treedepth=max_treedepth)
                steps = [ia_effect_loader, nuts]
                trace = pm.sample(samples, steps, chains=chains, cores=cores,
                                  compute_convergence_checks=False, **kwargs)
        else:
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
            window=False,
            init="auto"):

        all_days = pd.DatetimeIndex(
            [d for d in target_days] + [d for d in prediction_days])
        # extract features
        features = self.evaluate_features(all_days, target_counties)
        num_counties = 412 #hardcoded
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

        if self.include_ia:
            W_ia = parameters["W_ia"]
            ia_l = IAEffectLoader(None, self.ia_effect_filenames,
                                  target_days, target_counties, predict_for=prediction_days)

        num_predictions = len(target_days) * len(target_counties) + \
            len(prediction_days) * len(target_counties)
        num_parameter_samples = α.size
        y = np.zeros((num_parameter_samples, num_predictions), dtype=int)
        μ = np.zeros((num_parameter_samples, num_predictions),
                     dtype=np.float32)

        # only consider the mean effect of the delay polynomial // should be a function?!
        # mean_delay = np.zeros((num_predictions,))
        # for i in range(num_parameter_samples):
        #     mean_delay += np.dot(T_D, W_t_d[i])

     
        if window:
            # possibly four weeks instead of three
            expanded_Wtt = np.tile(np.reshape(W_t_t, newshape=(-1,1,412,2)), reps=(1,31, 1, 1))
            expanded_TT = np.reshape(T_T, newshape=(1,31,412,2))
            result_TT = np.reshape(np.sum(expanded_TT*expanded_Wtt,axis=-1), newshape=(-1,31*412))
        else:
            result_TT = tt.dot(T_T, W_t_t)
      
       # NOTE: the delay polynomial is left out here!
        # mean_delay /= num_parameter_samples
        if self.include_ia:
            for i in range(num_parameter_samples):
                IA_ef = np.dot(
                    ia_l.samples[np.random.choice(len(ia_l.samples))], W_ia[i])
                # np.dot(ia_l.samples[np.random.choice(len(ia_l.samples))], self.Q), W_ia[i])
                if average_all:
                    IA_ef = np.reshape(IA_ef, newshape=(31,412))
                    mean = np.mean(IA_ef, axis=0, keepdims=True)
                    IA_ef = np.reshape(np.tile(mean, reps=(31,1)), (-1)) 
                μ[i, :] = np.exp(IA_ef +
                            np.dot(T_S, W_t_s[i]) +
                            result_TT[i] + 
                            np.dot(TS, W_ts[i]) +
                            log_exposure)
                y[i, :] = pm.NegativeBinomial.dist(
                        mu=μ[i, :], alpha=α[i]).random()
        # again not modeled
        else:
            for i in range(num_parameter_samples):
                μ[i, :] = np.exp(np.dot(T_S, W_t_s[i]) +
                                 np.dot(T_T, W_t_t[i]) +
                                 np.dot(TS, W_ts[i]) +
                                 log_exposure)
                y[i, :] = pm.NegativeBinomial.dist(
                    mu=μ[i, :], alpha=α[i]).random()

        return {"y": y, "μ": μ, "α": α}
