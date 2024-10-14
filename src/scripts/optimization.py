from typing import Dict, Tuple
import itertools
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm.notebook import tqdm
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
import pickle

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class BayesianOptimizationConfig:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

  def __getattr__(self, name):
    return None

# source 1 https://machinelearningmastery.com/what-is-bayesian-optimization/
# source 2 https://pyro.ai/examples/bo.html
# source 3 https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f
class BayesianOptimization:
    def __init__(self, config: BayesianOptimizationConfig):
        self.hp_bounds: Dict[Tuple]    = config.hp_bounds
        self.search_space              = list(itertools.product(config.lags, config.differentiation))
        self.cross_validator           = config.cross_validator
        self.data: pd.DataFrame        = config.data
        self.search_size: int          = config.search_size
        self.objective                 = config.objective
        self.num_hp: int               = len(self.hp_bounds)
        # below will contain the searched parameters
        self.searched_params: Dict     = dict()
        self.gprs: Dict              = dict()

    # ------------------------------------
    #     Bayesian Opt. Workflow Start
    # ------------------------------------
    
    def _objective_function(self, forecaster, exog=True): # returns cross validated score
        maes = []
        for x, y in self.cross_validator.split(self.data):
            endog_train = self.data.loc[x, 'consumption'].reset_index(drop=True)
            endog_val = self.data.loc[y, 'consumption'].reset_index(drop=True)
            exog_train = self.data.loc[x] \
                            .drop(['date', 'consumption'], axis=1) \
                            .reset_index(drop=True) \
                            .astype(float) if exog else None
            exog_val = self.data.loc[y] \
                            .drop(['date', 'consumption'], axis=1) \
                            .set_index(
                                pd.RangeIndex(
                                    start=len(exog_train), 
                                    stop=len(exog_train)+len(y)
                                )
                            ) \
                            .astype(float) if exog else None
            forecaster.fit(y=endog_train, exog=exog_train)
            preds = forecaster.predict(steps=len(endog_val), exog=exog_val)
            maes.append(self.objective(preds, endog_val))
        return np.mean(maes), np.std(maes)

    def _surrogate_model(self, params: np.array, targets: np.array): # returns GPR model
        gpr = GaussianProcessRegressor(alpha=1e-3, n_restarts_optimizer=10, normalize_y=True)
        gpr.fit(params, targets)
        return gpr

    def _acquisition_function(
        self, 
        gpr: GaussianProcessRegressor, 
        params: np.array, 
        kappa=2.753
    ): # returns next params using LCB?
        mu, std = gpr.predict(params, return_std=True)
        return mu - kappa * std

    # ------------------------------------
    #     Bayesian Opt. Workflow End
    # ------------------------------------

    # -----------------------
    #  Using the Object Start
    # -----------------------
    
    def _params_to_dict(self, param_array: np.array):
        return {key: value.astype(type) for (key, (_, _, type)), value in zip(self.hp_bounds.items(), param_array)}
    
    def _uniform_random_params(self, size: int):
        array = np.array([
            np.random.uniform(low=low, high=high, size=size).round(3) \
            if dtype == float else np.random.randint(low=low, high=high, size=size) \
            for key, (low, high, dtype) in self.hp_bounds.items()
        ]).squeeze()
        return array
        
    def fit(self, n_iters: int):
        space = tqdm(self.search_space, position=0, total=len(self.search_space), desc="HP Search")
        for lag, diff in space:
            searched_params, param_objs = None, None
            stds = []
            param_array = self._uniform_random_params(1)
            params = self._params_to_dict(param_array)
            iters = tqdm(range(n_iters), position=1, leave=False, desc=f'diff={diff} lag={lag}')
            for _ in iters:
                forecaster = ForecasterAutoreg(
                    regressor       = XGBRegressor(**params, random_state=RANDOM_STATE),
                    lags            = lag,
                    differentiation = diff,
                )
                mean, std = self._objective_function(forecaster, exog=True)
                # foresake this run since it will mess up our plots
                if (param_objs is not None) and (mean > np.max(param_objs) * 3.5):
                    gpr = self._surrogate_model(searched_params, param_objs)
                    random_space = self._uniform_random_params(self.search_size)
                    proposed_objs = self._acquisition_function(gpr, random_space.T)
                    idx = np.argmin(proposed_objs)
                    param_array = random_space.T[idx]
                    params = self._params_to_dict(param_array)
                    continue
                searched_params = np.vstack([searched_params, param_array]) if searched_params is not None else np.array([param_array])
                param_objs = np.vstack([param_objs, mean]) if param_objs is not None else np.array([mean])
                stds.append(std)
                gpr = self._surrogate_model(searched_params, param_objs)
                
                random_space = self._uniform_random_params(self.search_size)
                proposed_objs = self._acquisition_function(gpr, random_space.T)
                idx = np.argmin(proposed_objs)
                param_array = random_space.T[idx]
                params = self._params_to_dict(param_array)
                
            self.searched_params[(lag, diff)] = (searched_params, param_objs, stds)
            self.gprs[(lag, diff)] = gpr

    def best_params(self):
        best = np.inf
        best_params = None
        best_lag = None
        for lag, (param_array, objs, stds) in self.searched_params.items():
            if np.min(objs) < best:
                idx = np.argmin(objs)
                params = param_array[idx]
                best_params = self._params_to_dict(params)
                best_lag = lag
                std = stds[idx]
                best = np.min(objs)
        return {
            'params': best_params,
            'lag': best_lag[0], 
            'differentiation': best_lag[1],
            'mean': best,
            'std': std
            }

    def best_model(self):
        params, reg_params, _ = self.best_params()
        forecaster = ForecasterAutoreg(
            regressor = XGBRegressor(**params, random_state=RANDOM_STATE),
            **reg_params
        )
        return forecaster

    def gpr_(self, lag: int):
        return self.gprs[lag]

    def training_schedule(self, lag: int):
        return self.searched_params[lag]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    # -----------------------
    #  Using the Object End
    # -----------------------