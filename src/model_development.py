import logging
from abc import ABC, abstractmethod
import optuna
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract base class for all models.
    """
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        pass

class RandomForestModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(
            x_train, y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        return reg.score(x_test, y_test)

class LightGBMModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
        reg = self.train(
            x_train, y_train,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        return reg.score(x_test, y_test)

class XGBoostModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
        reg = self.train(
            x_train, y_train,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        return reg.score(x_test, y_test)

class LinearRegressionModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)

class HyperparameterTuner:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.model.optimize(
                trial, self.x_train, self.y_train, self.x_test, self.y_test
            ),
            n_trials=n_trials
        )
        return study.best_trial.params



















# import optuna
# from sklearn.ensemble import RandomForestRegressor
# from lightgbm import LGBMRegressor
# import xgboost as xgb
# from sklearn.linear_model import LinearRegression

# class Model(ABC):
#     @abstractmethod
#     def train(self, x_train, y_train):
#         pass

#     @abstractmethod
#     def optimize(self, trial, x_train, y_train, x_test, y_test):
#         pass

# class RandomForestModel(Model):
#     def train(self, x_train, y_train, **kwargs):
#         return RandomForestRegressor(**kwargs).fit(x_train, y_train)

#     def optimize(self, trial, x_train, y_train, x_test, y_test):
#         reg = self.train(
#             x_train, y_train,
#             n_estimators=trial.suggest_int("n_estimators", 10, 200),
#             max_depth=trial.suggest_int("max_depth", 2, 20),
#             min_samples_split=trial.suggest_int("min_samples_split", 2, 20)
#         )
#         return reg.score(x_test, y_test)

# class LightGBMModel(Model):
#     def train(self, x_train, y_train, **kwargs):
#         return LGBMRegressor(**kwargs).fit(x_train, y_train)

#     def optimize(self, trial, x_train, y_train, x_test, y_test):
#         reg = self.train(
#             x_train, y_train,
#             n_estimators=trial.suggest_int("n_estimators", 10, 200),
#             learning_rate=trial.suggest_uniform("learning_rate", 0.01, 0.5),
#             max_depth=trial.suggest_int("max_depth", 2, 20)
#         )
#         return reg.score(x_test, y_test)

# class XGBoostModel(Model):
#     def train(self, x_train, y_train, **kwargs):
#         return xgb.XGBRegressor(**kwargs).fit(x_train, y_train)

#     def optimize(self, trial, x_train, y_train, x_test, y_test):
#         reg = self.train(
#             x_train, y_train,
#             n_estimators=trial.suggest_int("n_estimators", 10, 200),
#             learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 1.0),
#             max_depth=trial.suggest_int("max_depth", 2, 30)
#         )
#         return reg.score(x_test, y_test)

# class LinearRegressionModel(Model):
#     def train(self, x_train, y_train, **kwargs):
#         return LinearRegression(**kwargs).fit(x_train, y_train)

#     def optimize(self, trial, x_train, y_train, x_test, y_test):
#         return self.train(x_train, y_train).score(x_test, y_test)

# class HyperparameterTuner:
#     def __init__(self, model, x_train, y_train, x_test, y_test):
#         self.model = model
#         self.x_train = x_train
#         self.y_train = y_train
#         self.x_test = x_test
#         self.y_test = y_test

#     def optimize(self, n_trials=100):
#         study = optuna.create_study(direction="maximize")
#         study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
#         return study.best_trial.params

