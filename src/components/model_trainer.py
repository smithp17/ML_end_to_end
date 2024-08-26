import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRF Regressor": XGBRFRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }
            
            params = {
                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None],
                },
                "Random Forest Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting Regressor": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.1, 0.05, 0.01, 1.0],
                    'loss': ['linear', 'square', 'exponential'],
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [10, 30, 50],
                },
                "XGBRF Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                },
                "CatBoost Regressor": {
                    'iterations': [100, 200, 300, 500],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.1, 0.01, 0.05],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'copy_X': [True, False],
                    'positive': [True, False],
                },
            }



            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params
            )

            # To get the best model score from dict
            best_model_score = max(model_report.values())

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model found with acceptable performance.")

            logging.info("Best Model found for both train and test")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
