import os
import sys
from dataclasses import dataclass
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XG Boost": XGBRegressor(),
                "Cat Boost": CatBoostRegressor(),
                "Ada Boost": AdaBoostRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Lasso": {
                    'alpha': np.linspace(0, 0.2, 21)
                },
                "Ridge": {
                    'alpha': np.arange(0, 1, 0.01)
                },
                "K Neighbors": {
                    'n_neighbors': np.arange(1, 21, 2),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                "XG Boost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Ada Boost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report: dict=evaluate_models(
                X_train=X_train, 
                y_train=Y_train, 
                X_test=X_test, 
                y_test=Y_test, 
                models=models,
                param=params
            )
            logging.info(model_report)
            #to get best model score from report dict
            best_model_score = max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]     
            best_model = models[best_model_name]   

            if best_model_score<0.6:
                raise CustomException("No best model found")    

            logging.info("Best model found: {} with r2 score of: {}".format(best_model_name, best_model_score))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException()