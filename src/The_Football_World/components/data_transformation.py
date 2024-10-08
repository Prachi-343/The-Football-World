import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier
from skopt import BayesSearchCV 
from skopt.space import Real, Integer
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException
from src.The_Football_World.utils.utils import save_object

class DataTransformationConfig:
    Preprocessor_obj_filePath = os.path.join('artifacts', 'Preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.dataTransformationConfig = DataTransformationConfig()
        
    def get_data_transformation(self):
        logging.info("get data transformation")
        try:
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("Encoder", TargetEncoder()), 
                ("clf", XGBClassifier(random_state=8, enable_categorical=True)) 
            ])

            search_space = {
                'clf__max_depth': Integer(2, 8),
                'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                'clf__subsample': Real(0.5, 1.0),
                'clf__colsample_bytree': Real(0.5, 1.0), 
                'clf__colsample_bylevel': Real(0.5, 1.0),
                'clf__colsample_bynode': Real(0.5, 1.0),
                'clf__reg_alpha': Real(0.0, 10.0),
                'clf__reg_lambda': Real(0.0, 10.0),
                'clf__gamma': Real(0.0, 10.0)
            }


            opt = BayesSearchCV(pipeline, search_space, cv=3, n_iter=18, scoring='roc_auc_ovr', random_state=8)

            return opt
        except Exception as e:
            logging.info(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)
    
    def initate_data_transformation(self, train_path, test_path):
        logging.info("Data transformation started")
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data successfully")
            logging.info(f"Train Data Sample: \n{train_df.head().to_string()}")
            logging.info(f"Test Data Sample: \n{test_df.head().to_string()}")

            # Target column and columns to drop
            target_column_name = "match_outcome"  # Outcome as target for classification
            drop_columns = [target_column_name, "date", "city", "country","neutral"]  # Drop irrelevant columns like 'date'

            # Separate input features and target for train and test datasets
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Get the preprocessing pipeline
            opt = self.get_data_transformation()

            logging.info("Preprocessing completed on training and testing datasets.")

            # Save the preprocessor object
            save_object(
                file_path=self.dataTransformationConfig.Preprocessor_obj_filePath,
                obj=opt
            )

            return (self.dataTransformationConfig.Preprocessor_obj_filePath,
                    input_feature_train_df,target_feature_train_df)

        except Exception as e:
            logging.info(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

