import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
            # Define categorical and numerical columns based on your dataset
            categorical_cols = ['home_team', 'away_team', 'tournament']
            numerical_cols = ['home_score', 'away_score']

            # Define preprocessing steps for numerical columns
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),  # Handling missing values
                ("scaler", StandardScaler())  # Scaling numerical features
            ])

            # Define preprocessing steps for categorical columns
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),  # Handling missing categorical values
                ("onehot", OneHotEncoder(handle_unknown='ignore'))  # Encoding categorical features
            ])

            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor
        
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
            drop_columns = [target_column_name, 'date', 'city', 'country']  # Drop irrelevant columns

            # Separate input features and target for train and test datasets
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformation()

            # Apply transformations to train and test data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Log the shapes of the transformed arrays
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")

            # Ensure target array is reshaped correctly for concatenation
            target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

            logging.info("Preprocessing completed on training and testing datasets.")

            # Check shapes before concatenation
            logging.info(f"Shape after reshaping - target_feature_train_df: {target_feature_train_df.shape}")
            logging.info(f"Shape after reshaping - target_feature_test_df: {target_feature_test_df.shape}")

            # Combine input features and target into final arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # Save the preprocessor object
            save_object(
                file_path=self.dataTransformationConfig.Preprocessor_obj_filePath,
                obj=preprocessor
            )

            return train_arr, test_arr

        except Exception as e:
            logging.info(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

TRAIN_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "train.csv")
TEST_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "test.csv")
a=DataTransformation()
a.initate_data_transformation(TRAIN_DATA_PATH,TEST_DATA_PATH)