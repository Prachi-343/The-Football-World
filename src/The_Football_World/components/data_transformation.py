import os
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException

class DataTransformationConfig:
    PREPROCESSOR_OBJ_FILE_PATH = os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        """Returns a preprocessing pipeline."""
        try:
            numerical_columns = ['home_score', 'away_score']
            categorical_columns = ['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral']

            # Numeric transformations
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="mean")),
                ('scaler', StandardScaler())
            ])

            # Categorical transformations
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_columns),
                ('cat', categorical_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Transforms the data."""
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data loaded for transformation")

            # Separate features and target
            input_features_train_df = train_df.drop(columns=['match_outcome'], axis=1)
            target_feature_train_df = train_df['match_outcome']
            input_features_test_df = test_df.drop(columns=['match_outcome'], axis=1)
            target_feature_test_df = test_df['match_outcome']

            # Get preprocessor object
            preprocessor = self.get_preprocessor()
            logging.info("Preprocessor object created")

            # Fit preprocessor on train data and transform both train and test data
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor.transform(input_features_test_df)

            # Save the preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.PREPROCESSOR_OBJ_FILE_PATH), exist_ok=True)
            joblib.dump(preprocessor, self.data_transformation_config.PREPROCESSOR_OBJ_FILE_PATH)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.PREPROCESSOR_OBJ_FILE_PATH}")

            return input_features_train_arr, target_feature_train_df, input_features_test_arr, target_feature_test_df
        except Exception as e:
            raise CustomException(e, sys)
