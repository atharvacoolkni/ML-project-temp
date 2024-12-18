import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os

from src.utilis import save_object

class datatransformconfig:
    preprocessor_ob_file_path = os.path.join('artifacts',"preprocessor.pkl")
class datatransform:
    def __init__(self):
        self.data_transformer_config = datatransformconfig()

    def get_data_transform_obj(self):
        try:
            numerical_column = ["writing_score" , "reading_score"]
            categorical_column = ["gender", 
                                "race_ethnicity", 
                                "parental_level_of_education",
                                "lunch", 
                                "test_preparation_course",]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy="median")),
                    ("scaler" , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder" , OneHotEncoder()),
                    
                ]
            )

            logging.info("nmerical standard scaling complete")
            logging.info("cat columns encoding xomplete")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline" , num_pipeline , numerical_column),
                    ("cat_pipelines" , cat_pipeline,categorical_column)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e ,sys)

    def intitate_data_transform(self , train_path ,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train test data completed")

            preproessing_obj = self.get_data_transform_obj()

            target_column  = "math_score"

            numerical_column = ["writing_score" , "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column] , axis =1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column] , axis =1)
            target_feature_test_df = test_df[target_column]


            input_feature_train_arr = preproessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preproessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")

            save_object(

                file_path = self.data_transformer_config.preprocessor_ob_file_path,
                obj = preproessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_ob_file_path,
            )
        except Exception as e:   
            raise CustomException(e , sys)

