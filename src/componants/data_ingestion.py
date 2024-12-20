import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.componants.data_transform import datatransform
from src.componants.data_transform import datatransformconfig

from src.componants.model_train import ModelTrainerConfig
from src.componants.model_train import ModelTrainer

@dataclass
class dataIngestconfig:
    train_data_path: str = os.path.join('artifacts' , "train.csv")
    test_data_path: str = os.path.join('artifacts' , "test.csv")
    raw_data_path: str = os.path.join('artifacts' , "data.csv")
    
class dataingest:

    def __init__(self):
        self.ingestion_config = dataIngestconfig()

    def initiate_data_ingestion(self):
        logging.info(" entered the data ingestion method")  
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("read dataset as df")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split intiated")
            train_set,test_set = train_test_split(df,test_size=0.2 , random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = dataingest()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transforamion = datatransform()
    train_arr , test_arr , _ = data_transforamion.intitate_data_transform(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr , test_arr))