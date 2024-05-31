"""This module contains all the components that are associated to Data Ingestion"""
import os
import sys
from zipfile import ZipFile
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.constants import *
from src.name_entity_recognition.entity.config_entity import DataIngestionConfig
from src.name_entity_recognition.entity.artifact_entity import DataIngestionArtifacts
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging

class DataIngestion:
    """This class encapulates all the methods that are used for data ingestion"""
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3_storage = S3Operations()

    def get_data_from_storage(self):
        """This method is to download the data from aws storage"""
        try:
            logging.info("Inside the get_data_from_storage method of \
                         src.name_entity_recognition.components.data_ingestion.DataIngestion class")
            
            self.s3_storage.download_object(
                key= self.data_ingestion_config.s3_data_file,
                bucket_name= self.data_ingestion_config.data_storage_bucket,
                filename = self.data_ingestion_config.s3_data_file_path
                )
            
            logging.info("Completed execution of get_data_from_storage method of \
                         src.name_entity_recognition.components.data_ingestion.DataIngestion class")
            
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

    def extract_data(self) ->None:
        """This method is used to extract all files from Zipped/Compressed files"""
        try:
            logging.info("Inside the extract_data method of \
                         src.name_entity_recognition.components.data_ingestion.DataIngestion class")
            
            with ZipFile(self.data_ingestion_config.s3_data_file_path, "r") as zipped_obj:
                zipped_obj.extractall(path= self.data_ingestion_config.output_file_path)


            logging.info("Completed extract_data method of \
                         src.name_entity_recognition.components.data_ingestion.DataIngestion class")
        
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        
    
    def initiate_data_ingestion(self) ->DataIngestionArtifacts:
        """This method is used to intitate the data ingestion"""
        try:
            logging.info("Inside the initiate_data_ingestion method of \
                         src.name_entity_recognition.components.data_ingestion.DataIngestion class")
            
            os.makedirs(self.data_ingestion_config.data_ingestion_artifacts_dir, exist_ok=True)
            logging.info(f"{self.data_ingestion_config.data_ingestion_artifacts_dir} has been created.")

            # Getting data from GCP
            self.get_data_from_storage()
            logging.info(f"Downloaded data from cloud into - {self.data_ingestion_config.output_file_path}")

            self.extract_data()
            logging.info(f"Extracted the data from zip file - {self.data_ingestion_config.output_file_path}")

            data_ingestion_artifact = DataIngestionArtifacts(
                zip_data_file_path=self.data_ingestion_config.s3_data_file_path,
                csv_data_file_path=self.data_ingestion_config.csv_data_file_path,
            )
            logging.info("Completed execution of initiate_data_ingestion method of \
                         src.name_entity_recognition.components.data_ingestion.DataIngestion class")
            
            return data_ingestion_artifact
        
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
