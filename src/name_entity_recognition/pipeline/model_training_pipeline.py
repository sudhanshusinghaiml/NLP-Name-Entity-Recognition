"""This module contains the code for Model Training Pipeline"""
import sys
from src.name_entity_recognition.components.data_ingestion import DataIngestion
# from src.name_entity_recognition.components.data_transformation import DataTransformation
# from src.name_entity_recognition.components.model_trainer import ModelTraining
# from src.name_entity_recognition.components.model_evaluation import ModelEvaluation
# from src.name_entity_recognition.components.model_pusher import ModelPusher
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.constants import *

from src.name_entity_recognition.entity.artifact_entity import (DataIngestionArtifacts
                                                                # DataTransformationArtifacts,
                                                                # ModelTrainingArtifacts,
                                                                # ModelEvaluationArtifacts,
                                                                # ModelPusherArtifacts
                                                                )


from src.name_entity_recognition.entity.config_entity import ( DataIngestionConfig
                                                                # DataTransformationConfig,
                                                                # ModelTrainingConfig,
                                                                # ModelEvalConfig,
                                                                # ModelPusherConfig
                                                              )

from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging

class TrainingPipeline:
    """This class contains the codes that is used to start the execution of all the pipeline"""
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        #self.data_transformation_config = DataTransformationConfig()
        #self.model_training_config = ModelTrainingConfig()
        #self.model_evaluation_config = ModelEvalConfig()
        #self.model_pusher_config = ModelPusherConfig()
        self.s3_storage = S3Operations()

    
     # This method is used to start the data ingestion
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """This method is used for triggering data ingestion from cloud storage to local storage"""
        try:
            logging.info("Inside the start_data_ingestion method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Completed execution of start_data_ingestion method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            return data_ingestion_artifact

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        
    
    def run_pipeline(self) -> None:
        """This method is used to start the training pipeline"""
        try:
            logging.info("Started Model training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            data_ingestion_artifact = self.start_data_ingestion()
            # data_transformation_artifacts = self.start_data_transformation(
            #     data_ingestion_artifact=data_ingestion_artifact
            # )
            # model_trainer_artifact = self.start_model_training(
            #     data_transformation_artifacts=data_transformation_artifacts
            # )
            # model_evaluation_artifact = self.start_model_evaluation(
            #     data_transformation_artifact=data_transformation_artifacts,
            #     model_trainer_artifact=model_trainer_artifact,
            # )

            # model_pusher_artifact = self.start_model_pusher(
            #     model_evaluation_artifact=model_evaluation_artifact
            # )

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error