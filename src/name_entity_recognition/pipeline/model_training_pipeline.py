"""This module contains the code for Model Training Pipeline"""
import sys
from src.name_entity_recognition.components.data_ingestion import DataIngestion
from src.name_entity_recognition.components.data_transformation import DataTransformation
from src.name_entity_recognition.components.model_trainer import ModelTraining
from src.name_entity_recognition.components.model_evaluation import ModelEvaluation
from src.name_entity_recognition.components.model_pusher import ModelPusher
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.constants import *

from src.name_entity_recognition.entity.artifact_entity import (DataIngestionArtifacts,
                                                                DataTransformationArtifacts,
                                                                ModelTrainingArtifacts,
                                                                ModelEvaluationArtifacts,
                                                                ModelPusherArtifacts
                                                                )


from src.name_entity_recognition.entity.config_entity import ( DataIngestionConfig,
                                                               DataTransformationConfig,
                                                               ModelTrainingConfig,
                                                               ModelEvaluationConfig,
                                                               ModelPusherConfig
                                                              )

from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging

class TrainingPipeline:
    """This class contains the codes that is used to start the execution of all the pipeline"""
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_training_config = ModelTrainingConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.s3_storage = S3Operations()

    
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
        

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifacts) -> DataTransformationArtifacts:
        """This method is used for triggering data transformation"""
        try:
            logging.info("Inside the start_data_transformation method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifacts=data_ingestion_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info("Completed execution of start_data_transformation method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            return data_transformation_artifact

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

    def start_model_training(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainingArtifacts:
        """This method is used for triggering model training"""
        try:
            logging.info("Inside the start_model_training method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            model_trainer = ModelTraining(
                model_training_config=self.model_training_config,
                data_transformation_artifacts=data_transformation_artifacts,
            )
            model_trainer_artifact = model_trainer.initiate_model_training()

            logging.info("Completed execution of start_model_training method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            return model_trainer_artifact

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

    def start_model_evaluation(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_artifacts: ModelTrainingArtifacts) -> ModelEvaluationArtifacts:
        """This method is used for triggering model evaluation"""
        try:
            logging.info("Inside the start_model_evaluation method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            model_evaluation = ModelEvaluation(
                data_transformation_artifacts = data_transformation_artifacts,
                model_training_artifacts = model_trainer_artifacts,
                model_evaluation_config=self.model_evaluation_config,
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()


            logging.info("Completed execution of start_model_evaluation method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            return model_evaluation_artifact

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

    def start_model_pusher(self, model_evaluation_artifacts: ModelEvaluationArtifacts) -> ModelPusherArtifacts:
        """This method is used for triggering pushing the best model to cloud storage"""
        try:
            logging.info("Inside the start_model_pusher method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifacts,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Completed execution of start_model_pusher method of \
                         src.name_entity_recognition.pipeline.model_training_pipeline.TrainingPipeline class")
            
            return model_pusher_artifact

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
    
    def run_pipeline(self) -> None:
        """This method is used to start the training pipeline"""
        try:
            logging.info("Started Model training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            
            data_ingestion_artifact = self.start_data_ingestion()
            
            logging.info("Completed Data Ingestion >>>>>>>>>>>>>>>>>>>>>>>>>>>")

            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            logging.info("Completed Data Transformation >>>>>>>>>>>>>>>>>>>>>>>>")

            model_trainer_artifact = self.start_model_training(
                data_transformation_artifacts=data_transformation_artifacts
            )
            logging.info("Completed Model Training >>>>>>>>>>>>>>>>>>>>>>>>")

            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifacts,
                model_trainer_artifact=model_trainer_artifact,
            )

            logging.info("Completed Model Evaluation >>>>>>>>>>>>>>>>>>>>>>>>>>")

            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

            logging.info("Completed Model Pusher >>>>>>>>>>>>>>>>>>>>>>>>>>")

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error