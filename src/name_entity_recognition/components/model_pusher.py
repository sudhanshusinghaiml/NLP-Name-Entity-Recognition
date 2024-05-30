"""This moulde contains emthods or classes for pushing the best models into S3 Bucket"""

import sys
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.entity.artifact_entity import (
    ModelPusherArtifacts,
    ModelTrainingArtifacts,
)
from src.name_entity_recognition.entity.config_entity import ModelPusherConfig
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging


class ModelPusher:
    """This class encapsulates the methods used for pushing the model in Storage"""

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_training_artifacts: ModelTrainingArtifacts,
        s3_storage: S3Operations,
    ):
        self.model_pusher_config = model_pusher_config
        self.model_training_artifacts = model_training_artifacts
        self.s3_storage = s3_storage

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """This method is used for pushing the models into S3 Bucket"""
        try:
            logging.info(
                "Inside initiate_model_pusher method of\
                         src.object_detection.model_pusher.ModelPusher"
            )

            self.s3_storage.upload_file(
                self.model_training_artifacts.bert_model_path,
                self.model_pusher_config.model_name,
                self.model_pusher_config.bucket_name,
                remove=False,
            )

            logging.info("Uploaded best model to S3 bucket")

            model_pusher_artifacts = ModelPusherArtifacts(
                bucket_name =self.model_pusher_config.bucket_name,
                model_path =self.model_pusher_config.upload_model_path,
            )

            logging.info(
                "Completed execution of initiate_model_pusher method of\
                         src.object_detection.model_pusher.ModelPusher"
            )

            return model_pusher_artifacts

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
