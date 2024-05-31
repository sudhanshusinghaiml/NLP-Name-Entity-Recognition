"""This module call the TrainPipeline.run_pipeline module for Training Models"""
import sys
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.pipeline.model_training_pipeline import TrainingPipeline
from src.name_entity_recognition.logger import logging


def training():
    """This method executes TrainPipeline.run_pipeline method"""
    try:
        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

    except Exception as error:
        logging.error(error)
        raise NERException(error, sys) from error


if __name__ == "__main__":
    training()
