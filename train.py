"""This module call the TrainPipeline.run_pipeline module for Training Models"""
import sys
from src.name_entity_recognition.exception import NerException
from src.name_entity_recognition.pipeline.model_training_pipeline import TrainPipeline


def training():
    """This method executes TrainPipeline.run_pipeline method"""
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

    except Exception as e:
        raise NerException(e, sys) from e


if __name__ == "__main__":
    training()
