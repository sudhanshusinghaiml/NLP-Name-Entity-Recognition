"""This module consists of artifacts configuration for all pipeline components"""

from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    """Data Ingestion Artifacts"""
    zip_data_file_path: str
    csv_data_file_path: str

@dataclass
class DataTransformationArtifacts:
    """Data Transformation Artifacts"""
    labels_to_ids_path: str
    ids_to_labels_path: str
    df_train_path: str
    df_val_path: str
    df_test_path: str
    unique_labels_path: str


@dataclass
class ModelTrainingArtifacts:
    """Model Training Artifacts"""
    bert_model_path: str
    tokenizer_file_path: str


@dataclass
class ModelEvaluationArtifacts:
    """Model Evaluation Artifacts"""
    model_accuracy: float
    is_model_accepted: bool


@dataclass
class ModelPusherArtifacts:
    """Model Pusher Artifacts"""
    bucket_name: str
    model_path: str
