"""This module contains constants that will be used for Pipeline"""

ARTIFACTS_DIR: str = "artifacts"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_S3_DATA_NAME: str = "isd_data.zip"

DATA_BUCKET_NAME: str = "name-entity-recognition-data-28052024"


"""
Data Transformation related constant start with DATA_INGESTION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_ingestion"

DATA_FEATURE_STORE_DIR: str = "feature_store"

DATA_TRANSFORMATION_S3_DATA_NAME: str = "isd_data.zip"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_training"

BEST_MODEL_NAME = ""

MODEL_TRAINER_NO_EPOCHS: int = 5

MODEL_TRAINER_BATCH_SIZE: int = 2


"""
MODEL PUSHER related constant start with MODEL_PUSHER var name
"""
MODEL_BUCKET_NAME = "name-entity-recognition-model-30052024"
S3_MODEL_NAME = ""
