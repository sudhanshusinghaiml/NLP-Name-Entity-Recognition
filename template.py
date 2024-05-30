"""
This module is typically used for creating project template
"""

import os
from pathlib import Path

PROJECT_DIRECTORY = "name_entity_recognition"

list_of_files = [
    f"src/__init__.py",
    f"src/{PROJECT_DIRECTORY}/__init__.py",
    f"src/{PROJECT_DIRECTORY}/components/__init__.py",
    f"src/{PROJECT_DIRECTORY}/components/data_ingestion.py",
    f"src/{PROJECT_DIRECTORY}/components/data_transformation.py",
    f"src/{PROJECT_DIRECTORY}/components/model_trainer.py",
    f"src/{PROJECT_DIRECTORY}/components/model_evaluation.py",
    f"src/{PROJECT_DIRECTORY}/components/model_pusher.py",
    f"src/{PROJECT_DIRECTORY}/configuration/__init__.py",
    f"src/{PROJECT_DIRECTORY}/configuration/aws_storage_operations.py",
    f"src/{PROJECT_DIRECTORY}/constants/__init__.py",
    f"src/{PROJECT_DIRECTORY}/entity/__init__.py",
    f"src/{PROJECT_DIRECTORY}/entity/config_entity.py",
    f"src/{PROJECT_DIRECTORY}/entity/artifacts_entity.py",
    f"src/{PROJECT_DIRECTORY}/exception/__init__.py",
    f"src/{PROJECT_DIRECTORY}/logger/__init__.py",
    f"src/{PROJECT_DIRECTORY}/pipeline/__init__.py",
    f"src/{PROJECT_DIRECTORY}/pipeline/model_training_pipeline.py",
    f"src/{PROJECT_DIRECTORY}/pipeline/prediction_pipeline.py",
    f"src/{PROJECT_DIRECTORY}/utils/__init__.py",
    f"src/{PROJECT_DIRECTORY}/utils/utils.py",
    "app.py",
    "train.py",
    "requirements.txt",
    "setup.py",
    "template/index.html",
]

for file_path in list_of_files:
    # file_path = Path(file_path)
    file_dir, file_name = os.path.split(Path(file_path))

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) != 0):
        with open(file_path, "w", encoding="utf-8") as f:
            pass
    else:
        print("File or directory is already exists at: ", file_path)
