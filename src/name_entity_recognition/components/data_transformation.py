"""This module contains all the components that are associated to Data Transformation"""

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.name_entity_recognition.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.entity.config_entity import DataTransformationConfig
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging
from src.name_entity_recognition.utils.utils import MainUtils


class DataTransformation:
    """This class encapulates all the methods that are used for data transformation"""
    def __init(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.utils = MainUtils()
        self.s3_storage = S3Operations()


    def split_data(self, df: DataFrame) -> tuple:
        """This method is used to split the data into train & test Dataframe"""
        try:
            logging.info("Inside the split_data method of \
                         src.name_entity_recognition.components.data_transformation.DataTransformation class")
            # Taking subset of data for training
            df = df[0:1000]

            labels = [i.split() for i in df["labels"].values.tolist()]
            unique_labels = set()

            for label in labels:
                for i in label:
                    if i not in unique_labels:
                        unique_labels.add(i)

            labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
            ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

            df_train, df_val, df_test = np.split(
                df.sample(frac=1, random_state=42),
                [int(0.8 * len(df)), int(0.9 * len(df))],
            )

            logging.info("Completed execution of split_data method of \
                         src.name_entity_recognition.components.data_transformation.DataTransformation class")

            return (labels_to_ids, ids_to_labels, df_train, df_val, df_test, unique_labels)
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error


    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """This method is used to initiate the data transformation"""
        try:
            logging.info("Inside the initiate_data_transformation method of \
                         src.name_entity_recognition.components.data_transformation.DataTransformation class")
            # Creating Data transformation artifacts directory
            os.makedirs(
                self.data_transformation_config.data_transformation_artifacts_dir,
                exist_ok=True,
            )
            logging.info(
                f"Created {self.data_transformation_config.data_transformation_artifacts_dir} directory."
            )

            df = pd.read_csv(self.data_ingestion_artifacts.csv_data_file_path)

            (labels_to_ids, ids_to_labels, df_train, df_val, df_test, unique_labels) = self.split_data(df=df)

            logging.info("Splitting the data")

            self.utils.dump_pickle_file(
                output_filepath=self.data_transformation_config.labels_to_ids_path,
                data=labels_to_ids,
            )
            logging.info(
                f"Saved the labels to ids pickle file to Artifacts directory.\
                    File name - {self.data_transformation_config.labels_to_ids_path}"
            )

            self.utils.dump_pickle_file(
                output_filepath=self.data_transformation_config.ids_to_labels_path,
                data=ids_to_labels,
            )
            logging.info(
                f"Saved the ids to labels pickle file to Artifacts directory.\
                    File name - {self.data_transformation_config.ids_to_labels_path}"
            )

            self.s3_storage.upload_file(
                self.data_transformation_config.ids_to_labels_path,
                self.data_transformation_config.ids_to_labels_file_name,
                self.data_transformation_config.data_storage_bucket,
                remove=False,
            )

            logging.info(
                f"Uploaded the ids to labels pickle file to AWS Storage.\
                    File name - {self.data_transformation_config.ids_to_labels_path}"
            )

            self.utils.dump_pickle_file(
                output_filepath=self.data_transformation_config.df_train_path,
                data=df_train,
            )

            logging.info(
                f"Saved the train df pickle file to Artifacts directory.\
                    File name - {self.data_transformation_config.df_train_path}"
            )

            self.utils.dump_pickle_file(
                output_filepath=self.data_transformation_config.df_val_path, 
                data=df_val
            )

            logging.info(
                f"Saved the val df pickle file to Artifacts directory.\
                    File name - {self.data_transformation_config.df_val_path}"
            )

            self.utils.dump_pickle_file(
                output_filepath=self.data_transformation_config.df_test_path,
                data=df_test,
            )
            
            logging.info(
                f"Saved the test df pickle file to Artifacts directory.\
                    File name - {self.data_transformation_config.df_test_path}"
            )

            self.utils.dump_pickle_file(
                output_filepath=self.data_transformation_config.unique_labels_path,
                data=unique_labels,
            )
            logging.info(
                f"Saved the unique labels pickle file to Artifacts directory.\
                    File name - {self.data_transformation_config.unique_labels_path}"
            )

            data_transformation_artifacts = DataTransformationArtifacts(
                labels_to_ids_path = self.data_transformation_config.labels_to_ids_path,
                ids_to_labels_path = self.data_transformation_config.ids_to_labels_path,
                df_train_path = self.data_transformation_config.df_train_path,
                df_val_path = self.data_transformation_config.df_val_path,
                df_test_path = self.data_transformation_config.df_test_path,
                unique_labels_path = self.data_transformation_config.unique_labels_path,
            )

            logging.info("Completed execution of initiate_data_transformation method of \
                         src.name_entity_recognition.components.data_transformation.DataTransformation class")
            return data_transformation_artifacts
        
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
