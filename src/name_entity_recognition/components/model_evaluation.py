"""This module contains all the components that are associated to Model Evaluation"""

import os
import sys
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from src.name_entity_recognition.components.model_trainer import DataSequence
from src.name_entity_recognition.constants import *
from src.name_entity_recognition.entity.artifact_entity import DataTransformationArtifacts, ModelTrainingArtifacts, ModelEvaluationArtifacts
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.entity.config_entity import ModelEvaluationConfig
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging
from src.name_entity_recognition.utils.utils import MainUtils


class ModelEvaluation:
    """This class encapulates all the methods that are used for Model Evaluation"""
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, 
               data_transformation_artifacts: DataTransformationArtifacts, 
               model_training_artifacts: ModelTrainingArtifacts):
        
        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_training_artifacts = model_training_artifacts
        self.utils = MainUtils()
        self.s3_storage = S3Operations()


    def evaluate(self, model: object, df_test: DataFrame) -> ModelEvaluationArtifacts:
        """This method is used to initiate the model evaluation"""
        try:
            logging.info("Inside the initiate_model_evaluation method of \
                         src.name_entity_recognition.components.data_transformation.ModelTraining class")
            
            tokenizer = self.utils.load_pickle_file(
                filepath=self.model_training_artifacts.tokenizer_file_path
            )
            logging.info("Loaded tokenizer")

            labels_to_ids = self.utils.load_pickle_file(
                filepath=self.data_transformation_artifacts.labels_to_ids_path
            )
            logging.info("labels to ids pickle file loaded")

            test_dataset = DataSequence(
                df=df_test, tokenizer=tokenizer, labels_to_ids=labels_to_ids
            )
            logging.info("Loaded test dataset for evaluation")

            test_dataloader = DataLoader(test_dataset, batch_size=1)

            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            if use_cuda:
                model = model.cuda()

            total_acc_test = 0.0

            for test_data, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_data["attention_mask"].squeeze(1).to(device)
                input_id = test_data["input_ids"].squeeze(1).to(device)
                _, logits = model(input_id, mask, test_label)

                for i in range(logits.shape[0]):
                    logits_clean = logits[i][test_label[i] != -100]
                    label_clean = test_label[i][test_label[i] != -100]

                    predictions = logits_clean.argmax(dim=1)
                    acc = (predictions == label_clean).float().mean()
                    total_acc_test += acc

            val_accuracy = total_acc_test / len(df_test)

            print(f"Test Accuracy: {val_accuracy: .3f}")

            logging.info("Completed execution of initiate_model_evaluation method of \
                         src.name_entity_recognition.components.data_transformation.ModelTraining class")

            return val_accuracy

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error


    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """This method is used to initiate the model evaluation"""
        try:
            logging.info("Inside the initiate_model_evaluation method of \
                         src.name_entity_recognition.components.data_transformation.ModelTraining class")
            
            # Creating Data Ingestion Artifacts directory inside artifacts folder
            os.makedirs(self.model_evaluation_config.model_evaluation_artifacts_dir, exist_ok=True)

            logging.info(f"Created {self.model_evaluation_config.model_evaluation_artifacts_dir} directory.")

            model = torch.load(self.model_training_artifacts.bert_model_path)
            logging.info("Loaded bert model")

            df_test = self.utils.load_pickle_file(
                filepath=self.data_transformation_artifacts.df_test_path
            )

            logging.info("Loaded Test dataset for evaluation")

            trained_model_accuracy = self.evaluate(model=model, df_test=df_test)
            logging.info(f"The accuracy on test dataset is - {trained_model_accuracy}")

            # Loading model from google container registry
            self.s3_storage.download_object(
                key = self.model_evaluation_config.aws_model_path,
                bucket_name = MODEL_BUCKET_NAME,
                filename = AWS_MODEL_NAME,
            )

            # Checking whether data file exists in the artifacts directory or not
            if os.path.exists(self.model_evaluation_config.aws_local_path) == True:
                logging.info("AWS model file available in the root directory")

                gcp_model = torch.load(self.model_evaluation_config.gcp_local_path, map_location=torch.device('cpu'))
                logging.info("AWS model loaded")

                aws_model_accuracy = self.evaluate(model=gcp_model, df_test=df_test)
                logging.info(
                    f"Calculated the AWS model's Test accuracy. - {aws_model_accuracy}"
                )

                tmp_best_model_score = (
                    0 if aws_model_accuracy is None else aws_model_accuracy
                )

            else:
                tmp_best_model_score = 0
                logging.info("AWS model is not available locally for comparison.")

            model_evaluation_artifact = ModelEvaluationArtifacts(
                trained_model_accuracy=trained_model_accuracy,
                is_model_accepted=trained_model_accuracy > tmp_best_model_score,
            )
                        
            logging.info("Completed execution of initiate_model_evaluation method of \
                         src.name_entity_recognition.components.data_transformation.ModelTraining class")

            return model_evaluation_artifact

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
