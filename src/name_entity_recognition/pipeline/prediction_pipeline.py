"""This module includes all the methods for Prediction Pipeline"""

import os
import sys
import torch
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.constants import *
from src.name_entity_recognition.entity.config_entity import ModelPredictorConfig
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging
from src.name_entity_recognition.utils.utils import MainUtils


class ModelPredictor:
    """This class encapsulates the methods for prediction pipeline"""
    def __init__(self):
        self.model_predictor_config = ModelPredictorConfig()
        self.utils = MainUtils()
        self.s3_storage = S3Operations()


    def align_word_ids(self, texts: str, tokenizer: dict) ->list:
        """This methods is used for aligning word ids"""
        try:
            logging.info("Inside the align_word_ids method of \
                         src.name_entity_recognition.pipeline.prediction_pipeline.ModelPredictor class")
            
            label_all_tokens = False

            tokenized_inputs = tokenizer(
                texts, padding="max_length", max_length=512, truncation=True
            )

            word_ids = tokenized_inputs.word_ids()

            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(1)
                    except:
                        label_ids.append(-100)
                else:
                    try:
                        label_ids.append(1 if label_all_tokens else -100)
                    except:
                        label_ids.append(-100)

                previous_word_idx = word_idx

            logging.info("Completed execution of align_word_ids method of \
                         src.name_entity_recognition.pipeline.prediction_pipeline.ModelPredictor class")
            
            return label_ids

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

    
    def evaluate_one_text(self, model: object, sentence: str, tokenizer: dict, ids_to_labels: dict) -> str:
        """This model is to evaluate one texts for Prediction"""
        try:
            logging.info("Inside the evaluate_one_text method of \
                         src.name_entity_recognition.pipeline.prediction_pipeline.ModelPredictor class")

            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            if use_cuda:
                model = model.cuda()

            text = tokenizer(
                sentence,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )

            mask = text["attention_mask"].to(device)
            input_id = text["input_ids"].to(device)
            label_ids = (
                torch.Tensor(self.align_word_ids(sentence, tokenizer))
                .unsqueeze(0)
                .to(device)
            )

            logits = model(input_id, mask, None)
            logits_clean = logits[0][label_ids != -100]

            predictions = logits_clean.argmax(dim=1).tolist()
            prediction_label = [ids_to_labels[i] for i in predictions]

            logging.info("Completed execution of align_word_ids method of \
                         src.name_entity_recognition.pipeline.prediction_pipeline.ModelPredictor class")
            
            return sentence, prediction_label
        
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

    def initiate_model_predictor(self, sentence: str) -> str:
        """This model is to initiate model predictor for texts for Prediction"""
        try:
            logging.info("Inside the initiate_model_predictor method of \
                         src.name_entity_recognition.pipeline.prediction_pipeline.ModelPredictor class")

            os.makedirs(self.model_predictor_config.best_model_dir, exist_ok=True)

            logging.info(f"Created {self.model_predictor_config.best_model_dir} directory.")

            self.s3_storage.download_object(
                key = TOKENIZER_FILE_NAME,
                bucket_name = MODEL_BUCKET_NAME,
                filename = self.model_predictor_config.tokenizer_local_path,
            )
            logging.info("Tokenizer Pickle file downloaded from google storage")

            self.s3_storage.download_object(
                key = IDS_TO_LABELS_FILE_NAME,
                bucket_name = DATA_BUCKET_NAME,
                filename = self.model_predictor_config.ids_to_labels_local_path
            )
            logging.info("ids to label Pickle file downloaded from google storage")

            self.s3_storage.download_object(
                key = AWS_MODEL_NAME, 
                bucket_name = MODEL_BUCKET_NAME,
                filename = self.model_predictor_config.best_model_path,
            )
            logging.info("Downloaded best model to Best_model directory.")

            tokenizer = self.utils.load_pickle_file(
                filepath=self.model_predictor_config.tokenizer_local_path
            )
            logging.info("Loaded tokenizer object")

            ids_to_labels = self.utils.load_pickle_file(
                filepath=self.model_predictor_config.ids_to_labels_local_path
            )
            logging.info("Loaded ids to lables file.")

            model = torch.load(self.model_predictor_config.best_model_path, map_location=torch.device('cpu'))
            logging.info("Best model loaded for prediction.")

            sentence, prediction_label = self.evaluate_one_text(
                model=model,
                sentence=sentence,
                tokenizer=tokenizer,
                ids_to_labels=ids_to_labels,
            )

            logging.info("Completed execution of initiate_model_predictor method of \
                         src.name_entity_recognition.pipeline.prediction_pipeline.ModelPredictor class")
            
            return sentence, prediction_label
        
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error