"""This module contains all the components that are associated to Model Training"""

import os
import sys
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast
from models.bert import BertModel
from src.name_entity_recognition.constants import *
from src.name_entity_recognition.entity.artifact_entity import DataTransformationArtifacts, ModelTrainingArtifacts
from src.name_entity_recognition.configuration.aws_storage_operations import S3Operations
from src.name_entity_recognition.entity.config_entity import ModelTrainingConfig
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging
from src.name_entity_recognition.utils.utils import MainUtils


class DataSequence(torch.utils.data.Dataset):
    """This class encapsulates the methods to form data sequences"""
    def __init__(self, df, tokenizer, labels_to_ids):

        lb = [i.split() for i in df["labels"].values.tolist()]
        txt = df["text"].values.tolist()

        self.texts = []
        for i in txt:
            self.texts.append(tokenizer(str(i), padding="max_length", max_length=512, truncation=True, return_tensors="pt"))

        self.labels = []
        for i, j in zip(txt, lb):
            self.labels.append(self.align_label(i, j, tokenizer, labels_to_ids))


    def __len__(self) -> int:
        """This method returns the length of labels"""
        return len(self.labels)

    def get_batch_data(self, idx):
        """This method returns the data for the given indexes"""
        return self.texts[idx]

    def get_batch_labels(self, idx):
        """This method returns the labels for the given indexes"""
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        """This method is used to fetch the batch details"""
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels

    def align_label(self, texts: str, labels: str, tokenizer: dict, labels_to_ids: dict) -> list:
        """This method is used to aligns label"""
        try:
            logging.info("Inside the align_label method of \
                         src.name_entity_recognition.components.model_trainer.DataSequence class")
            
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
                        label_ids.append(labels_to_ids[labels[word_idx]])
                    except:
                        label_ids.append(-100)
                else:
                    try:
                        if label_all_tokens:
                            label_ids.append(labels_to_ids[labels[word_idx]])
                        else:
                            label_ids.append(-100)
                    except:
                        label_ids.append(-100)

                previous_word_idx = word_idx

            logging.info("Completed execution of align_label method of \
                         src.name_entity_recognition.components.model_trainer.DataSequence class")
            return label_ids

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
        

class ModelTraining:
    """This class encapulates all the methods that are used for Model Training"""
    def __init(self, model_training_config: ModelTrainingConfig, data_transformation_artifacts: DataTransformationArtifacts):
        self.model_training_config = model_training_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.utils = MainUtils()
        self.s3_storage = S3Operations()


    def initiate_model_training(self) -> ModelTrainingArtifacts:
        """This method is used to initiate the model training"""
        try:
            logging.info("Inside the initiate_model_training method of \
                         src.name_entity_recognition.components.model_trainer.ModelTraining class")
            
            os.makedirs(self.model_training_config.model_training_artifacts_dir, exist_ok=True)

            logging.info(f"Created {self.model_training_config.model_training_artifacts_dir} directory.")

            tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
            
            logging.info("Downloaded tokenizer")
            
            unique_labels = self.utils.load_pickle_file(
                filepath=self.data_transformation_artifacts.unique_labels_path
            )

            logging.info(f"Loaded {self.data_transformation_artifacts.unique_labels_path} from directory")

            df_train = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.df_train_path)

            logging.info(f"Loaded {self.data_transformation_artifacts.df_train_path} from directory.")

            df_val = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.df_val_path)

            logging.info(f"Loaded {self.data_transformation_artifacts.df_val_path} from directory.")

            labels_to_ids = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.labels_to_ids_path)

            logging.info(f"Loaded {self.data_transformation_artifacts.labels_to_ids_path} from directory.")

            model = BertModel(unique_labels=unique_labels)

            logging.info("created model class for bert")

            train_dataset = DataSequence(df=df_train, tokenizer=tokenizer, labels_to_ids=labels_to_ids)

            logging.info("Created train dataset")
            
            val_dataset = DataSequence(df=df_val, tokenizer=tokenizer, labels_to_ids=labels_to_ids)

            logging.info("created val dataset")

            train_dataloader = DataLoader( train_dataset, 
                                          batch_size = self.model_training_config.model_training_batch_size, 
                                          shuffle=True)

            val_dataloader = DataLoader(val_dataset, 
                                        batch_size = self.model_training_config.model_training_batch_size)
            
            logging.info("created train and val dataloader")

            use_cuda = torch.cuda.is_available()
            
            device = torch.device("cuda" if use_cuda else "cpu")

            optimizer = SGD(model.parameters(), lr=self.model_training_config.model_training_learning_rate)
            
            logging.info("created optimizer")

            if use_cuda:
                model = model.cuda()

            best_acc = 0
            best_loss = 1000

            for epoch_num in range(self.model_training_config.model_training_epochs):

                total_acc_train = 0
                total_loss_train = 0

                for train_data, train_label in tqdm(train_dataloader):

                    train_label = train_label.to(device)
                    mask = train_data["attention_mask"].squeeze(1).to(device)
                    input_id = train_data["input_ids"].squeeze(1).to(device)

                    optimizer.zero_grad()
                    loss, logits = model(input_id, mask, train_label)

                    for i in range(logits.shape[0]):

                        logits_clean = logits[i][train_label[i] != -100]
                        label_clean = train_label[i][train_label[i] != -100]

                        predictions = logits_clean.argmax(dim=1)
                        acc = (predictions == label_clean).float().mean()
                        total_acc_train += acc
                        total_loss_train += loss.item()

                    loss.backward()
                    optimizer.step()

                model.eval()

                total_acc_val = 0
                total_loss_val = 0

                for val_data, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_data["attention_mask"].squeeze(1).to(device)
                    input_id = val_data["input_ids"].squeeze(1).to(device)

                    loss, logits = model(input_id, mask, val_label)

                    for i in range(logits.shape[0]):

                        logits_clean = logits[i][val_label[i] != -100]
                        label_clean = val_label[i][val_label[i] != -100]

                        predictions = logits_clean.argmax(dim=1)
                        acc = (predictions == label_clean).float().mean()
                        total_acc_val += acc
                        total_loss_val += loss.item()

                val_accuracy = total_acc_val / len(df_val)
                val_loss = total_loss_val / len(df_val)

                print(
                    f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {val_loss: .3f} | Accuracy: {val_accuracy: .3f}"
                )

            torch.save(model, self.model_training_config.bert_model_instance_path)
            logging.info(f"Model {self.model_training_config.bert_model_instance_path} daved")

            self.utils.dump_pickle_file(
                output_filepath=self.model_training_config.tokenizer_file_path,
                data=tokenizer,
            )

            logging.info(f"Dumped pickle file {self.model_training_config.tokenizer_file_path}")

            self.s3_storage.upload_file(
                
                from_filename= self.model_training_config.tokenizer_file_path,
                to_filename = TOKENIZER_FILE_NAME,
                bucket_name = MODEL_BUCKET_NAME,
            )
            logging.info(f"Uploaded pickle {self.model_training_config.tokenizer_file_path} to AWS")

            model_training_artifacts = ModelTrainingArtifacts(
                bert_model_path=self.model_training_config.bert_model_instance_path,
                tokenizer_file_path=self.model_training_config.tokenizer_file_path,
            )
                        
            logging.info("Completed execution of initiate_model_training method of \
                         src.name_entity_recognition.components.model_trainer.ModelTraining class")
            
            return model_training_artifacts
            
        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
