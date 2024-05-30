"""This module stores the Bert Models Details"""

import sys
import torch
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging
from transformers import BertForTokenClassification


class BertModel(torch.nn.Module):
    """This encapsulates the function for training BERT Model """
    def __init__(self, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            "bert-base-cased", num_labels=len(unique_labels)
        )

    def forward(self, input_id, mask, label):
        """This method is used to invoke BERT Forward layer"""
        try:
            output = self.bert(
                input_ids=input_id, attention_mask=mask, labels=label, return_dict=False
            )

            return output

        except Exception as error:
            logging.error(error)
            raise NERException(error, sys) from error
