"""This module consists of all important utilities for Name Entity Recognition Project"""
import sys
from typing import Dict
import pickle
import dill
import numpy as np
import yaml
from src.name_entity_recognition.constants import *
from src.name_entity_recognition.exception import NERException
from src.name_entity_recognition.logger import logging

# initiatlizing logging


class MainUtils:
    """
    This class encapsulates the methods to save or load a particular file in 
    spcific format
    """
    def read_yaml_file(self, filename: str) -> Dict:
        """This method is used for reading the YAML Configuration file"""
        logging.info("Entered the read_yaml_file method of MainUtils class")
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def dump_pickle_file(output_filepath: str, data) -> None:
        """This method is used for saving the objects in pickle format"""
        try:
            with open(output_filepath, "wb") as encoded_pickle:
                pickle.dump(data, encoded_pickle)

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def load_pickle_file(filepath: str) -> object:
        """This method is used for loading the objects that is saved in pickle format"""
        try:
            with open(filepath, "rb") as pickle_obj:
                obj = pickle.load(pickle_obj)
            return obj

        except Exception as e:
            raise NERException(e, sys) from e

    def save_numpy_array_data(self, file_path: str, array: np.array) -> str:
        """This method is used for saving the data in numpy array format"""
        logging.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logging.info("Exited the save_numpy_array_data method of MainUtils class")
            return file_path

        except Exception as e:
            raise NERException(e, sys) from e

    def load_numpy_array_data(self, file_path: str) -> np.array:
        """This method is used for loading the saved data in numpy array format"""
        logging.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """This method is used for saving the file in dill format"""
        logging.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def load_object(file_path: str) -> object:
        """This method is used for loading the object that is saved in dill format"""
        logging.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)
            logging.info("Exited the load_object method of MainUtils class")
            return obj

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def read_txt_file(file_path: str) -> str:
        """This method is used for loading the file in text format"""
        logging.info("Entered the read_txt_file method of MainUtils class")
        try:
            # Opening file for read only
            file1 = open(file_path, "r", encoding="utf8")
            # read all text
            text = file1.readlines()
            # close the file
            file1.close()
            logging.info("Exited the read_txt_file method of MainUtils class")
            return text

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def save_descriptions(descriptions, filename) -> None:
        """This method is used for saving the descriptions in the file and return text format"""
        try:
            lines = list()
            for key, desc_list in descriptions.items():
                for desc in desc_list:
                    lines.append(key + " " + desc)
            data = "\n".join(lines)
            file1 = open(filename, "w")
            file1.write(data)
            file1.close()
            return filename

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def save_txt_file(output_file_path: str, data: list) -> str:
        """This method is used for saving text file """
        try:
            with open(output_file_path, "w") as file:
                file.writelines("% s\n" % line for line in data)

            return output_file_path

        except Exception as e:
            raise NERException(e, sys) from e

    @staticmethod
    def max_length_desc(descriptions: dict) -> int:
        """This method is used for getting the maximum length of texts"""
        try:
            all_desc = list()
            for key in descriptions.keys():
                for d in descriptions[key]:
                    all_desc.append(d)
            return max(len(d.split()) for d in all_desc)

        except Exception as e:
            raise NERException(e, sys) from e
