"""This module includes codes that defines custom Exception"""

import types


def error_message_detail(error, error_detail: types.ModuleType):
    """This function is used to"""
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occurred python script name {file_name} \
    line number {exc_tb.tb_lineno} error message {str(error)}"

    return error_message


class NERException(Exception):
    """This class encapsulated the method that returns error message"""

    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
