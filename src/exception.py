# exception.py
import sys

def get_error_details(error):
    # Yeh line error ki detail nikalti hai
    _, _, exc_tb = sys.exc_info()
    
    # Konsi file mein error tha
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Konsi line pe error tha
    line_number = exc_tb.tb_lineno
    
    # Error message banao
    error_message = "Error in file [{0}] at line [{1}]: {2}".format(
        file_name,
        line_number,
        str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message):
        # Parent class ko batao
        super().__init__(error_message)
        
        # Detail message lo
        self.error_message = get_error_details(error_message)
    
    def __str__(self):
        return self.error_message