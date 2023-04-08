import time
import logging

logger = logging.getLogger(__name__)

def measure_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        function_return = function(*args, **kwargs)
        end = time.time()

        logger.info(f"{function.__name__}: {round((end - start) / 60, 7)} minutes elapsed")
        
        return function_return
    return wrapper