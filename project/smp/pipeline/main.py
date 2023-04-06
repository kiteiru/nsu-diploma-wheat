import os
import sys
import time
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from utils.set_seed import set_seed
from pipeline.test import testing
from train_and_val import train_and_val_model
from utils.get_args import parse_input_args, check_args_and_init_config

def generate_timestamp():
    return str(time.strftime("%H-%M-%S:%d-%m", time.gmtime()))

if __name__ == "__main__":

    set_seed()
    
    timestamp = generate_timestamp()
    logfile_name = os.path.join("..", "logs", "experiment_" + timestamp + ".log")

    logging.basicConfig(filename=logfile_name,
                        filemode='w',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info("Program has started running...")

    config = {}

    args = parse_input_args()
    config = check_args_and_init_config(config, args, timestamp)

    config = train_and_val_model(config)
    testing(config)

    logger.info("Program has finished running...")
