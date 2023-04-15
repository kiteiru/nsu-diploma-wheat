import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# from train_and_val import train_and_val_model

from utils.get_args import parse_input_args
from utils.get_args import check_args_and_init_config
from preprocessing.crop_images import crop_images
from preprocessing.augment_data import augment_data

logging.basicConfig(filename="program.log",
                    filemode='w',
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logger.info("Program has started running...")

    args = parse_input_args()
    config = check_args_and_init_config(args)

    # model = train_and_val(config)
    # model = train_and_val_model()

    logger.info("Program has finished running...")
