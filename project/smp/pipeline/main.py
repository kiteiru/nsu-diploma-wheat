from train_and_val import train_and_val_model

from utils.get_args import parse_input_args
from utils.get_args import check_args_and_init_config
from preprocessing.crop_images import crop_images
from preprocessing.augment_data import augment_data

if __name__ == "__main__":
    # logging.basicConfig(filename="program_logs.log")
    # logging.info('Program started working...')

    # args = parse_input_args()
    # config = check_args_and_init_config(args)

    # if config["DATA_BOOST"] is not None:
    #     if config["CROP_SIZE"] is not None:
    #         crop_images(config)
    #     augment_data(config)
    # model = train_and_val(config)
    model = train_and_val_model()
