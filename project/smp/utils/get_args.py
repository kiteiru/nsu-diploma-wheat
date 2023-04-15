import json
import os
import sys
import torch
import argparse
import logging
import segmentation_models_pytorch as smp

from pathlib import Path
from torch import nn, optim

from utils.exit_from_program import exit_from_program

logger = logging.getLogger(__name__)

def check_if_dir_exists(config, NAME, PATH):
    if os.path.exists(PATH):
        if not os.path.isdir(PATH):
            logger.error(f'Usage: {NAME} has to be directory path but was received path to the file.')
            exit_from_program()
        else:
            config[NAME] = PATH
            logger.info(f'{NAME} = "{PATH}"')
    else:
        logger.error(f'Directory "{PATH}" does not exist.')
        exit_from_program()

def check_input_data_dir(config, NAME, PATH, GEOMETRY):
    check_if_dir_exists(config, NAME, PATH)
    check_if_dir_exists(config, "IMAGES_PATH", os.path.join(PATH, "images"))
    check_if_dir_exists(config, "MASKS_PATH", os.path.join(PATH, GEOMETRY))

def check_crop_size(config, NAME, ARGV):
    if ARGV > 0:
        if ARGV % 32 != 0:
            config[NAME] = (ARGV // 32) * 32
            logger.warning(f'Crop size {ARGV} is not divided by 32. It was changed to nearest divided: {config[NAME]}')
    else:
        logger.error(f'Usage: {NAME} has to be a positive number.')
        exit_from_program()


def check_if_json_file_exists(config, NAME, PATH):
    if os.path.exists(PATH):
        if not os.path.isfile(PATH):
            logger.error(f'Usage: {NAME} has to be filepath but was received path to the directory.')
            exit_from_program()
        else:
            if Path(PATH).suffix == ".json":
                with open(PATH, 'r') as f:
                    config[NAME] = json.load(f)
                logger.info(f'{NAME}_PATH = "{PATH}"')
            else:
                logger.error(f'Usage: {NAME}_PATH has to be json file.')
                exit_from_program()
    else:
        logger.error(f'File "{PATH}" does not exist.')
        exit_from_program()

def set_data_organization_json(config, NAME, PATH, ORG):
    if ORG == "other":
        check_if_json_file_exists(config, NAME, PATH)
    else:
        PATH = os.path.join("../data_organization/", ORG + ".json")
        with open(PATH, 'r') as f:
            config[NAME] = json.load(f)
        logger.info(f'{NAME}_PATH = "{PATH}"')

def check_if_argv_is_positive(config, NAME, ARGV):
    if ARGV > 0:
        config[NAME] = ARGV
        logger.info(f'{NAME} = {ARGV}')
    else:
        logger.error(f'Usage: {NAME} has to be a positive number.')
        exit_from_program()

def set_device(config, NAME, ARGV):
    if ARGV == "gpu" and torch.cuda.is_available():
        config[NAME] = "cuda"
    else:
        config[NAME] = ARGV
    logger.info(f'{NAME}: {ARGV}')

def check_if_encoder_is_available(config, NAME, ARGV):
    # TODO as class not just list
    encoders = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'densenet121', 'densenet169', 'densenet201', 'densenet161',
                'vgg11', 'vgg13', 'vgg16', 'vgg19',
                'mobilenet_v2']
    
    if ARGV in encoders:
        config[NAME] = ARGV
        logger.info(f'{NAME}: {ARGV}')
    else:
        logger.error(f'Encoder "{ARGV}" does not exist or not available in this program.')
        logger.warning(f'Available encoders: {encoders}')
        exit_from_program()

def check_if_architecture_is_available(config, NAME, ARGV):
    # TODO as class not just dictionary
    archs = {"unet": smp.Unet(encoder_name=config["ENCODER"],
                              in_channels=config["INPUT_CHANNELS"],
                              classes=config["CLASSES"],
                              activation=None)
            }
    
    if ARGV in archs:
        config[NAME] = archs[ARGV].to(config["DEVICE"])
        logger.info(f'{NAME}: {archs[ARGV].__class__.__name__}')
    else:
        logger.error(f'Architecture "{ARGV}" does not exist or not available in this program.')
        logger.warning(f'Available architectures: {list(archs.keys())}')
        exit_from_program()

def check_if_optimizer_is_available(config, NAME, ARGV):
    # optim_args = [config["ARCHITECTURE"].parameters(), config["LEARNING_RATE"]]
    
    # TODO as class not just dictionary
    # TODO figure out why doesnt work with *optim_args
    optims = {"adam": optim.Adam(config["ARCHITECTURE"].parameters(), config["LEARNING_RATE"]),
              "adamw": optim.AdamW(config["ARCHITECTURE"].parameters(), config["LEARNING_RATE"]),
              "adagrad": optim.Adagrad(config["ARCHITECTURE"].parameters(), config["LEARNING_RATE"]),
              "sgd": optim.SGD(config["ARCHITECTURE"].parameters(), config["LEARNING_RATE"]), 
              "rmsprop": optim.RMSprop(config["ARCHITECTURE"].parameters(), config["LEARNING_RATE"])
             }
    
    if ARGV in optims:
        config[NAME] = optims[ARGV]
        logger.info(f'{NAME}: {optims[ARGV].__class__.__name__}')
    else:
        logger.error(f'Optimizer "{ARGV}" does not exist or not available in this program.')
        logger.warning(f'Available optimizers: {list(optims.keys())}')
        exit_from_program()

def check_if_loss_function_is_available(config, NAME, ARGV):
    # TODO as class not just dictionary
    loss_funcs = {"bce_with_logits": nn.BCEWithLogitsLoss(),
                  "bce": nn.BCELoss(),
                  "crossentropy": nn.CrossEntropyLoss(),
                  "mse": nn.MSELoss()
                  }
    if ARGV in loss_funcs:
        config[NAME] = loss_funcs[ARGV]
        logger.info(f'{NAME}: {loss_funcs[ARGV].__class__.__name__}')
    else:
        logger.error(f'Loss function "{ARGV}" does not exist or not available in this program.')
        logger.warning(f'Available loss functions: {list(loss_funcs.keys())}')
        exit_from_program()


def check_args_and_init_config(args):
    # TODO config as class not just dictionary
    config = {}

    check_input_data_dir(config, "INPUT_DATA_PATH", args.inputdata, args.geometry)

    check_crop_size(config, "CROP_SIZE", args.cropsize)

    set_data_organization_json(config, "DATA_ORG", args.dopath, args.dataorg)

    check_if_json_file_exists(config, "COEFS", args.coefs)
    check_if_json_file_exists(config, "RATIOS", args.ratios)

    check_if_argv_is_positive(config, "EPOCHS", args.epoch)
    check_if_argv_is_positive(config, "BATCH_SIZE", args.batchsize)
    check_if_argv_is_positive(config, "LEARNING_RATE", args.learningrate)

    check_if_argv_is_positive(config, "INPUT_CHANNELS", args.inchannels)
    check_if_argv_is_positive(config, "CLASSES", args.outclasses)

    set_device(config, "DEVICE", args.device)

    check_if_encoder_is_available(config, "ENCODER", args.encoder)
    check_if_architecture_is_available(config, "ARCHITECTURE", args.architecture)
    check_if_optimizer_is_available(config, "OPTIMIZER", args.optimizer)
    check_if_loss_function_is_available(config, "LOSS_FUNCTION", args.lossfunc)

    check_if_dir_exists(config, "SAVE_MODEL_PATH", args.savemodel)
    check_if_dir_exists(config, "SAVE_OUTPUT_PATH", args.saveoutput)

    return config

def parse_input_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--inputdata", "-in", type=str, default="../../../cropped_384", help="directory path with cropped data")
    parser.add_argument("--geometry", "-g", type=str, default="circles", choices=["circles", "ellipses"], help="type of markup")

    parser.add_argument("--cropsize", "-cs", type=int, default=384, help="image and mask crop size")

    parser.add_argument("--dataorg", "-org", type=str, default="equal", choices=["random", "equal", "certain", "other"], help="data organization type")
    parser.add_argument("--dopath", "-dopath", type=str, default="", help='data organization path if was chosen "other" option')

    parser.add_argument("--coefs", "-coef", type=str, default="../scale_jsons/coefs.json", help='filepath to scale coefficients')
    parser.add_argument("--ratios", "-ratios", type=str, default="../scale_jsons/ratios.json", help='file path to crop resizing ratios')

    parser.add_argument("--epoch", "-e", type=int, default=2, help="training epoch num")
    parser.add_argument("--batchsize", "-bs", type=int, default=8, help="batch size")
    parser.add_argument("--learningrate", "-lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--inchannels", "-inch", type=int, default=3, help="channel num of input image")
    parser.add_argument("--outclasses", "-cls", type=int, default=1, help="classes num of model output")

    parser.add_argument("--device", "-dev", type=str, default="gpu", choices=["cpu", "gpu"], help="on which device model will be trained")

    parser.add_argument("--architecture", "-arch", type=str, default="unet", help="architecture name")
    parser.add_argument("--encoder", "-en", type=str, default="efficientnet-b3", help="encoder name")
    parser.add_argument("--optimizer", "-op", type=str, default="adam", help="optimizer name")
    parser.add_argument("--lossfunc", "-lf", type=str, default="bce_with_logits", help="loss function name")

    parser.add_argument("--savemodel", "-sm", type=str, default="../tmp_models", help="directory path for saving model")
    parser.add_argument("--saveoutput", "-so", type=str, default="../tmp_results", help="directory path for saving model outputs")
    
    args = parser.parse_args()
    
    return args
