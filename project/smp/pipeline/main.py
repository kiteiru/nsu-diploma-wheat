import os
import sys
import time
import optuna
import joblib
import logging
import torch
import gc
import segmentation_models_pytorch as smp

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from torch import nn, optim
from utils.set_seed import set_seed
from pipeline.test import testing
from train_and_val import train_and_val_model
from utils.get_args import parse_input_args, check_args_and_init_config

config = {}

SEED = 197
os.environ["PYTHONHASHSEED"] = str(SEED)

def generate_timestamp():
    return str(time.strftime("%H-%M-%S:%d-%m", time.gmtime()))


def log_optimization_mode(NAME, ARGV):
    logger.info(f'{NAME}: {ARGV}')


def init_model(config):
    model_name = config["ARCHITECTURE"]
    model = None
    if model_name == "Unet":
        model = smp.Unet(encoder_name=config["ENCODER"],
                         in_channels=config["INPUT_CHANNELS"],
                         classes=config["CLASSES"],
                         activation=None).to(config["DEVICE"])
    elif model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(encoder_name=config["ENCODER"],
                                 in_channels=config["INPUT_CHANNELS"],
                                 classes=config["CLASSES"],
                                 activation=None).to(config["DEVICE"])
    elif model_name == "LinkNet":
        model = smp.Linknet(encoder_name=config["ENCODER"],
                            in_channels=config["INPUT_CHANNELS"],
                            classes=config["CLASSES"],
                            activation=None).to(config["DEVICE"])
    elif model_name == "FPN":
        model = smp.FPN(encoder_name=config["ENCODER"],
                        in_channels=config["INPUT_CHANNELS"],
                        classes=config["CLASSES"],
                        activation=None).to(config["DEVICE"])
    elif model_name == "DeepLabV3":
        model = smp.DeepLabV3(encoder_name=config["ENCODER"],
                             in_channels=config["INPUT_CHANNELS"],
                             classes=config["CLASSES"],
                             activation=None).to(config["DEVICE"])
    elif model_name == "PSPNet":
        model = smp.PSPNet(encoder_name=config["ENCODER"],
                           in_channels=config["INPUT_CHANNELS"],
                           classes=config["CLASSES"],
                           activation=None).to(config["DEVICE"])
    elif model_name == "PAN":
        model = smp.PAN(encoder_name=config["ENCODER"],
                        in_channels=config["INPUT_CHANNELS"],
                        classes=config["CLASSES"],
                        activation=None).to(config["DEVICE"])
    return model


def init_optimizator(model, config):
    optim_name = config["OPTIMIZER"]
    optimizer = None
    logger.info("")
    log_optimization_mode("LEARNING_RATE", config["LEARNING_RATE"])
    if optim_name == "Adagrad":
        log_optimization_mode("EPSILON", config["EPSILON"])

        optimizer = optim.Adagrad(params=model.parameters(),
                                  lr=config["LEARNING_RATE"],
                                #   lr_decay=config["LEARNING_RATE_DECAY"],
                                #   weight_decay=config["WEIGHT_DECAY"],
                                  eps=config["EPSILON"])
    elif optim_name == "Adam":
        log_optimization_mode("BETA_1", config["BETA_1"])
        log_optimization_mode("BETA_2", config["BETA_2"])
        log_optimization_mode("EPSILON", config["EPSILON"])

        optimizer = optim.Adam(params=model.parameters(),
                               lr=config["LEARNING_RATE"],
                               betas=(config["BETA_1"], config["BETA_2"]),
                               eps=config["EPSILON"])
                            #    weight_decay=config["WEIGHT_DECAY"])
    elif optim_name == "AdamW":
        log_optimization_mode("BETA_1", config["BETA_1"])
        log_optimization_mode("BETA_2", config["BETA_2"])
        log_optimization_mode("EPSILON", config["EPSILON"])

        optimizer = optim.AdamW(params=model.parameters(),
                                lr=config["LEARNING_RATE"],
                                betas=(config["BETA_1"], config["BETA_2"]),
                                eps=config["EPSILON"])
                                # weight_decay=config["WEIGHT_DECAY"])
    elif optim_name == "RMSprop":
        log_optimization_mode("EPSILON", config["EPSILON"])
        log_optimization_mode("MOMENTUM", config["MOMENTUM"])

        optimizer = optim.RMSprop(params=model.parameters(),
                                  lr=config["LEARNING_RATE"],
                                  eps=config["EPSILON"],
                                #   weight_decay=config["WEIGHT_DECAY"],
                                  momentum=config["MOMENTUM"])
    elif optim_name == "SGD":
        log_optimization_mode("MOMENTUM", config["MOMENTUM"])

        optimizer = optim.SGD(params=model.parameters(),
                              lr=config["LEARNING_RATE"],
                              momentum=config["MOMENTUM"])
                            #   weight_decay=config["WEIGHT_DECAY"])
    elif optim_name == "NAdam":
        log_optimization_mode("BETA_1", config["BETA_1"])
        log_optimization_mode("BETA_2", config["BETA_2"])
        log_optimization_mode("EPSILON", config["EPSILON"])
        
        optimizer = optim.NAdam(params=model.parameters(),
                                lr=config["LEARNING_RATE"],
                                betas=(config["BETA_1"], config["BETA_2"]),
                                eps=config["EPSILON"])
                                # weight_decay=config["WEIGHT_DECAY"],
                                # momentum_decay=config["MOMENTUM_DECAY"])
    return optimizer


def initialization(config):
    model = init_model(config)
    optimizer = init_optimizator(model, config)
    loss_fn = config["LOSS_FUNCTION"]
    return model, optimizer, loss_fn


def objective(trial, config):
    set_seed(SEED)

    # config["LEARNING_RATE"] = trial.suggest_float('lr', 5e-4, 1e-2)
    # config["BETA_1"] = trial.suggest_float('beta1', 0.8, 0.99)
    # config["BETA_2"] = trial.suggest_float('beta2', 0.8, 0.999)
    # config["EPSILON"] = trial.suggest_float('epsilon', 1e-10, 1e-5)
    # config["MOMENTUM"] = trial.suggest_float('momentum', 0.0, 1.0)
    
    # config["WEIGHT_DECAY"] = trial.suggest_float('weight_decay', 0.0, 1e-2)
    # config["LEARNING_RATE_DECAY"] = trial.suggest_float('lr_decay', 0.0, 1.0)
    # config["MOMENTUM_DECAY"] = trial.suggest_float('momentum_decay', 0.0, 1.0)

    config["HORIZONTAL_FLIP_PROBABILITY"] = trial.suggest_float('horizontal_flip_prob', 0.0, 1.0)
    config["VERTICAL_FLIP_PROBABILITY"] = trial.suggest_float('vertical_flip_prob', 0.0, 1.0)
    config["ROTATE_PROBABILITY"] = trial.suggest_float('rotate_prob', 0.0, 1.0)
    config["COLOR_JITTER_PROBABILITY"] = trial.suggest_float('color_jitter_prob', 0.0, 1.0)
    # config["DOWNSCALE_PROBABILITY"] = trial.suggest_float('downscale_prob', 0.0, 1.0)
    # config["ISO_NOISE_PROBABILITY"] = trial.suggest_float('iso_noise_prob', 0.0, 1.0)
    # config["GAUSSIAN_BLUR_PROBABILITY"] = trial.suggest_float('gaussian_blur_prob', 0.0, 1.0)
    # config["MOTION_BLUR_PROBABILITY"] = trial.suggest_float('motion_blur_prob', 0.0, 1.0)
    # config["RGB_SHIFT_PROBABILITY"] = trial.suggest_float('rgb_shift_prob', 0.0, 1.0)

    model, optimizer, loss_fn = initialization(config)
    
    model, optimizer, loss_fn, config, val_fscore = train_and_val_model(model, optimizer, loss_fn, config)

    test_fscore = 0
    if val_fscore != 0:
        test_fscore = testing(model, config)

    del model
    del optimizer
    del loss_fn
    gc.collect()
    torch.cuda.empty_cache()
    return test_fscore


if __name__ == "__main__":
    
    timestamp = generate_timestamp()
    log_filename = os.path.join("..", "logs", "experiment_" + timestamp + ".log")
    optuna_filename = os.path.join("..", "optuna", "optimization_" + timestamp + ".pkl")

    logging.basicConfig(filename=log_filename,
                        filemode='w',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info("Program has started running...")

    args = parse_input_args()
    config = check_args_and_init_config(config, args, timestamp)

    if config["OPTIMIZATION_MODE"]:
        func = lambda trial: objective(trial, config)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED), 
                                    direction='maximize')
        study.optimize(func, 
                       n_trials=100,
                       gc_after_trial=True)
        joblib.dump(study, optuna_filename)
    else:
        set_seed(SEED)

        model, optimizer, loss_fn = initialization(config)
    
        model, optimizer, loss_fn, config, val_fscore = train_and_val_model(model, optimizer, loss_fn, config)
        test_fscore = testing(model, config)

    logger.info("Program has finished running...")
