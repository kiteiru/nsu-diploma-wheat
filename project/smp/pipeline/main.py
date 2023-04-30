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


def objective(trial, config):
    set_seed(SEED)

    model = smp.Unet(encoder_name=config["ENCODER"],
                     in_channels=config["INPUT_CHANNELS"],
                     classes=config["CLASSES"],
                     activation=None)

    config["LEARNING_RATE"] = trial.suggest_float('lr', 5e-4, 1e-2)
    config["BETA_1"] = trial.suggest_float('beta1', 0.8, 0.99)
    config["BETA_2"] = trial.suggest_float('beta2', 0.8, 0.999)
    config["EPSILON"] = trial.suggest_float('epsilon', 1e-10, 1e-5)
    # config["WEIGHT_DECAY"] = trial.suggest_float('weight_decay', 0.0, 1e-2)
    # config["LEARNING_RATE_DECAY"] = trial.suggest_float('lr_decay', 0.0, 1.0)
    config["MOMENTUM"] = trial.suggest_float('momentum', 0.0, 1.0)
    # config["MOMENTUM_DECAY"] = trial.suggest_float('momentum_decay', 0.0, 1.0)
    
    model, config, val_fscore = train_and_val_model(model, config)

    test_fscore = 0
    if val_fscore != 0:
        test_fscore = testing(model, config)

    del model
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
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=197), 
                                    direction='maximize')
        study.optimize(func, 
                       n_trials=3,
                       gc_after_trial=True)
        joblib.dump(study, optuna_filename)
    else:
        set_seed(SEED)

        model = smp.Unet(encoder_name=config["ENCODER"],
                         in_channels=config["INPUT_CHANNELS"],
                         classes=config["CLASSES"],
                         activation=None)

        model, config, val_fscore = train_and_val_model(model, config)
        test_fscore = testing(model, config)

    logger.info("Program has finished running...")
