import os
import sys
import argparse
import logging

def check_boost_data_num(config, NAME, ARGV):
    MAX_LIMIT = 50
    if ARGV > 0:
        if ARGV < MAX_LIMIT:
            config[NAME] = ARGV
        else:
            sys.exit(f"Usage: {NAME} is less than {MAX_LIMIT}.")
        # logger.error('This is an ERROR message')
    else:
        sys.exit(f"Usage: {NAME} is positive number.")

def check_crop_size(config, NAME, ARGV):
    if ARGV > 0:
        if ARGV % 32 != 0:
            config['CROP_SIZE'] = (ARGV // 32) + 32
            print(f"Crop size {ARGV} is not divided by 32.\n"
                  f"It was changed on nearest divided: {config['CROP_SIZE']}")
        # logger.error('This is an ERROR message')
    else:
        sys.exit(f"Usage: {NAME} is positive number.")

def check_if_argv_is_positive(config, NAME, ARGV):
    if ARGV > 0:
        config[NAME] = ARGV
        # logger.error('This is an ERROR message')
    else:
        sys.exit(f"Usage: {NAME} is positive number.")

# maybe wrong
def check_if_dir_exists(config, NAME, PATH):
    if os.path.exists(PATH):
        if not os.path.isdir(PATH):
            sys.exit(f"Usage: {NAME} path have to be directory.\n"
                      "But was received path to file.")
        else:
            config[NAME] = PATH
    else:
        # create
        os.makedirs(PATH)
        config[NAME] = PATH

# TODO make ability to check that string arguments are correct, if not suggest to input new

def check_args_and_init_config(args):
    config = {}

    
    check_if_dir_exists(config, "DATA_DIR", args.datadir)
    # check_if_dir_exists(config, "CROP_DIR", args.cropdir)
    # check_if_dir_exists(config, "AUG_DIR", args.augdir)
    check_if_dir_exists(config, "SPLIT_DIR", args.splitdir)


    # maybe add ability to ask user to change crop size
    if args.crop:
        check_crop_size(config, "CROP_SIZE", args.cropsize)
    else:
        config["CROP_SIZE"] = None

    if args.aug:
        check_boost_data_num(config, "DATA_BOOST", args.databoost)
    else:
        config["DATA_BOOST"] = None

    config["DATA_JSON"] = "../data_organization/" + args.dataorg + ".json"

    check_if_argv_is_positive(config, "EPOCHS", args.epoch)
    check_if_argv_is_positive(config, "BATCH_SIZE", args.batchsize)

    # check on encoders
    config["ENCODER"] = args.encoder

    # check on optimizer
    config["OPTIMIZER"] = args.optimizer

    # check on loss func
    config["LOSS_FUNC"] = args.lossfunc

    config["MODEL_PATH"] = args.savepath

    return config

def parse_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datadir", "-dd", type=str, default="../data/augmented", help="dir with data")
    # parser.add_argument("--cropdir", "-cd", type=str, default=None, help="dir with cropped data")
    # parser.add_argument("--augdir", "-ad", type=str, default="../data/augmented", help="dir with augmented data")
    parser.add_argument("--splitdir", "-sd", type=str, default="../data/splited", help="dir with splited data by set: train, val and test")

    parser.add_argument("--crop", "-c", type=bool, default=False, help="is needed to do image cropping")
    parser.add_argument("--aug", "-a", type=bool, default=False, help="is needed to do image augmentation")

    parser.add_argument("--cropsize", "-cs", type=int, default=384, help="image and mask crop size")
    parser.add_argument("--databoost", "-bst", type=int, default=2, help="image and mask number will be boosted by augmentation")

    parser.add_argument("--dataorg", "-org", type=str, default="equal", choices=["random", "equal", "certain"], help="data organization type")

    parser.add_argument("--epoch", "-e", type=int, default=2, help="training epoch num")
    parser.add_argument("--batchsize", "-bs", type=int, default=32, help="batch size")
    parser.add_argument("--encoder", "-en", type=str, default="efficientnet-b2", help="encoder name")
    parser.add_argument("--optimizer", "-op", type=str, default="adam", help="optimizer name")
    parser.add_argument("--lossfunc", "-lf", type=str, default="binary_crossentropy", help="loss function")

    parser.add_argument("--savepath", "-sp", type=str, default="../saved_models", help="path for saving model")
    
    args = parser.parse_args()
    
    return args
