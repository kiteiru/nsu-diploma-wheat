from torch.utils.data import DataLoader
from preprocessing.spikelets_dataset import SpikeletsDataset


def create_dataloaders(config, set_name, transformations):
    dataset = SpikeletsDataset(config["DATA_ORG"], config["IMAGES_PATH"], config["MASKS_PATH"], set_name, transformations)
    dataloader = DataLoader(dataset, config["BATCH_SIZE"], shuffle=False)
    return dataloader