# Wheat Spikelets Automatic Detection Method

## General

My graduation work is a project suggested by Institute of Cytology and Genetics of SB RAS. Its development was aimed to upgrade accuracy and efficiency of detecting and counting wheat spikelets on spike images.

Developed method was compared with [existing ones](https://github.com/kiteiru/nsu-diploma-wheat/tree/main/articles_methods) by using several mathematical and technical criteria which have had different priorities and explicitly affected final result of comparison.

Model was developed with experimental approach searching optimal hyperparameters and parts of model architecture.

Training, validationg and testing was done on separated non-intersecting sets of images with ratio 60:20:20.
Moreover, all stages of pipeline were processed on 3 different dats organisations according to clusterisation of data:
- "Certain" every cluster fully is put into one of the set;
- "Equal" every cluster is separated with the same ratio 60:20:20 and every part is put into training, validating and testing set respectively;
- "Random" every image is put into one set randomly.

There was released [final model](https://github.com/kiteiru/nsu-diploma-wheat/releases/tag/v1.0.0) that showed better results in accuracy prediction of localization spikelets in comparison to existing methods.

Also as part of the work was developed MVP of web interface that let user load picture of spikelet and run prediction mode of the model get binary mask with localized spikelets and also csv table with image name and number of found spikelets on picture.

## Main Used Technologies, Libraries and Frameworks
- PyTorch
- Segmentation Models PyTorch
- Optuna
- Numpy
- Matplotlib
- Albumentations
- Flask

## Source

You can get acquainted with full textwork, description of method development process and achieved results in details [here](https://drive.google.com/file/d/1ov_lkkyoP-X2i4P7mZV5h6Qq2OKCUpeq/view?usp=sharing).