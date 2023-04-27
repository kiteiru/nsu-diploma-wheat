# Boosted-Unet + MSE, Unet + WHD

In this directory two methods used the same testing and inferencing code and differenciable only with training one and several input params

## Preparation

Before training models necessary to have scale [coefficient](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/coefs.json) and [ratio](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/ratios.json) json files for further counting proper normalised distance metric

I got scale coefficients by dividing colorchecker pixel area on real one and took square root

Then i croped spikes on images, resize square crop to 384x384 px size and after that size in pixels i got before divide on 384

Also before training its necessary to have json file, for example like [that](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/splits/certain.json), where every image belongs only to one set

Every set is list of dictionaries that consists of image and binary mask path

## Training

For training model of first method - use [following code](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/boosted_net_detection_trainer.py)

For second method - [this one](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/hausdorff_net_detection_trainer.py)

Input parameters that need notes:

* splitpath - path to json with name of files and its belonging to one of the set: train, val, test
* backbone - encoder name from segmentation_models_pytorch library
* modelname - under what name model will be saved
* debugname - under what name json with metrics on each epoch will be saved
* ntree - time of architecture boosting

## Testing

For testing both methods use [code](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/detection_scorer.py)

## Inference

For inference two methods use sequence of [this](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/detection_infer.py) and [that code](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/detection_infer_with_mask.py)

Following params:

* imgpath - directory with images for inference
* outpath - directory for saving predicted outputs

NOTE: Be aware of ntree param, use the same value in testing and inference code as while training model

[There is example](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/boostedunet_mse_and_unet_whd/run.sh) of script running whole pipeline
