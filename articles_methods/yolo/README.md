# YOLO v.8

For using code firstly clone repo:

```
git clone https://github.com/ultralytics/ultralytics
```

## Labeling

There are several apps for doing handmade markup, for example web application [MakeSense](https://www.makesense.ai/)

However, I used [automatic markup](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/notebooks/select_yolo_bbox_sizes/select_bbox_width_height.ipynb) with rough approximization of spikelets sizes for each clusters according to its similarity within one

As a result for each image i got txt markup with rows of following type:

**class_label** x_center/WIDTH **y_center/HEIGHT** bbox_width/WIDTH **bbox_height/HEIGHT**


## Configuration

Then set [configuration](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/yolo/spikelets.yaml) where will be class label and paths to ONLY image directories, however for each set label directory have to be nearby image one, for example:
```
\_yolo_data
    \_images
    \_labels
```
## Running

Before using running scripts install ultralytics package:
```
pip install ultralytics
```
Then i used [script](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/articles_methods/yolo/run.sh) for training, testing model and use then for predicting

Parameters that probably need explanatory note:

* data - yaml config with your data paths and class labels
* model - type of model by its size (n, s, m, l, x) and version
* save_period - period of epochs in which model will be saved automatically
* name - name of results directory for more convinient work with experiments
* patience - if it 0 model trains till the num epochs you set, otherwise it will be checking if for "patience" epochs num there were no changes with model metrics, if yes it stops training
* conf - confidence threshold for prediction of objects
* source - path to data you want to use for inference
* save_txt - saving bbox predicted markup and labels in txt format