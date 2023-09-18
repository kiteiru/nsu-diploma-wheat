# Cropping Images

*cropping.ipynb* notebook lets to crop images with spikes and colorchecker to make first item being in the center of picture. Further it is resized to 384 x 384 px and ratio of original cropped size to 384 px.

*cropping.py* program file lets to augment dataset by cropping each image for *PARTS_NUM* different scaling. It gives model sustainability to different scaling of spike on crop. Program creates new coefs, ratio and data organization files. Name of every cropped file contains margin in percentage from minimal side of original picture.

TIP: Despite two outer loops iterating through ready-made data organization file, you can iterate through files in one directory using one loop:
```
for path in Path(CIRCLE_PATH).glob('*_spmk_1.0.png'):
```

Example of different scale of spike on cropped images and model prediction of its spikelets:

![alt text](Scale.png)