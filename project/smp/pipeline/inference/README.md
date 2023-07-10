## Steps for Running Inference

1. Install packages from full [*requirements.txt*](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/requirements.txt) or [*minimal_requirements.txt*](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/project/smp/pipeline/inference/minimal_requirements.txt) file:

    ```
    pip install -r requirements.txt 
    ```
    or
    ```
    pip install -r minimal_requirements.txt 
    ```


2. Make cofficient file according to [that example](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/notebooks/approximate_colorchecker/coefs.json) by [segmentation of colorchecker](https://github.com/kiteiru/nsu-diploma-wheat/tree/main/notebooks/colorchecker_segmentation) on every image and [counting its area](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/notebooks/approximate_colorchecker/approx.ipynb)
3. Prepare your images by cropping and resizing it to 384 x 384 px via [this notebook](https://github.com/kiteiru/nsu-diploma-wheat/blob/main/notebooks/images_cropping/cropping.ipynb)
4. Create folder for input images nearby *inference.py* script and move prepared images to this folder, let it call **inference_input**
5. Create folder for output images nearby *inference.py* script, let it call **inference_output**
6. Put [final binary model](https://github.com/kiteiru/nsu-diploma-wheat/releases/tag/v1.0.0) nearby *inference.py* script, let it call **circles.pt**
7. Run script with default or specified arguments, for example:

    ```
    python inference.py -in inference_input -model circles.pt -out inference_output
    ```