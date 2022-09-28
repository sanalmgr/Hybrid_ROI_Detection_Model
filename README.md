# Hybrid_ROI_Detection_Model
The code is writtent in user-friendly English language. Every function in has name according to the purpose it serves.
```
Input: .mp4 video

Output: .npz file (and/or Frames with bounding boxes if ``` save_maps = True ```)

Main script: main.py
```

The program uses the sk-video library to read video frames. You can install the library as follows:
```
> pip install sk-video
> pip install ffmpeg -c mrinaljain17
```

## Instructions to run the code:
1. Download the weights from [HERE](https://drive.google.com/file/d/1P3PTSQ-iSyp6zq_V67QbU1Gb6rm38hE9/view?usp=sharing), and:

- Add *weights/two_stream_model/sal_model_t9.hdf5* file into *Hybrid_ROI_Detection_Model/two_stream_model* folder.
- Add *weights/yolo/coco.names*, *weights/yolo/yolov3.cfg*, and *weights/yolo/yolov3.weights* files into *Hybrid_ROI_Detection_Model/yolo* folder of the project.

2. Setup the paths and parameters in *main.py* file accordingly, and run it.


## General requirements are:
```
> Python 3.9.7
> Tensorflow/  Keras  2.7.0
> OpenCV / OpenCV Contribution 4.5.5.64
> Numpy 1.19.5
```
