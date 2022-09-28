# Hybrid_ROI_Detection_Model
Input: .mp4 video

Output: .npz file (and/or Frames with bounding boxes if ``` save_maps = True ```)

Main script: main.py

The program uses the sk-video library to read video frames. You can install the library as follows:
```
> pip install sk-video
> pip install ffmpeg -c mrinaljain17
```

General requirements are:
```
> Python 3.9.7
> Tensorflow/  Keras  2.7.0
> OpenCV / OpenCV Contribution 4.5.5.64
> Numpy 1.19.5
```
