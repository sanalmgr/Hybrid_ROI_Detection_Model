# Hybrid_ROI_Detection_Model

Please refer to this [PDF: Deep Hybrid Model for Region of Interest
Detection in Omnidirectional Videos](https://github.com/sanalmgr/Hybrid_ROI_Detection_Model/blob/main/Individual_Report.pdf) to know more about this project and the proposed method.

The code is writtent in user-friendly English language. Every function in has name according to the purpose it serves.
```
Input: .mp4 video

Output: .npz file (and/or Frames with bounding boxes if "save_maps = True")

Main script: main.py
```

The program uses the sk-video library to read video frames. You can install the library as follows:
```
> pip install sk-video
> pip install ffmpeg -c mrinaljain17
```

## Instructions to run the code:
1. Download the weights from [HERE](https://drive.google.com/file/d/1P3PTSQ-iSyp6zq_V67QbU1Gb6rm38hE9/view?usp=sharing), and:

- Add ```weights/two_stream_model/sal_model_t9.hdf5``` file into ```Hybrid_ROI_Detection_Model/two_stream_model``` folder.
- Add ```weights/yolo/coco.names```, ```weights/yolo/yolov3.cfg```, and ```weights/yolo/yolov3.weights``` files into ```Hybrid_ROI_Detection_Model/yolo``` folder of the project.

2. Setup the paths and parameters in ```main.py``` file accordingly, and run it.

## Points to note: 
Remeber that, if no object is found under a specific euclidean distance, then an empty set with 0 bounding boxes is returned. In this case, we give priority to the predictions of Multi-projection YOLOV3 (MPYOLO) model. But, it is worth demonstrating the difference in final performance by changing the priorities.

In ```/functions/helping_classes.py``` file, the function ```refine_coords_as_frames_with_bbox``` sets the priority. In this function, ```obj_bbox```, ```bbox```, and ```coords_final``` represent the MPYOLO predictions, bottom-up saliency predictions, and the final predictions after computing euclidean distance, respectively.


## General requirements are:
```
> Python 3.9.7
> Tensorflow / Keras  2.7.0
> OpenCV / OpenCV Contribution 4.5.5.64
> Numpy 1.19.5
```
