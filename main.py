# -*- coding: utf-8 -*-
"""
Created on Friday September 02 07:49:20 2022

@author: sanaalamgeer
"""
import numpy as np
import os

#pip install sk-video
#conda install ffmpeg -c mrinaljain17
import skvideo.io
from utils.optical_flow_v2 import calcOpticalFlow
from skimage import color
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import subprocess
import skimage.io
import skimage.color
import skimage.filters

from functions.helping_classes import *
from two_stream_model.atrous_test_random import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#%%
# If set true, it will save frames with bounding boxes.
save_maps = False

# if number of predicted rois is less than the num_rois, all rois are returned.
# based on largest areas, the rois are selected and returned.
# set it to "all" for all predicted rois.s
num_rois = 1 #1, 2, 3, "all", etc.

path_to_output = 'D:/Projects/6/saliency_model/all_in_one/output/'

path_to_hdf5_model = 'D:/Projects/6/saliency_model/all_in_one/two_stream_model/'

path_to_yolo360 = 'D:/Projects/6/saliency_model/all_in_one/yolo/'

# test video
video_name = 'Xinwen1_1.mp4'
path2video = 'input/'

#.npz file
output_filename_npz = video_name.split(".mp4")[0] + "_coordinates"

# Destination for maps and csv of coordinates
dest = path_to_output + video_name + "/"
if not os.path.exists(dest):
	os.mkdir(dest)
	
if not os.path.exists(dest+"/btmup_sal/"):
	os.mkdir(dest+"/btmup_sal/")
	
if not os.path.exists(dest+"/hybrid_sal/"):
	os.mkdir(dest+"/hybrid_sal/")
	
if not os.path.exists(dest+"/semantic_sal/"):
	os.mkdir(dest+"/semantic_sal/")
#%%
if __name__ == "__main__":
	start_time = time.time()
	# for yolo360 prediction	
	my_net = initialize_yolo360(path_to_yolo360)
	# Pre-processing
	orig_frame, mask_rgb, opt_map = preprocess_video(path2video, video_name)
	
	# Two-stream Model Evaluation
	frames_with_predictions = evaluate_2stream_model(mask_rgb, opt_map, path_to_hdf5_model)
	
	# Post-processing
	post_processing(frames_with_predictions, my_net, orig_frame, save_maps, path_to_yolo360, dest, output_filename_npz, num_rois)
	
	print("%s Hybrid model finsihed in seconds:" % (time.time() - start_time))
#################END#################
#%%
#to load .npz file
#npzFiles = np.load('D:/Projects/6/saliency_model/all_in_one/output/Xinwen1_1.mp4/Xinwen1_1_coordinates.npz', allow_pickle=True)
#btmup_sal = npzFiles['btmup_sal']
#semantic_sal = npzFiles['semantic_sal']
#hybrid_sal = npzFiles['hybrid_sal']
#print(hybrid_sal[3])
#print(btmup_sal[3])
#print(semantic_sal[3])

