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
save_maps = True

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
#%%
if __name__ == "__main__":
	start_time = time.time()
	# for yolo360 prediction	
	my_net = initialize_yolo360(path_to_yolo360)
	# Pre-processing
	orig_frame, mask_rgb, opt_map = preprocess_video(path2video, video_name)
	
	# Two-stream Model Evaluation
	predictions = evaluate_2stream_model(mask_rgb, opt_map, path_to_hdf5_model)
	
	# Post-processing
	post_processing(predictions, my_net, orig_frame, save_maps, path_to_yolo360, dest, output_filename_npz)
	
	print("%s Hybrid model finsihed in seconds:" % (time.time() - start_time))
#################END#################
#%%
#to load .npz file
# npzFiles = np.load('D:/Projects/6/saliency_model/all_in_one/output/Xinwen1_1.mp4/Xinwen1_1_coordinates.npz', allow_pickle=True)
# frame_numbers = npzFiles['frame_number']
# frame_coords = npzFiles['frame_coords']
# print(frame_coords[0])







