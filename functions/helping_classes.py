import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
import cv2
import os
from os import listdir
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
from skimage.morphology import disk, opening 
from scipy.spatial import distance
from yolo.detection import *
from yolo.stereo import *

#%%
orig_width, orig_height = 1080, 540

#%%
def create_masked_images(img):
	gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #skimage.color.rgb2gray(img)
	# blur the image to denoise
	blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)

	# create a mask based on the threshold
	t = 0.9 # lower the value, more thresholded area will be
	binary_mask = blurred_image < t

	selection_gray = gray_image.copy()
	selection_gray[~binary_mask] = 0
	
	selection_rgb = img.copy()
	selection_rgb[~binary_mask] = 0
	
	return selection_gray, selection_rgb
	
def extract_frames_of_videos(input_path, video):
	print("Extracting frames from", video)
	#read video to create frames-generator
	videogen = skvideo.io.vread(input_path + video)
	print(f'Shape of video {video} = {videogen.shape}')
	
	frames = []
	for img in videogen[:]:
		a =  Image.fromarray(np.uint8(img))
		a = a.resize((orig_width, orig_height), resample=Image.Resampling.BICUBIC)
		frames.append(a)
		
	return frames

def preprocess_video(input_path, video):
	frames = extract_frames_of_videos(input_path, video)
	
	width, height = 200, 200

	### prepare features
	orig_frame = []
	mask_rgb = []
	opt_map = []

	#now read images and prepare outputs
	prev_frame = frames[0]
	prev_frame = prev_frame.resize((width, height), resample=Image.Resampling.BICUBIC)
	prev_gray, prev_rgb = create_masked_images(np.asarray(prev_frame))
	hsv = np.zeros_like(prev_rgb)
	
	#saving features of the first frame
	motion_sal = calcOpticalFlow(hsv, prev_rgb, prev_rgb)
	optflows = motion_sal.reshape(motion_sal.shape[0], motion_sal.shape[1], 1)
	
	opt_map.append(np.float32(optflows) / 255.0) 
	mask_rgb.append(np.float32(prev_rgb) / 255.0)
	orig_frame.append(np.asarray(frames[0]))
	
	for img in range(len(frames))[1:]:
			##Optical FLow Block
			current_frame = frames[img] #img
			current_frame = current_frame.resize((width, height), resample=Image.Resampling.BICUBIC)
			for_opt_map = np.asarray(current_frame)
			
			#Masked image - threshold
			gray, rgb = create_masked_images(for_opt_map)
			masked_rgb = rgb
			for_opt = rgb
	# 		
			motion_sal = calcOpticalFlow(hsv, prev_rgb, for_opt)
			optflows = motion_sal.reshape(motion_sal.shape[0], motion_sal.shape[1], 1)
			
			current_frame = np.float32(current_frame) / 255.0
			masked_rgb = np.float32(masked_rgb) / 255.0
			optflows = np.float32(optflows) / 255.0

			opt_map.append(optflows) 
			mask_rgb.append(masked_rgb)
			orig_frame.append(np.asarray(frames[img]))
			
			prev_rgb = for_opt
			
	opt_map = np.asarray(opt_map)
	mask_rgb = np.asarray(mask_rgb)
	orig_frame = np.asarray(orig_frame)
				
	print("orig_frame:", orig_frame.shape, "\nmask_rgb:", mask_rgb.shape, "\nopt_map:", opt_map.shape)
	
	return orig_frame, mask_rgb, opt_map
	
def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

def combine_boxes(boxes):
	noIntersectLoop = False
	noIntersectMain = False
	posIndex = 0
	# keep looping until we have completed a full pass over each rectangle
	# and checked it does not overlap with any other rectangle
	while noIntersectMain == False:
		noIntersectMain = True
		posIndex = 0
         # start with the first rectangle in the list, once the first 
         # rectangle has been unioned with every other rectangle,
         # repeat for the second until done
		while posIndex < len(boxes):
			noIntersectLoop = False
			while noIntersectLoop == False and len(boxes) > 1:
				a = boxes[posIndex]
				listBoxes = np.delete(boxes, posIndex, 0)
				index = 0
				for b in listBoxes:
					#if there is an intersection, the boxes overlap
					if a.any() and b.any(): #intersection(a, b):
						newBox = union(a,b)
						listBoxes[index] = newBox
						boxes = listBoxes
						noIntersectLoop = False
						noIntersectMain = False
						index = index + 1
						break
					noIntersectLoop = True
					index = index + 1
			posIndex = posIndex + 1

	return boxes.astype("int")

def get_a_number_of_bbox(bbox, area, number):
	ind_array = []
	
	if number == 1:
		indexx = area.index(max(area)) # take index of max element in list
		ind_array.append(bbox[indexx])
	else:
		for i in range(0, number):
			indexx = area.index(max(area)) # take index of max element in list
			ind_array.append(bbox[indexx])
			
			bbox.pop(indexx)
			area.pop(indexx)

	return ind_array

def get_bbox_from_prediction(thresh):
	thresh22 = cv2.threshold(thresh, (50*thresh.max()//100), thresh.max(), cv2.THRESH_BINARY)[1]

	cnts = cv2.findContours(thresh22, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	
	bbox = []
	area = []	
	if len(cnts) == 1:
		x,y,w,h = cv2.boundingRect(cnts[0])
		arr = w*h
		bbox.append([x,y,w,h])
		area.append(arr)
	else:
		for c in cnts:
			x,y,w,h = cv2.boundingRect(c)
			arr = w*h
# 			bbox.append([x,y,w,h])
# 			area.append(arr)
			if arr > 1000:
				bbox.append([x,y,w,h])
				area.append(arr)
			else:
				pass
	
	return bbox, area

def initialize_yolo360(path_to_yolo360):
	# yolov3 for 360-degree images
	my_net = Yolo()

	#general yolo v3
	# Load names of classes and get random colors
	classes = open(path_to_yolo360+'coco.names').read().strip().split('\n')
	np.random.seed(42)
	colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

	# Give the configuration and weight files for the model and load the network.
	net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

	# determine the output layer
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	
	return my_net

def get_detected_objects_yolo360(img, my_net, path_to_yolo360):
	
	projections = pano2stereo(img)
	output_frame, bboxes, area_yolo = my_net.process_output(img, projections)
	
	return output_frame, bboxes, area_yolo	
	
	
def compute_distance_and_merge_groups(bbox, obj_bbox):
	coords_final = []
	#iou_semi = 0
	for z in range(len(bbox)):
		coord = []
		for j in range(len(obj_bbox)):
			#distance.euclidean(bbox[z][:2], obj_bbox[j][:2])
			dist_out = distance.euclidean(bbox[z], obj_bbox[j])
			#print(dist_out)
			#print(bbox[z], obj_bbox[j])
			if dist_out < 150.0: #160, 180, 190, 200
				coord.append(bbox[z])
				coord.append(obj_bbox[j])				
		coord = np.asarray(coord)
		if coord.size > 1:
			coords_final.append(combine_boxes(coord)[0])
			
	return coords_final

def write_frame_with_bbox(output_path, count, orig):
	dest = output_path + 'frame_' + str(count) + '.jpg'
	imageio.imwrite(dest, orig)

def get_area_of_boxes(boxes):
	#print(boxes)
	area = []
	if len(boxes) > 1:
		for i in boxes:
			x1, y1, w1, h1 = i
			area.append(w1*h1)
	else:
		x1, y1, w1, h1 = boxes[0]
		area.append(w1*h1)
	return area

def refine_coords_as_frames_with_bbox(bbox, area, obj_bbox, area_yolo, coords_final, count, output_path, orig, save_maps):
	#count += 1
	coords_final = np.asarray(coords_final)
	
	if coords_final.size == 0:
		#coords_final_new = []
		if obj_bbox.size == 0:
			max_ind = np.argmax(area)
			x2,y2,w2,h2 = bbox[max_ind]
			if save_maps:
				cv2.rectangle(orig, (x2,y2), (x2+w2,y2+h2), (36,255,12), 2)
				cv2.putText(orig, str(1), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
				write_frame_with_bbox(output_path, count, orig)
			return bbox[max_ind]
		else:
			max_ind = np.argmax(area_yolo)
			x2,y2,w2,h2 = obj_bbox[max_ind]
			if save_maps:
				cv2.rectangle(orig, (x2,y2), (x2+w2,y2+h2), (36,255,12), 2)
				cv2.putText(orig, str(1), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
				write_frame_with_bbox(output_path, count, orig)
			return obj_bbox[max_ind]
	else:		
		cc = 1
		cord_area = get_area_of_boxes(coords_final)
		max_ind = np.argmax(cord_area)
		x2,y2,w2,h2 = coords_final[max_ind]
		if save_maps:
			cv2.rectangle(orig, (x2,y2), (x2+w2,y2+h2), (36,255,12), 2)
			cv2.putText(orig, str(1), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
			write_frame_with_bbox(output_path, count, orig)
		return coords_final[max_ind]
	#plt.imshow(orig)
	

def post_processing(predictions, my_net, orig_frame, save_maps, path_to_yolo360, path_to_output_maps, output_filename_npz):
	print("Generating output from hybrid model...")
	frame_number = []
	frame_coords = []
	for i in range(len(predictions))[:]:
		image_real = orig_frame[i]	
		image = image_real.copy()
		orig = image_real.copy()
		#plt.imshow(image_real)
		#plt.imshow(orig)
		#plt.imshow(image)
		#plt.axis('off')
		
		thresh = np.squeeze(predictions[i], axis=-1)
		thresh = (255 * thresh).astype(np.uint8)
		thresh = cv2.resize(thresh, (orig_width, orig_height))
		#plt.imshow(thresh, cmap='gray')
		
		bbox, area = get_bbox_from_prediction(thresh)
		bbox = np.asarray(bbox)
		
		#yolov3 for 360-degree images
		output_frame, obj_bbox, area_yolo = get_detected_objects_yolo360(image, my_net, path_to_yolo360)
		obj_bbox = np.asarray(obj_bbox)
		
		coords_final = compute_distance_and_merge_groups(bbox, obj_bbox)
		
		coords_final = refine_coords_as_frames_with_bbox(bbox, area, obj_bbox, area_yolo, coords_final, i, path_to_output_maps, orig, save_maps)
		
		#preparing to write output
		frame_number.append(str(i))
		frame_coords.append(coords_final)
		
	write_npz_variables(path_to_output_maps, output_filename_npz, frame_number, frame_coords)
		
def write_npz_variables(path_to_output_maps, output_filename_npz, frame_number, frame_coords):
	print("Writing variable...{} ".format(output_filename_npz))
	BASE_PATH = path_to_output_maps
	np.savez(os.path.join(BASE_PATH, output_filename_npz), frame_number=frame_number, frame_coords=frame_coords)
	print("Saved with filename {} Done!".format(output_filename_npz))