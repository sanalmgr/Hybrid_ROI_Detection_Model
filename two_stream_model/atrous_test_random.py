# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:07:59 2022

@author: sanaalamgeer
"""
#import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #gpus[0], gpus[1], gpus[2]
######################################################################################
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#########################################
import keras
from keras.layers import Input, Conv2D, Conv1D, ELU, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, concatenate, add, AdditiveAttention, MaxPooling1D, LSTM, Reshape, UpSampling2D, Lambda, BatchNormalization
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from scipy.stats import spearmanr, kendalltau, pearsonr
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import h5py
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from skimage import color
import cv2
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
#########################################
#%%
def reload_model():
	input_layer1 = Input(shape=(200, 200, 1)) 
	feature_block =Conv2D(32, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv1')(input_layer1)
	feature_block=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(feature_block)
	feature_block=BatchNormalization()(feature_block)
	#conv2
	feature_block=Conv2D(32, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv2')(feature_block)
	feature_block=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(feature_block)
	feature_block=BatchNormalization()(feature_block)
	#conv3
	feature_block=Conv2D(64, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv3')(feature_block)
	feature_block=BatchNormalization()(feature_block)
	#conv4
	feature_block=Conv2D(64, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv4')(feature_block)
	feature_block = BatchNormalization()(feature_block)
	#conv5
	feature_block=Conv2D(128, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv5')(feature_block)
	feature_block = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(feature_block)
	feature_block = BatchNormalization()(feature_block)
	#############################################

	left_atr_conv1 = Conv2D(64, (3, 3), dilation_rate=6, padding='same', activation='relu')(feature_block)
	left_atr_conv1 = Conv2D(2, (1, 1), activation='relu')(left_atr_conv1)
	left_atr_conv1 = BatchNormalization()(left_atr_conv1)

	left_atr_conv2 = Conv2D(32, (3, 3), dilation_rate=12, padding='same', activation='relu')(feature_block)
	left_atr_conv2 = Conv2D(2, (1, 1), activation='relu')(left_atr_conv2)
	left_atr_conv2 = BatchNormalization()(left_atr_conv2)

	###############################################
	input_layer2 = Input(shape=(200, 200, 3)) 

	feature_block2 = Conv2D(32, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv11')(input_layer2)
	feature_block2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool11')(feature_block2)
	feature_block2 = BatchNormalization()(feature_block2)
	#conv2
	feature_block2=Conv2D(32, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv22')(feature_block2)
	feature_block2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool22')(feature_block2)
	feature_block2 = BatchNormalization()(feature_block2)
	#conv3
	feature_block2=Conv2D(64, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv33')(feature_block2)
	feature_block2 = BatchNormalization()(feature_block2)
	#conv4
	feature_block2=Conv2D(64, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv44')(feature_block2)
	feature_block2 = BatchNormalization()(feature_block2)
	#conv5
	feature_block2=Conv2D(128, (3, 3), kernel_initializer='normal', padding='same', activation='relu', name='conv55')(feature_block2)
	feature_block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool55')(feature_block2)
	feature_block2 = BatchNormalization()(feature_block2)
	##################################################
	right_atr_conv1 = Conv2D(64, (3, 3), dilation_rate=6, padding='same', activation='relu')(feature_block2)
	right_atr_conv1 = Conv2D(2, (1, 1), activation='relu')(right_atr_conv1)
	right_atr_conv1 = BatchNormalization()(right_atr_conv1)

	right_atr_conv2 = Conv2D(32, (3, 3), dilation_rate=12, padding='same', activation='relu')(feature_block2)
	right_atr_conv2 = Conv2D(2, (1, 1), activation='relu')(right_atr_conv2)
	right_atr_conv2 = BatchNormalization()(right_atr_conv2)
	#################################################
	left_atr_fusion1 = keras.layers.add([left_atr_conv1, left_atr_conv2])
	right_atr_fusion1 = keras.layers.add([right_atr_conv1, right_atr_conv2])

	main_fusion1 = keras.layers.concatenate([left_atr_fusion1, right_atr_fusion1])

	main_fusion1 = BatchNormalization()(main_fusion1)
	#################################################

	upsample = UpSampling2D(size=(8, 8), interpolation='bilinear')(main_fusion1)

	output_layer = Conv2D(1, (1, 1), activation='sigmoid')(upsample)

	model = Model([input_layer1, input_layer2], output_layer)

	#model.summary()
	
	return model

def evaluate_2stream_model(mask_rgb, opt_map, path_to_hdf5):
	print("Running two-stream model...")
	model = reload_model()
	model.load_weights(path_to_hdf5 + 'sal_model_t9.hdf5')	
	
	start_time = time.time()
	x_test = opt_map
	test_rgb = mask_rgb
	
	#print(f"x_test-{x_test.shape}")
	#print(f"test_rgb-{test_rgb.shape}")
	
	print("--- %s  Dataset loaded in seconds ---" % (time.time() - start_time))
	
	start_time = time.time()
	x_predict = model.predict([x_test, test_rgb], batch_size=4, verbose=1) 
	
	print("--- %s testing finsihed in seconds ---" % (time.time() - start_time))
	
	#print(x_predict[0])
	
	return x_predict
#######################END##################