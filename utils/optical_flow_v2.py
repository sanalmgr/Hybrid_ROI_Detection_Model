# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:30:37 2022

@author: sanaalamgeer
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calcOpticalFlow(hsv, prvsF, nextF):
	"""
        Extract optical flow from two consecutive frames
        Args:
            prev_frame: previous frame
            cur_frame: current frame
            res: resolution: (width, height)
        Returns:
            absflow: Flow intensity image
            flow: Optical flow
	"""
	#computing optical flow map
	prvsF = cv2.cvtColor(prvsF, cv2.COLOR_RGB2GRAY)
	nextF = cv2.cvtColor(nextF, cv2.COLOR_RGB2GRAY)
	
	dim = (prvsF.shape[1], prvsF.shape[0])
	nextF = cv2.resize(nextF, dim)
	flow = cv2.calcOpticalFlowFarneback(prvsF, nextF, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	
	bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
	
	return bgr