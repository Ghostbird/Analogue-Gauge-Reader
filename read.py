#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:55:01 2018

@author: Gijsbert ter Horst
"""

import argparse
import cv2
import numpy as np
from math import pi as pi
from matplotlib import pyplot as plt
from scipy.interpolate import spline

def read_video(filename, seconds_per_frame = 10, min_value = 0, max_value=1200):
    """ This is a highly specific function that needs tweaking before it will work for you use case.
    
    It reads a video file of an analogue gauge and calculates the time and needle angle combinations
    using openCV image recognition. It returns these as a list of tuples (time in seconds, angle in radians)
    
    In the current version many values are hard coded, such as:
        the frame crop
        the HSV filter to find the red needle
        filters out horizontal lines, so if you have a gauge whose needle passes the horizontal, you'll have to remove that.
        weights for the moving average of the angle, to reduce noise
        all the parameters for the Probabilistic Hough Lines transform
    """
    cap = cv2.VideoCapture(filename)
    print(cv2.__version__)
    # See: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
    fgbg=cv2.createBackgroundSubtractorMOG2(10,30)
    
    # Small kernel used to remove noise
    kernel = np.ones((2,2), np.uint8)
    
    # Running average theta, not used at the moment.
    avg_theta = None
    
    # Array to store the image recognition results
    angles = []
    
    # Video frame index
    index = 0
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Break if no more frames available
        if not ret:
            break
        
        # Crop the frame to the part that shows the gauge
        # Reduces false positives caused by noise in irrelevant parts of the image
        crop = frame[100:550,600:1500]
        
        # Calculate a foreground image mask
        # See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
        fgmask = fgbg.apply(crop)
        
        # Convert the frame to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Create a mask that only leaves the red values.
        # I don't understand why the hue component is 20-240 either.
        # That would normally exclude red AFAIK
        # However, this gave decent results in my case after some trial and error.
        lower_red = np.array([20,20,100])
        upper_red = np.array([240,250,230])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Apply the foreground mask to the colour mask
        mask = cv2.bitwise_and(mask,mask, mask=fgmask)
        
        # Use small kerneled morphological
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        lines = cv2.HoughLinesP(morph, 1, pi/360, 5, minLineLength=50, maxLineGap=20)
        if lines is not None and len(lines) > 0:
            # The HoughLinesP implementation wraps the return value in an extra redundant array.
            for l in lines[0]:
                x1, y1, x2, y2 = l
                # Ensure the line is drawn bottom to top
                # otherwise the theta can flip pi between two frames
                if y2 < y1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                theta = np.arctan2(y2-y1, x2 - x1)
                if avg_theta is None:
                    avg_theta = theta
                # Rudimentary filtering
                if abs(theta - pi) < 0.2 or abs(theta) < 0.2:
                    # Look for the next line found in this frame
                    continue
                avg_theta = avg_theta * 0.8 + theta * 0.2
                #print(x1, y1, x2, y2)
                #print(theta)
                angles.append((index,theta))
                cv2.line(crop, (x1, y1), (x2, y2), (0,0,255), 3)
                break
        
        # If you do not care about visual feeback at all and just want to run at max speed,
        # comment the next few lines until, and including, the break.
        
        # Show the image, otherwise you have no idea what's happening. This is great feedback.
        # During development, you can show intermediate results and masks instead of "crop"
        # That makes it easy to gauge (no pun intended) the effects of changes to processing parameters
        cv2.imshow('gauge', crop)
        
        # Show the frame for 1ms and record keypresses in practice the frame is probably shown longer due to processing time.
        # However, during development you might want to increase the wait time, to see the effects of process changes more clearly
        # 1ms is great when your algorithm is finished, and you just want to run it as fast as possible, but still see an indication of progress.
        
        k = cv2.waitKey(1) & 0xff
        # Stop image processing on ESC key press
        if k == 27:
            break
        
        #increment frame counter
        index += 1
    
    # Unload the video file from memory
    cap.release()
    # Close the window(s) that show the frame
    cv2.destroyAllWindows()
    
    # Calculate the maximimum angle seen in de video
    max_theta = np.max([theta for _, theta in angles])
    # And the minimum angle seen in the video
    min_theta = np.min([theta for _, theta in angles])
    
    # Calculate the mapping factor to translate the needle angle to the gauge units
    map_factor = (max_value - min_value) / (max_theta - min_theta)
    
    # Return a list of tuples (seconds, gauge value)
    return [(ix * seconds_per_frame, (theta - min_theta) * map_factor) for ix, theta in angles]
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read value from analogue meter.')
    parser.add_argument('filename', type=str, help='The file to read the meter from')
    args = parser.parse_args()
    data = read_video(args.filename)
    t = np.array([s for s, _ in data])
    T = np.array([theta for _, theta in data])
    plt.plot(t, T)