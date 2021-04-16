#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:55:01 2018

@author: Gijsbert ter Horst
"""

import argparse
import cv2
import numpy as np
from math import pi
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def read_video(filename, seconds_per_frame, value_min, value_max, crop_x_min, crop_x_max, crop_y_min, crop_y_max,
               saturation_min, hough_threshold, hough_linelength_min, hough_linegap_max):
    """ This is a highly specific function that needs tweaking before it will work for you use case.
    
    It reads a video file of an analogue gauge and measures the and needle angle in each frame
    using openCV image recognition. It maps and returns these as a list of tuples (time in seconds, value)
    """
    cap = cv2.VideoCapture(filename)
    
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
        crop = frame[crop_y_min:crop_y_max,crop_x_min:crop_x_max]
        
        # Convert the frame to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Create a mask that only retains the coloured pixels
        lower_colour = np.array([0,saturation_min,0])
        upper_colour = np.array([255,255,255])
        cmask = cv2.inRange(hsv, lower_colour, upper_colour)
        
        # Use Probabilistic Hough Lines transformation to find lines in the image.
        # See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        lines = cv2.HoughLinesP(cmask, 1, pi/360, hough_threshold, minLineLength=hough_linelength_min, maxLineGap=hough_linegap_max)
        if lines is not None and len(lines) > 0:
            # The HoughLinesP implementation wraps the return value in an extra redundant array.
            for l in lines[0]:
                x1, y1, x2, y2 = l
                # Ensure the line is drawn bottom to top
                # otherwise the theta can flip π between two frames
                if y2 < y1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                # Calculate angle theta
                theta = np.arctan2(y2-y1, x2 - x1)
                
                # Add angle and frame index to results
                angles.append((index,theta))
                
                # Draw line on the cropped frame
                cv2.line(crop, (x1, y1), (x2, y2), (0,0,255), 3)
                
                # Only use the first line returned by the Hough Lines transform
                break
        
        # If you do not care about visual feeback at all and just want to run at max speed,
        # comment the next few lines until, and including, the break.
        
        # Show the image, otherwise you have no idea what's happening. This is great feedback.
        # During development, you can show intermediate results and masks instead of "crop"
        # That makes it easy to gauge (no pun intended) the effects of changes to processing parameters
        cv2.imshow('gauge', crop)
        
        # Show the frame for 1ms and record keypresses. In practice the frame is probably shown longer due to processing time.
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
    map_factor = (value_max - value_min) / (max_theta - min_theta)
    
    # Return a list of tuples (seconds, gauge value)
    return [(ix * seconds_per_frame, (theta - min_theta) * map_factor) for ix, theta in angles]
    

def main():
    parser = argparse.ArgumentParser(description='Read value from analogue meter.')
    parser.add_argument('filename', type=str, help='The file to read the meter from')
    args = parser.parse_args()
    data = read_video(
            args.filename,
            seconds_per_frame=10,
            value_min=0,
            value_max=1160,
            crop_x_min=250,
            crop_x_max=1820,
            crop_y_min=200,
            crop_y_max=750,
            saturation_min=70,
            hough_threshold=30,
            hough_linelength_min=250,
            hough_linegap_max=20)
    t = np.array([s for s, _ in data])
    T = np.array([theta for _, theta in data])
    plt.close('all')
    plt.ioff()
    fig = plt.figure('curves')
    axis = fig.add_subplot(1,1,1)
    axis.plot(t, T)
    
    # Find the index where cooling commences
    pivot_index = 0
    n = 30
    for i in range(n, len(data)):
        if len(data) <= i + n:
            # No pivot found at all
            break
        # Check whether a significant descent starts within the next n entries
        if np.mean(T[i+1:i+1+n]) < np.mean(T[i-n:i]):
            # Check the next n entries in reverse order, until we find the first local maximum
            for j in reversed(range(i,i+n)):
                if data[j] < data[j+1]:
                    pivot_index = j
                    break
            break
    
    curve = lambda x, a, b, c: a * np.log(b * x) + c
    
    params_heat = curve_fit(curve, t[0:pivot_index or -1], T[0:pivot_index or -1])
    print('Heating curve:\nT = {} log({}x) + {}'.format(*params_heat[0]))
    t_heat = range(0,data[pivot_index or -1][0])
    heating_curve = [curve(x, params_heat[0][0], params_heat[0][1], params_heat[0][2]) for x in t_heat]
    #print(heating_curve)
    axis.plot(t_heat, heating_curve)
    
    if pivot_index > 0:
        params_cool = curve_fit(curve, t[pivot_index:-1], T[pivot_index:-1])
        print('Cooling curve:\nT = {} log({}x) + {}'.format(*params_cool[0]))
        t_cool = range(data[pivot_index][0], data[-1][0])
        cooling_curve = [curve(x, params_cool[0][0], params_cool[0][1], params_cool[0][2]) for x in t_cool]
        #print(cooling_curve)
        axis.plot(t_cool, cooling_curve)
    
    plt.show()
    
if __name__ == '__main__':
    main()
