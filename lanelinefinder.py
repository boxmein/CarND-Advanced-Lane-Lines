#!/usr/bin/env python3
import numpy as np
import cv2
import pickle
import math
import os

DEBUG = True

SAT_THRESH = (120, 255)
LIG_THRESH = (40, 255) 
SOBEL_ORIENT = 'x'
SOBEL_THRESH = (12, 255)

TRANSFORM_SRC = np.array([
    [577, 464],
    [707, 464],
    [289, 663],
    [1019, 663]
], dtype=np.float32)

TRANSFORM_DST = np.array([
    [361, 0],
    [963, 0],
    [361, 720],
    [963, 720]
], dtype=np.float32)

TRANSFORM = cv2.getPerspectiveTransform(TRANSFORM_SRC, TRANSFORM_DST)
INV_TRANSFORM = cv2.getPerspectiveTransform(TRANSFORM_DST, TRANSFORM_SRC)

def abs_sobel_thresh(img, orient=SOBEL_ORIENT, thresh=SOBEL_THRESH):
    # Apply the following steps to img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x': 
        sobeled = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    else:
        sobeled = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3) Take the absolute value of the derivative or gradient
    sobeled = np.absolute(sobeled)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobeled = np.uint8(sobeled / np.max(sobeled) * 255.0)
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    mask = np.zeros_like(sobeled)
    mask[(sobeled > thresh[0]) & (sobeled < thresh[1])] = 1
    
    return mask

def thresh(img, thresh=SAT_THRESH):
    mask = np.zeros_like(img)
    mask[(img >= thresh[0]) & (img < thresh[1])] = 1
    return mask

def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)

def thresh_pipeline(img):
    hls   = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    b_sblx = abs_sobel_thresh(s_channel)
    
    s_thr = thresh(s_channel, SAT_THRESH)
    l_thr = thresh(l_channel, LIG_THRESH)
    
    mask = np.zeros_like(s_channel)
    mask[(b_sblx == 1) | (s_thr == 1) & (l_thr == 1)] = 1
    print("Threshold result", mask.shape)
    return mask

def transform_pipeline(img):
    img = cv2.warpPerspective(img, TRANSFORM, (img.shape[1], img.shape[0]))
    return img




def process(image):
    print("Undistorting", image.shape)
    image = undistort(image)

    cv2.imshow("Image", image)

    print("Thresholding", image.shape)
    img = thresh_pipeline(image)

    cv2.imshow("Thresh", img * 255)

    print("Transforming", img.shape)
    img = transform_pipeline(img) 

    cv2.imshow("Transform", img * 255)
    return img

class LaneFinder:
    def __init__(self):
        pass

    def sliding_fit_find_lanes(self, pipeline_image, nwindows=14):
        # 
        # Find approximate lane positions considering entire bottom half of image
        # 
        
        # 1. Find lane pixel count per column over entire bottom half 
        #    - This gives us two peaks where lane lines were found
        histogram = np.sum(pipeline_image[int(pipeline_image.shape[0]/2):,:], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((pipeline_image, pipeline_image, pipeline_image))*255

        # 2. Split the histogram into a left and right half
        midpoint = np.int(histogram.shape[0]/2)

        # 3. Find the peak of the left and right halves of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Calculate window heights
        window_height = np.int(pipeline_image.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = pipeline_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Each window will shift the peaks left and right according to their own peaks
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # 
        # Start sliding-window search to approximate lane line shape
        # 

        # Step through the windows one by one
        for window in range(nwindows):
            
            # Calculate window boundaries in x and y (and right and left)
            win_y_low = pipeline_image.shape[0] - (window+1)*window_height
            win_y_high = pipeline_image.shape[0] - window*window_height
            
            # Calculate left & right margins of the window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        cv2.imshow("Mapped", out_img)

        cv2.waitKey(1000)
        return self.left_fit, self.right_fit

    def find_next_lane_on_window(self, pipeline_image):
        # Find lane lines in a 100px left-right margin around the last fit
        nonzero = pipeline_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        return self.left_fit, self.right_fit

    def draw_lane_regions(self, pipeline_image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds):
        # Generate x and y values for plotting
        ploty = np.linspace(0, pipeline_image.shape[0]-1, pipeline_image.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((pipeline_image, pipeline_image, pipeline_image))*255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        cv2.imshow("Regions", result)



#
# Entry point
# 

# Test on a few images


with open("camera.p", "rb") as cam_f:
    mtx, dist = pickle.load(cam_f)


def run_image(filename):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pipeline_image = process(img)

    lf = LaneFinder()
    
    fitx, fity = lf.sliding_fit_find_lanes(pipeline_image)
    print("Fit 1 X: ", fitx)
    print("Fit 1 Y: ", fity)

    fitx, fity = lf.find_next_lane_on_window(pipeline_image)
    print("Fit 2 X: ", fitx)
    print("Fit 2 Y: ", fity)

for f in os.listdir("./test_images"):
    run_image("test_images/" + f)
