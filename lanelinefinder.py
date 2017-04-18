#!/usr/bin/env python3
# Advanced Lane Line Finding
# Johannes Kadak

# Imports
import numpy as np
import cv2
import pickle
import math
import os
from moviepy.editor import VideoFileClip
from collections import deque


# Distance average seems to be 600, so let's allow values from 600 - MAXDEV to 600 + MAXDEV
DISTANCE_AVG = 600
DISTANCE_MAXDEV = 90

# Saturation and lightness thresholds
SAT_THRESH = (120, 255)
LIG_THRESH = (40, 255) 

# Sobel filter thresholds
SOBEL_ORIENT = 'x'
SOBEL_THRESH = (12, 255)

# Image sizes passed into the finder, for reference
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Transformation coordinates - source and destination
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

# Calculate transform matrices before time
TRANSFORM = cv2.getPerspectiveTransform(TRANSFORM_SRC, TRANSFORM_DST)
INV_TRANSFORM = cv2.getPerspectiveTransform(TRANSFORM_DST, TRANSFORM_SRC)

# Use example code pixel/meter constants

# On the y-axis, meters/px
Y_M_PX = 30/720
# On the x-axis, meters/px
X_M_PX = 3.7/700


# 
# Calculate the absolute Sobel threshold 
# img : np.array(x, y) - the grayscaled image to threshold
# orient : char        - Sobel orientation, either 'x' or 'y'
# thresh : tuple(2)    - a tuple of (lower, upper) threshold from 0-255
# returns the new masked image
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

# 
# Take a threshold of the grayscaled image with lower and upper bounds
# img : np.array(x, y) - the grayscaled image to threshold
# thresh : tuple(2)    - a tuple of lower and upper thresholds from 0 to 255
# returns the new thresholded image
def thresh(img, thresh=SAT_THRESH):
    mask = np.zeros_like(img)
    mask[(img >= thresh[0]) & (img < thresh[1])] = 1
    return mask

# 
# Undistort an image using the camera matrices calculated/loaded ahead of time
# The camera matrix must be stored as mtx, and the distribution vectors as dist.
# returns the new undistorted image
def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)

#
# Implement the full image threshold pipeline for detecting lane lines
# Uses the Saturation and Lightness channels of the image to detect lane lines via thresholds. More details in the
# writeup.
# img : np.array(x, y, 3) - a RGB image
# returns the new thresholded image
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

# 
# Implements the entire transform pipeline to transform the perspective image into a birds-eye view image.
def transform_pipeline(img):
    img = cv2.warpPerspective(img, TRANSFORM, (img.shape[1], img.shape[0]))
    return img

# Process an image entirely by thresholding and transforming the image.
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

# The Lane Line Finding algorithm lives inside here.
# It's a separate class because it keeps internal state.
class LaneFinder:
    def __init__(self):

        # True if the last frame was found to have a good fit
        self.we_good = False
        
        # A deque for left-lane fitting data in the form of np.array([A, B, C])
        self.left_fit = deque(maxlen=10)
        # A deque for right-lane fitting data in the form of np.array([A, B, C])
        self.right_fit = deque(maxlen=10)

        # X-coordinates for the left-fit polynomial. Used for generating imagery and calculating curvatures.
        self.left_fitx = None
        # X-coordinates for the right-fit polynomial
        self.right_fitx = None
        
        # A simple linear space from 0 to 719 to match the X-coordinates in the left/right fits
        self.ploty = np.linspace(0, 719, 720 )
    
    # 
    # Find the lane line curvature for the left and right lanes, in meters.
    # Returns (left, right) the left and right curvatures in meters
    def calculate_curvature(self):
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*Y_M_PX, self.left_fitx*X_M_PX, 2)
        right_fit_cr = np.polyfit(self.ploty*Y_M_PX, self.right_fitx*X_M_PX, 2)
        # Calculate the new radii of curvature
        left_curverad = self.find_curve_radius(left_fit_cr)
        right_curverad = self.find_curve_radius(right_fit_cr)
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        return left_curverad, right_curverad
    
    #
    # Find the car's / camera's distance from the center of the lane in meters.
    # Assumes the camera center is dead center of the image (IMAGE_WIDTH / 2).
    def distance_from_center(self, left_fitx, right_fitx):
        xl = left_fitx[719]
        xr = right_fitx[719]

        lane_center = xr - xl / 2

        dist = (IMAGE_WIDTH/2.) - lane_center

        dist_m = dist * X_M_PX
        return dist_m


    # Find the radius for a curve with coefficients (A, B, C) = fit
    def find_curve_radius(self, fit, y=720):
        A, B, C = fit
        radius = (1 + (2 * A * y + B) ** 2) ** 1.5 / abs(2 * A)
        return radius

    # Simply the best sanity checker. It's true.
    # Detects if the lane polynomials make "sense" in the real world, as well.
    # If the lanes are positioned logically, have similar curvature and are around 600 px apart, then the frame 
    # is considered to be good.
    def check_good(self, left_fit, right_fit, left_fitx, right_fitx):
        if len(left_fit) == 0 or len(right_fit) == 0:
            self.we_good = False
            return

        dA = abs(left_fit[0] - right_fit[0])
        dB = abs(left_fit[1] - right_fit[1])
        
        l_lane_pos_ok = 150 < left_fit[2] < 600
        r_lane_pos_ok = 550 < right_fit[2] < 1350

        parallel_ok = dA < 0.01 and dB < 1
        print("dA: {}, dB: {}, okay = {}".format(dA, dB, parallel_ok))

        x1 = left_fitx[719]
        x2 = right_fitx[719]

        dist = abs(x1-x2)

        distance_ok = 600 - 90 < dist < 600 + 90

        print("x1: {}, x2: {}, dist: {}, okay = {}".format(x1, x2, dist, distance_ok))

        self.we_good = parallel_ok and distance_ok and l_lane_pos_ok and r_lane_pos_ok
    
    # Fit lane polynomials to the given image.
    # Automatically decides between using sliding window search or using a reduced search area.
    # If the reduced search area fails, falls back to sliding window search.
    # Returns the mean of the last 10 frame fittings, which is very smooth for our purposes.
    def fit_lane(self, image):
        if len(self.left_fit) > 0 and len(self.right_fit) > 0:
            print("Current fit data:")
            print(self.left_fit[0], self.right_fit[0])
        else:
            print("No fit data")
        
        # If the last frame detected well, we can reuse the window data.
        if self.we_good:
            print("Reusing windows")
            lf = self.left_fit[0]
            rf = self.right_fit[0]
            left_fit, right_fit, fit_img, left_fitx, right_fitx = self.find_next_lane_on_window(image, lf, rf)
        else:
            print("Searching new windows")
            left_fit, right_fit, fit_img, left_fitx, right_fitx = self.sliding_fit_find_lanes(image)

        # Check if this frame detected well
        self.check_good(left_fit, right_fit, left_fitx, right_fitx)
        
        # If it didn't give it a try with the sliding window method.
        if not self.we_good:
            print("Re-searching with window method")
            left_fit, right_fit, fit_img, left_fitx, right_fitx = self.sliding_fit_find_lanes(image)
            self.check_good(left_fit, right_fit, left_fitx, right_fitx)
        
        # If we're good now, then save the data.
        if self.we_good:
            print("Got good fit, adding")
            self.left_fit.appendleft(left_fit)
            self.right_fit.appendleft(right_fit)
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx
        
        else:
            print("Failed to get a good fit. Using earlier data.")

        # Otherwise, simply output the mean data without the last frame's points.
        lf_out = np.mean(self.left_fit, axis=0)
        rf_out = np.mean(self.right_fit, axis=0)

        return lf_out, rf_out, fit_img

    # Find lane lines using the Sliding Window method
    # Slight edits aside, this code is pretty much the Udacity source.
    def sliding_fit_find_lanes(self, pipeline_image, nwindows=14):
        
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
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return left_fit, right_fit, out_img, left_fitx, right_fitx

    # Find the next lane when previous left and right fits exist
    # Also pretty much the same as the Udacity source, adapted for my usage. 
    def find_next_lane_on_window(self, pipeline_image, left_fit, right_fit, margin=100):
        # Find lane lines in a 100px left-right margin around the last fit
        nonzero = pipeline_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        
        rendered = self.draw_lane_regions(pipeline_image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, margin)
        return left_fit, right_fit, rendered, left_fitx, right_fitx

    # Output the lane regions by drawing two polygons for the left and right lanes, used when we can use reduced
    # search area.
    # Only used for debug purposes, because the drawn frame doesn't end up on the fit image
    def draw_lane_regions(self, pipeline_image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, margin):
        # Generate x and y values for plotting

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((pipeline_image, pipeline_image, pipeline_image))*255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        return result
    
    
    # Draw the valid driving region onto the driving image.
    # Used in the end result for the green polygon covering the image.
    def draw_lane(self, image, left_fit, right_fit):
        # without calculating the X fits, can't draw the frame
        if self.left_fitx is None or self.right_fitx is None:
            print("SANITY: can't draw frame without left/right fit X data")
            return image
        zeros_image = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
        color_image = np.dstack((zeros_image, zeros_image, zeros_image))

        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_image, np.int_([pts]), (0,255, 0))

        # Draw text onto the screenie
        cv2.putText(image,"L: {:.4f} {:.4f} {:.4f}".format(*left_fit), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(image,"R: {:.4f} {:.4f} {:.4f}".format(*right_fit), (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(image,"rX: {:.4f} rY: {:.4f}".format(*self.calculate_curvature()), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(image,"dist from center: {:.4f}m".format(self.distance_from_center(left_fitx, right_fitx)), (0, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_image, INV_TRANSFORM, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

#
# Entry point
# 

# # Test on a few images


# Load the pre-trained camera matrix. The code for it exists in the iPython notebook.
with open("camera.p", "rb") as cam_f:
    mtx, dist = pickle.load(cam_f)

lf = LaneFinder()

# Find and draw lanes on a given image.
def run_image(img):
    print("---- FIT LANE ----")
    pipeline_image = process(img)

    fitl, fitr, fit_img = lf.fit_lane(pipeline_image)
    
    cv2.imshow("Image", fit_img)
    cv2.waitKey(3)

    print("Fit: ", fitl, fitr)

    lane_img = lf.draw_lane(img, fitl, fitr)
    cv2.imshow("Lane", lane_img)

    return lane_img

if __name__ == '__main__':
    # Run the code on all images in this video.
    clip = VideoFileClip("project_video.mp4")
    output = clip.fl_image(run_image)
    output.write_videofile("output_images/test.mp4", audio=False)
