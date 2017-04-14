#!/usr/bin/env python3
import numpy as np
import cv2
import pickle
import math
import os
from moviepy.editor import VideoFileClip
from collections import deque


DEBUG = True

SAT_THRESH = (120, 255)
LIG_THRESH = (40, 255) 
SOBEL_ORIENT = 'x'
SOBEL_THRESH = (12, 255)

LANE_WIDTH = 3.7
LANE_HEIGHT = 30

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

Y_M_PX = 30/720 # meters per pixel in y dimension
X_M_PX = 3.7/700 # meters per pixel in x dimension


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
        self.we_good = False
        self.recent_xfitted = deque(maxlen=10)
        self.left_fit = None
        self.right_fit = None
        
        self.radius = None
        self.distance_from_line = None

        self.ploty = np.linspace(0, 719, 720 )
    
    def get_best_fit(self):
        
        avgl = np.zeros(3)
        avgr = np.zeros(3)

        for lfit, rfit in recent_xfitted:
            avgl += lfit
            avgr += rfit
        
        avgl /= len(recent_xfitted)
        avgr /= len(recent_xfitted)

        return avgl, avgr
    
    def metric_fit(self):
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

    def find_curve_radius(self, fit, y=720):
        A, B, C = fit
        radius = (1 + (2 * A * y + B) ** 2) ** 1.5 / abs(2 * A)
        return radius

    # The best sanity checker
    def check_good(self):
        self.we_good = True

    def fit_lane(self, image):
        if self.we_good:
            print("Reusing windows")
            return self.find_next_lane_on_window(image)
        else:
            print("Searching new windows")
            self.we_good = True
            return self.sliding_fit_find_lanes(image)

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

        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return self.left_fit, self.right_fit, out_img

    def find_next_lane_on_window(self, pipeline_image, margin=100):
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

        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        
        rendered = self.draw_lane_regions(pipeline_image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, margin)
        return self.left_fit, self.right_fit, rendered

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

    def draw_lane(self, image):

        zeros_image = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
        color_image = np.dstack((zeros_image, zeros_image, zeros_image))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_image, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_image, INV_TRANSFORM, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

#
# Entry point
# 

# Test on a few images


with open("camera.p", "rb") as cam_f:
    mtx, dist = pickle.load(cam_f)


lf = LaneFinder()

def run_image(img):
    pipeline_image = process(img)

    fitl, fitr, fit_img = lf.fit_lane(pipeline_image)

    lane_img = lf.draw_lane(img)
    lf.metric_fit()

    cv2.imshow("Image", fit_img)
    cv2.imshow("Lane", lane_img)
    cv2.waitKey(3)

    print("Fit L: ", fitl)
    print("Fit R: ", fitr)


    return lane_img

# for f in os.listdir("./test_images"):
#     img = cv2.imread("test_images/" + f)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     run_image(img)

clip = VideoFileClip("project_video.mp4")
output = clip.fl_image(run_image)
output.write_videofile("output_images/test.mp4", audio=False)