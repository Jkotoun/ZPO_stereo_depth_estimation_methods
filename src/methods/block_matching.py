#author: Josef Kotoun
import numpy as np
import cv2
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


class BlockMatching:

    def __init__(self, blockSize=21, maxDisparity=64):
        self.maxDisparity = maxDisparity
        self.blockSize = blockSize

  
    def process(self, left, right):
        # Convert the left and right images to grayscale
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        # Initialize the disparity map with zeros
        disparity = np.zeros(left.shape, np.uint8)
        
        # Get the width and height of the left image
        w = left.shape[1]
        h = left.shape[0]
        
        # Calculate half of the window size
        half_window = self.blockSize // 2
        
        # Calculate adjustment factor for the disparity values
        adjust_factor = 255 / self.maxDisparity
        
        # Loop through each pixel in the left image
        for i in tqdm(range(half_window, h - half_window)):
            for j in range(half_window, w - half_window):
                # Define the search range within the right image based on max_offset
                min_limit = max(half_window, j - self.maxDisparity)
                max_limit = min(w, j + self.maxDisparity - half_window)
                
                if self.blockSize == 1:
                    # If window size is 1, calculate pixel-wise absolute differences
                    values = np.abs(right[i][min_limit:max_limit] - left[i][j]) ** 2
                    # Find the index of the minimum difference
                    index = np.argmin(values) + min_limit
                    # Calculate disparity and assign it to the disparity map
                    disparity[i][j] = np.abs(index - j)
                else:
                    # If window size is greater than 1, use block matching
                    reference = left[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
                    minimum = math.inf
                    index = -1
                    
                    # Loop through the search range in the right image
                    for k in range(min_limit, max_limit - self.blockSize + 1):
                        # Calculate the sum of squared differences within the window
                        values = np.abs(right[i - half_window:i + half_window + 1, k - half_window:k + half_window + 1] - reference) ** 2
                        value = np.sum(values)
                        # Update minimum difference and corresponding index if found
                        if value < minimum:
                            minimum = value
                            index = k
                    # Calculate disparity and assign it to the disparity map
                    disparity[i][j] = np.abs(index - j)
        
        
        # Return the final disparity map
        return np.expand_dims(disparity, (0,-1))

class BlockMatchingOpenCV(BlockMatching):   
    def process(self, imgBGRLeft, imgBGRRight):
        imgGrayscaleLeft = cv2.cvtColor(imgBGRLeft, cv2.COLOR_BGR2GRAY)
        imgGrayscaleRight = cv2.cvtColor(imgBGRRight, cv2.COLOR_BGR2GRAY)

        #max disparity must be divisible by 16 in opencv functions for some reason
        disparityFactor = round(self.maxDisparity / 16) * 16
        stereo = cv2.StereoBM.create(
            numDisparities=disparityFactor, blockSize=self.blockSize)

        disparity = stereo.compute(imgGrayscaleLeft, imgGrayscaleRight)
        true_dmap = (disparity + abs(disparity.min())) * (1.0 / 16.0)
        # normalized_disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
        result = np.expand_dims(true_dmap, (0,-1))

        return result

