#author: Josef Kotoun
import numpy as np
import cv2

class SemiglobalBlockMatching:

    def __init__(self, blockSize=3, maxDisparity=14):
        self.blockSize = blockSize
        self.maxDisparity = maxDisparity

    def process(self, imgBGRLeft, imgBGRRight, applyColorMap=False):
        imgGrayscaleLeft = cv2.cvtColor(imgBGRLeft, cv2.COLOR_BGR2GRAY)
        imgGrayscaleRight = cv2.cvtColor(imgBGRRight, cv2.COLOR_BGR2GRAY)
        #max disparity must be divisible by 16 in opencv functions for some reason
        disparityFactor = round(self.maxDisparity / 16) * 16
        stereo  = cv2.StereoSGBM_create(minDisparity=-1,
                                    numDisparities=disparityFactor,
                                    blockSize=self.blockSize,
                                    P1=8*3*self.blockSize,
                                    P2=32*3*self.blockSize,
                                    disp12MaxDiff=12,
                                    uniquenessRatio=10,
                                    speckleWindowSize=50,
                                    speckleRange=32,
                                    preFilterCap=63,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    
        disparity = stereo.compute(imgGrayscaleLeft, imgGrayscaleRight)  # .astype(np.float32)/16
        # normalized_disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        # normalized_disparity = np.uint8(normalized_disparity)

        # if applyColorMap:
        #     normalized_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)

        # opencv outputs disparity multiplied by 16. also it can be negative so convert to positive
        true_dmap = (disparity + abs(disparity.min())) * (1.0 / 16.0)
        prev= true_dmap.astype('uint8')
        
        return np.expand_dims(true_dmap, (0,-1))
        
