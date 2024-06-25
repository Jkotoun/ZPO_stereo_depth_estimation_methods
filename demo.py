# author: Josef Kotoun
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.methods.semiglobal_block_matching import SemiglobalBlockMatching
from src.methods.block_matching import BlockMatchingOpenCV, BlockMatching
from src.methods.hitnet import HitNet
from src.methods.depth_anything import DepthAnything


def plot_disparities(disparitySGBM, disparityBM_opencv):
    # plt.figure()
    # plt.title("BM")
    # # plt.imshow(disparityBM, cmap='jet')
    # plt.colorbar()
    plt.figure()
    plt.title("SGBM")
    plt.imshow(disparitySGBM, cmap='jet')
    plt.colorbar()
    plt.figure()
    plt.title("BM_opencv")
    plt.imshow(disparityBM_opencv, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    bm = BlockMatchingOpenCV(blockSize=5, maxDisparity=32)
    bm_ours = BlockMatching(blockSize=5, maxDisparity=32)
    sgbm = SemiglobalBlockMatching(blockSize=3, maxDisparity=32)
    hitnet = HitNet(r"models\flyingthings_finalpass_xl.pb")
    depth_anything = DepthAnything("small")
    # Path to the left input image
    left_image_path = './test_images/downsized/img2_left.png'
    # Path to the right input image
    right_image_path = './test_images/downsized/img2_right.png'
    imgLeft = cv2.imread(left_image_path)
    imgRight = cv2.imread(right_image_path)

    # disparity_SGBM = sgbm.process(imgLeft, imgRight)
    disparity_BM = bm.process(imgLeft, imgRight).squeeze(0)
    plt.figure()
    plt.title("BM_opencv")
    plt.imshow(disparity_BM, cmap='jet')
    plt.colorbar()

    disparity_SGBM = sgbm.process(imgLeft, imgRight).squeeze(0)
    plt.figure()
    plt.title("SGBM")
    plt.imshow(disparity_SGBM, cmap='jet')
    plt.colorbar()

    disparity_HITNET = hitnet.process(imgLeft, imgRight).squeeze(0)
    plt.figure()
    plt.title("HitNet")
    plt.imshow(disparity_HITNET, cmap='jet')
    plt.colorbar()

    disparity_DEPTH_ANYTHING = depth_anything.process(imgLeft)
    plt.figure()
    plt.title("DepthAnything")
    plt.imshow(disparity_DEPTH_ANYTHING)
    plt.colorbar()

    disparity_BM_ours = bm_ours.process(imgLeft, imgRight).squeeze(0)
    plt.figure()
    plt.title("BM_ours")
    plt.imshow(disparity_BM_ours, cmap='jet')
    plt.colorbar()

    plt.show()

    # plot_disparities(disparity_SGBM, disparity_BM)
