#author: Jiri Vlasak, Vit Tlustos
import argparse
import cv2

from src.methods.depth_anything import DepthAnything 
from src.augmentations.lighting import adjust_brightnes, adjust_contrast
from src.augmentations.noise import add_gaussian_noise, add_salt_and_pepper_noise

def main():
    parser = argparse.ArgumentParser(description='Comparison CLI')
    parser.add_argument('--img')

    args = parser.parse_args()

    # ---------------- INTIALIZE  ----------------
    depth_anything = DepthAnything("small")

    # ---------------- IMAGE PROCESSING ----------------

    # 1. load image
    img_orig = cv2.imread(args.img)

    # 2. augment the image
    #img_aug = adjust_brightnes(img_orig, 0)
    img_aug = adjust_contrast(img_orig, 2.)
    #img_aug = adjust_gamma(img_aug, 1)
    #img_aug = add_gaussian_noise(img_orig, 0, 0.1)
    #img_aug = add_salt_and_pepper_noise(img_orig, 0.5, 0.05)

    # ---------------- DEPTH ESTIMATION ----------------

    # 3. estimate depth
    depth_orig = depth_anything.process(img_orig)
    depth_aug = depth_anything.process(img_aug)
    diff = cv2.absdiff(depth_orig, depth_aug)

    # ---------------- VISUALIZATION ----------------
    cv2.imshow("Image [Original]", img_orig)
    cv2.imshow("Image [Augmented]", img_aug)
    cv2.imshow("Depth [Original]", depth_orig)
    cv2.imshow("Depth [Augmented]", depth_aug)
    cv2.imshow("Depth [Difference]", diff)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()