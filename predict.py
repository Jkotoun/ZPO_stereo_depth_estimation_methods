#author: Jiri Vlasak, Vit Tlustos
# EXAMPLE USAGE:
# SMGB 
# python predict.py --data_pattern="data/ --iml_pattern="**\im0.png" --imr_pattern="**\im1.png" --gtl_pattern="**/disp0GT.pfm" --png_disparity_factor=256 --evaluate --predictor="sgbm" --model_path="models/flyingthings_finalpass_xl.pb" --blockSize=7 --maxDisparity=128
# BM
# python predict.py --data_pattern="data/" --iml_pattern="**\im0.png" --imr_pattern="**\im1.png" --gtl_pattern="**/disp0GT.pfm" --png_disparity_factor=256 --evaluate --predictor="bm" --model_path="models/flyingthings_finalpass_xl.pb" --blockSize=7 --maxDisparity=128
# HITNET
# python predict.py --data_pattern="data/" --iml_pattern="**\im0.png" --imr_pattern="**\im1.png" --gtl_pattern="**/disp0GT.pfm" --png_disparity_factor=256 --evaluate --predictor="hitnet" --model_path="models/flyingthings_finalpass_xl.pb" --blockSize=7 --maxDisparity=128

"""HITNet prediction main file.

This script processes pairs of images with a frozen HITNet model and saves the
predictions as 16bit PNG or fp32 PFM files.
"""
import glob
import io

from absl import app
from absl import flags
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import cv2
from tqdm import tqdm

from src.methods.hitnet import HitNet
from src.methods.semiglobal_block_matching import SemiglobalBlockMatching 
from src.methods.block_matching import BlockMatching, BlockMatchingOpenCV

from src.augmentations.lighting import adjust_brightnes, adjust_contrast
from src.augmentations.noise import add_gaussian_noise, add_salt_and_pepper_noise

AUGMENTIONS = {
    'none' : lambda x: x,
    'brightness_+50' : lambda x: adjust_brightnes(x, 50),
    'brightness_-50' : lambda x: adjust_brightnes(x, -50),
    'contrast_1.5' : lambda x: adjust_contrast(x, 1.5),
    'contrast_2.' : lambda x: adjust_contrast(x, 2.),
    'gaussian_0.1' : lambda x: add_gaussian_noise(x, 0, 0.1),
    'gaussian_0.5' : lambda x: add_gaussian_noise(x, 0, 0.5),
    'salt_and_pepper_0.5_0.05' : lambda x: add_salt_and_pepper_noise(x, 0.5, 0.05),
    'salt_and_pepper_0.5_0.1' : lambda x: add_salt_and_pepper_noise(x, 0.5, 0.1),
    'salt_and_pepper_0_0.05' : lambda x: add_salt_and_pepper_noise(x, 0, 0.05),
    'salt_and_pepper_1_0.05' : lambda x: add_salt_and_pepper_noise(x, 1, 0.05)
}

flags.DEFINE_string('data_pattern', 'data', 'Input sstable file pattern')
flags.DEFINE_string('predictor', 'hitnet', 'Type of predictor to use (hitnet, sgbm, bm)')
flags.DEFINE_string('model_path', 'model_path', 'Path to frozen model file')
flags.DEFINE_float('png_disparity_factor', 256, 'Disparity multiplcation factor'
                   ' for output.')
flags.DEFINE_string('iml_pattern', 'left*.png', 'Input left image pattern')
flags.DEFINE_string('imr_pattern', 'right*.png', 'Input right image pattern')
flags.DEFINE_string('gtl_pattern', 'gtl*.png', 'Input left gt pattern')
flags.DEFINE_integer('input_channels', 3,
                     'Number of input channels for the model.')
flags.DEFINE_boolean('evaluate', False, 'Compute metrics.')
flags.DEFINE_boolean('save_source', False, 'save source images with augmentations')

flags.DEFINE_boolean(
    'predict_right', False,
    'Whether to query and save disparity for secondary image.')
flags.DEFINE_boolean('save_png', True, 'Whether to save output as png.')
flags.DEFINE_boolean('save_palette', True, 'Whether to save output as png with color palette.')
flags.DEFINE_boolean('save_pfm', False, 'Whether to save output as pfm.')
flags.DEFINE_integer('blockSize', 5, 'Block size for block matching methods')
flags.DEFINE_integer('maxDisparity', 64, 'Maximum disparity for block matching methods')
FLAGS = flags.FLAGS

def pfm_as_bytes(filename):
    """Reads a disparity map groundtruth file (.pfm)."""
    with open(filename, 'rb') as f:
        header = f.readline().strip()
        width, height = [int(x) for x in f.readline().split()]
        scale = float(f.readline().strip())
        endian = '<' if scale < 0 else '>'
        shape = (height, width, 3) if header == 'PF' else (height, width, 1)
        data = np.frombuffer(f.read(), endian + 'f')
        data = data.reshape(shape)[::-1]  # PFM stores data upside down
    return data


def load_images(file_names,
                input_channels=3):
    """Load an image pair and optionally a GT pair from files.

    Optionally crops the inputs before they are seen by the model and model
    preprocessor.

    Args:
      file_names: Tuple with input and GT filenames.
      input_channels: number of input channels required by the frozen model.
      crop_left: Left crop amount.
      crop_right: Right crop amount.
      crop_top: Top crop amount.
      crop_bottom: Bottom crop amount.

    Returns:
      An np array with left and right images and optionall left and right GT.
    """
    left = cv2.imread(file_names[0])
    right = cv2.imread(file_names[1])
    gt = None
    if len(file_names) > 2:
        gt = pfm_as_bytes(file_names[2])
    if len(file_names) > 3:
        gtr = pfm_as_bytes(file_names[3])
        gt = np.concatenate((gt, gtr), axis=-1)
    num_dims = len(left.shape)
    # Make sure input images have 3-dim shape and 3 channels.
    if num_dims < 3:
        left = np.expand_dims(left, axis=-1)
        right = np.expand_dims(right, axis=-1)
        left = np.tile(left, (1, 1, input_channels))
        right = np.tile(right, (1, 1, input_channels))
    else:
        _, _, channels = left.shape
        if channels > input_channels:
            left = left[:, :, :input_channels]
            right = right[:, :, :input_channels]

    return left, right, gt


def encode_image_as_16bit_png(data, filename):
    with io.BytesIO() as im_bytesio:
        height, width = data.shape
        array_bytes = data.astype(np.uint16).tobytes()
        array_img = Image.new('I', (width, height))
        array_img.frombytes(array_bytes, 'raw', 'I;16')
        array_img.save(im_bytesio, format='png')
        with open(filename, 'wb') as f:
            f.write(im_bytesio.getvalue())


def encode_image_as_pfm(data, filename):
    with open(filename, 'wb') as f:
        f.write(bytes('Pf\n', 'ascii'))
        f.write(bytes('%d %d\n' % (data.shape[1], data.shape[0]), 'ascii'))
        f.write(bytes('-1.0\n', 'ascii'))
        f.write(data[::-1].tobytes())  # PFM stores data upside down


def evaluate(disparity, gt, psm_threshold=192, max_disparity=1e6):
    """Computes metrics for predicted disparity against GT.

    Computes:
      PSM EPE: average disparity error for pixels with less than psm_threshold GT
      disparity value.
      bad_X: percent of pixels with disparity error larger than X. The divisor is
      the number of pixels with valid GT in the image.

    Args:
      disparity: Predicted disparity.
      gt: GT disparity.
      psm_threshold: Disparity threshold to compute PSM EPE.
      max_disparity: Maximum valid GT disparity.

    Returns:
      An np array with example metrics.
      [psm_epe, bad_0.1, bad_0.5, bad_1.0, bad_2.0, bad_3.0].
    """
    # gt_uint = gt.astype(np.uint8)
    # disparity_uint = disparity.astype(np.uint8).squeeze(0)
    gt_mask = np.where((gt > 0) & (gt < max_disparity), np.ones_like(gt),
                       np.zeros_like(gt))
    gt_diff = np.where(gt_mask > 0, np.abs(gt - disparity), np.zeros_like(gt))
    psm_mask = np.where(gt < psm_threshold, gt_mask, np.zeros_like(gt))
    gt_mask_count = np.sum(gt_mask) + 1e-5
    psm_mask_count = np.sum(psm_mask) + 1e-5
    bad01 = np.where(gt_diff > 0.1, np.ones_like(
        gt_diff), np.zeros_like(gt_diff))
    bad05 = np.where(gt_diff > 0.5, np.ones_like(
        gt_diff), np.zeros_like(gt_diff))
    bad1 = np.where(gt_diff > 1.0, np.ones_like(
        gt_diff), np.zeros_like(gt_diff))
    bad2 = np.where(gt_diff > 2.0, np.ones_like(
        gt_diff), np.zeros_like(gt_diff))
    bad3 = np.where(gt_diff > 3.0, np.ones_like(
        gt_diff), np.zeros_like(gt_diff))

    bad01 = 100.0 * np.sum(bad01 * gt_mask) / gt_mask_count
    bad05 = 100.0 * np.sum(bad05 * gt_mask) / gt_mask_count
    bad1 = 100.0 * np.sum(bad1 * gt_mask) / gt_mask_count
    bad2 = 100.0 * np.sum(bad2 * gt_mask) / gt_mask_count
    bad3 = 100.0 * np.sum(bad3 * gt_mask) / gt_mask_count
    psm_epe = np.sum(gt_diff * psm_mask) / psm_mask_count
    return np.array([psm_epe, bad01, bad05, bad1, bad2, bad3])


def main(argv):
    del argv  # Unused

    # will hold results for each augmentation pipeline
    results = {
        aug: [] 
        for aug in AUGMENTIONS.keys()
    }

    # 1. generate lists of images to process
    iml_files = sorted(glob.glob(FLAGS.data_pattern + FLAGS.iml_pattern))
    imr_files = sorted(glob.glob(FLAGS.data_pattern + FLAGS.imr_pattern))
    if len(iml_files) ==0 or len(imr_files) == 0:
        print("No files found for pattern: ", FLAGS.data_pattern + FLAGS.iml_pattern)
        return

    if FLAGS.evaluate:
        gtl_files = sorted(
            glob.glob(FLAGS.data_pattern + FLAGS.gtl_pattern))
        all_files = zip(iml_files, imr_files, gtl_files)
    else:
        all_files = zip(iml_files, imr_files)

    if FLAGS.predictor == 'hitnet':
        predictor = HitNet(FLAGS.model_path)
    elif FLAGS.predictor == 'sgbm':
        predictor = SemiglobalBlockMatching(blockSize=FLAGS.blockSize, maxDisparity=FLAGS.maxDisparity)
    elif FLAGS.predictor == 'bm':
        predictor = BlockMatchingOpenCV(blockSize=FLAGS.blockSize, maxDisparity=FLAGS.maxDisparity)
    else:
        raise ValueError(f"Invalid predictor_type: {FLAGS.predictor}")

    print("Using predictor: ", FLAGS.predictor)


    # 1. process all files
    for file_names in tqdm(all_files, total=len(iml_files)):
        print(file_names)
        
        # 1.1 load new pair of images to process.
        left, right, np_gt = load_images(file_names, FLAGS.input_channels)

        filename = file_names[0].replace('.png', '')
        
        # 1.2 process for all augmentations
        for aug_key, aug_fce in AUGMENTIONS.items():
            # 1.2 augment images
            left_aug = aug_fce(left)
            right_aug = aug_fce(right)

            # 1.3 process images by the predictor
            reference_disparity = predictor.process(left_aug, right_aug)
            #prev = reference_disparity.astype('uint8')

            # 1.4 run evaluation
            if FLAGS.evaluate:
                results[aug_key].append(
                    evaluate(reference_disparity, np_gt)
                )

            if FLAGS.save_source:
                cv2.imwrite(f'{filename}_{aug_key}_left.png', left_aug)
            
            # 1.5 save the results
            if FLAGS.save_png:
                encode_image_as_16bit_png(
                    reference_disparity[0, :, :, 0] *
                    FLAGS.png_disparity_factor,
                    f'{filename}_{FLAGS.predictor}_{aug_key}.png'
                )
            if FLAGS.save_pfm:
                encode_image_as_pfm(
                    reference_disparity[0, :, :, 0] *
                    FLAGS.png_disparity_factor,
                    f'{filename}_{FLAGS.predictor}_{aug_key}.pfm')
                
            if FLAGS.save_palette:
                #normalize the disparity map
                disparity = reference_disparity[0, :, :, 0]
                # disparity = (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity))
                # disparity = (disparity * 255).astype(np.uint8)

                # Save the disparity map with matplotlib color palette
                plt.imsave(f'{filename}_{aug_key}_palette.png', disparity, cmap='jet', vmin=0, vmax=255)
            
        if FLAGS.save_pfm:
            encode_image_as_pfm(
                reference_disparity[0, :, :, 0] *
                FLAGS.png_disparity_factor,
                filename + '_reference.pfm')

    if FLAGS.evaluate:
        print(f'Images processed:')
        print('ID psm_epe bad_0.1 bad_0.5 bad_1.0 bad_2.0 bad_3.0')
        for ix, (aug_key, res) in enumerate(results.items()):
            np_res = np.array(res)
            if res is not None:
                # compute mean and std
                mean_results = np.mean(np_res, axis=0)
                std_results = np.std(np_res, axis=0)
  
                # print results in latex format
                output = f'{str(ix)} & '
                for i in range(6):
                    output += f' ${mean_results[i]:.2f} \pm {std_results[i]:.2f}$'
                    if i < 5:
                        output += ' & '
                print(output + ' \\\\ \hline')


if __name__ == '__main__':
    app.run(main)
