import glob
import sys
import timeit
import argparse
import os
import json

import numpy as np
import scipy
import cv2

from skimage.color import rgb2gray
# from skimage import io
from skimage.morphology import binary_opening, binary_closing

import get_maps
import preprocessing
import descriptor
import template
import minutiae_AEC_modified as minutiae_AEC
import enhancement_AEC
import show
import descriptor_PQ
import descriptor_DR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FeatureExtractionRolled:
    """Rolled impressions feature extractor"""

    def __init__(self, patch_types=None, des_model_dirs=None,
                 minu_model_dir=None, enhancement_model_dir=None):

        # Setting instance params
        self.des_models = None
        self.patch_types = patch_types
        self.minu_model = None
        self.minu_model_dir = minu_model_dir
        self.des_model_dirs = des_model_dirs
        self.enhancement_model_dir = enhancement_model_dir

        print("Loading models, this may take some time...")
        if self.minu_model_dir is not None:
            print("Loading minutiae model: " + minu_model_dir)
            self.minu_model = minutiae_AEC.ImportGraph(minu_model_dir)

        # Constructing orientation dictionaries
        dicts = get_maps.construct_dictionary(ori_num=24)
        self.dict, self.spacing, self.dict_all = dicts[:3]
        self.dict_ori, self.dict_spacing = dicts[3:]

        # Setting patch index table
        patch_size = 160
        ori_num = 64
        if des_model_dirs is not None and len(des_model_dirs) > 0:
            self.patch_index_v = descriptor.get_patch_index(
                patch_size, patch_size, ori_num, isMinu=1
            )

        # Loading descriptor models
        if self.des_model_dirs is not None:
            self.des_models = []
            for i, model_dir in enumerate(des_model_dirs):
                echo_info = (i + 1, len(des_model_dirs), model_dir)
                print("Loading descriptor model (%d of %d ): %s" % echo_info)
                self.des_models.append(descriptor.ImportGraph(
                    model_dir, input_name="inputs:0",
                    output_name='embedding:0')
                )
            self.patch_size = 96

        if self.enhancement_model_dir is not None:
            print("Loading enhancement model: " + self.enhancement_model_dir)
            emodel = enhancement_AEC.ImportGraph(enhancement_model_dir)
            self.enhancement_model = emodel

    def remove_spurious_minutiae(self, mnt, mask, r=5):
        """Remove minutiae outside and in the borders of a mask"""
        minu_num = len(mnt)
        if minu_num <= 0:
            return mnt

        # Marking minutiae to be removed
        flag = np.ones((minu_num,), np.uint8)
        h, w = mask.shape[:2]
        for i in range(minu_num):
            x, y = np.int(mnt[i, 0]), np.int(mnt[i, 1])

            if x < r or y < r or x > w - r - 1 or y > h - r - 1:
                flag[i] = 0
            elif(mask[y - r, x - r] == 0 or mask[y - r, x + r] == 0 or
                 mask[y + r, x - r] == 0 or mask[y + r, x + r] == 0):
                flag[i] = 0

        # Filtering minutiae
        mnt = mnt[flag > 0, :]
        return mnt

    def feature_extraction_single(self, img_file, enhancement=False,
                                  output_dir=None, ppi=500):
        """Extracting features from a single image"""

        # Global param
        block_size = 16

        # Img name
        img_name = tuple(os.path.basename(img_file).split("."))

        # Checking if image exists
        if not os.path.exists(img_file):
            return None

        # Loading image
        # img = io.imread(img_file, as_gray=True)
        img = cv2.imread(img_file, 0)

        # Resizing to 500 ppi
        if ppi != 500:
            img = cv2.resize(img, (0, 0), fx=500.0 / ppi, fy=500.0 / ppi)

        # Adjusting image size to block_size
        img = preprocessing.adjust_image_size(img, block_size)

        # Converting to gray scale if needed (not-required I think)
        if len(img.shape) > 2:
            img = rgb2gray(img)

        # current image shape
        h, w = img.shape

        if not enhancement:
            # Intensity quality map
            start = timeit.default_timer()
            mask = get_maps.get_quality_map_intensity(img)
            stop = timeit.default_timer()
            print('time for cropping : %f' % (stop - start))

            # Saving quality map
            fname = os.path.join(output_dir, "%s_qmi.%s" % img_name)
            cv2.imwrite(fname, mask * 255)

        # Obtaining texture image
        start = timeit.default_timer()
        contrast_img = preprocessing.local_constrast_enhancement(img)
        texture_img = preprocessing.FastCartoonTexture(
            contrast_img, sigma=2.5, show=False
        )
        stop = timeit.default_timer()
        print('time for texture_img : %f' % (stop - start))

        # Saving texture image
        fname = os.path.join(output_dir, "%s_tex.%s" % img_name)
        cv2.imwrite(fname, texture_img)

        mnt_img = texture_img
        if enhancement:
            start = timeit.default_timer()
            print("Running enhancement autoencoder")
            stft_texture_img = preprocessing.STFT(texture_img)

            # Saving stft image
            fname = os.path.join(output_dir, "%s_stft.%s" % img_name)
            cv2.imwrite(fname, stft_texture_img)

            aec_img = self.enhancement_model.run_whole_image(stft_texture_img)
            stop = timeit.default_timer()
            print('time for enhancement img: %f' % (stop - start))

            # Saving enhanced image
            fname = os.path.join(output_dir, "%s_aec.%s" % img_name)
            cv2.imwrite(fname, aec_img.astype(np.uint8))

            # Quality maps
            maps = get_maps.get_quality_map_dict(
                aec_img, self.dict_all, self.dict_ori,
                self.dict_spacing, R=500.0
            )
            quality_map_aec, dir_map_aec, fre_map_aec = maps

            # Obtaining mask
            mask = quality_map_aec > 0.35  # 0.45 in latent images
            mask = binary_closing(mask, np.ones((3, 3))).astype(np.int)
            mask = binary_opening(mask, np.ones((3, 3))).astype(np.int)
            blkmask_ssim = get_maps.SSIM(stft_texture_img, aec_img, thr=0.2)
            blkmask = blkmask_ssim * mask
            blk_h, blk_w = blkmask.shape
            mask = cv2.resize(
                blkmask.astype(float),
                (block_size * blk_w, block_size * blk_h),
                interpolation=cv2.INTER_LINEAR
            )
            mask[mask > 0] = 1

            # Saving mask
            fname = os.path.join(output_dir, "%s_mask.%s" % img_name)
            cv2.imwrite(fname, mask * 255)

            mnt_img = aec_img

        # Extracting minutiae
        start = timeit.default_timer()
        mnt = self.minu_model.run_whole_image(mnt_img, minu_thr=0.15)
        stop = timeit.default_timer()
        print('time for minutiae : %f' % (stop - start))

        # Filtrating minutiae
        mnt = self.remove_spurious_minutiae(mnt, mask)

        # Saving minutiae
        fname = os.path.join(output_dir, "%s_mnt.txt" % img_name[0])
        with open(fname, "w") as mf:
            mf.write("%d\n" % len(mnt))
            for m in mnt:
                m = tuple(m)
                mf.write("%d %d %f %f\n" % m)

        # Plotting minutiae
        fname = os.path.join(output_dir, "%s_mntp.tif" % img_name[0])
        show.show_minutiae_sets(mnt_img, [[], mnt], mask=mask,
                                block=False, fname=fname)

        # Minutiae descriptor
        start = timeit.default_timer()
        des = descriptor.minutiae_descriptor_extraction(
            img, mnt, self.patch_types, self.des_models, self.patch_index_v,
            batch_size=256, patch_size=self.patch_size
        )
        stop = timeit.default_timer()
        print('time for descriptor : %f' % (stop - start))

        # Obtaining orientation map and frequency map with the STFT
        start = timeit.default_timer()
        dir_map, freq_map = get_maps.get_maps_STFT(
            img, patch_size=64, block_size=block_size, preprocess=True)
        stop = timeit.default_timer()
        print('time for STFT : %f' % (stop - start))

        # Saving orientation map
        new_dir_map = (np.degrees(dir_map) + 90).astype(np.uint8)
        new_dir_map[new_dir_map < 0] += 180
        new_dir_map[new_dir_map >= 180] -= 180
        new_dir_map = (new_dir_map / 1.5).astype(np.uint8)

        fname = os.path.join(output_dir, "%s_bori.%s" % img_name)
        cv2.imwrite(fname, new_dir_map)

        # Creating templates
        blkH = h // block_size
        blkW = w // block_size
        minu_template = template.MinuTemplate(
            h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt,
            des=des, oimg=dir_map, mask=mask
        )
        rolled_template = template.Template()
        rolled_template.add_minu_template(minu_template)

        # Texture template
        start = timeit.default_timer()
        stride = 16
        x = np.arange(24, w - 24, stride)
        y = np.arange(24, h - 24, stride)

        # Virtual minutiae
        virtual_minutiae = []
        dist_bg = scipy.ndimage.morphology.distance_transform_edt(mask)
        for y_i in y:
            for x_i in x:
                if (dist_bg[y_i][x_i] <= 24):
                    continue
                ofY = int(y_i / 16)
                ofX = int(x_i / 16)

                ori = -dir_map[ofY][ofX]
                virtual_minutiae.append([x_i, y_i, ori])
        virtual_minutiae = np.asarray(virtual_minutiae)

        if len(virtual_minutiae) > 1000:
            virtual_minutiae = virtual_minutiae[:1000]
        print("Virtual minutiae %d" % len(virtual_minutiae))

        # Saving virtual minutiae
        fname = os.path.join(output_dir, "%s_mntv.txt" % img_name[0])
        with open(fname, "w") as mf:
            mf.write("%d\n" % len(mnt))
            for m in virtual_minutiae:
                m = tuple(m)
                mf.write("%d %d %f\n" % m)

        # Obtaining texture template
        if len(virtual_minutiae) > 3:
            virtual_des = descriptor.minutiae_descriptor_extraction(
                contrast_img, virtual_minutiae, self.patch_types,
                self.des_models, self.patch_index_v, batch_size=128
            )
            texture_template = template.TextureTemplate(
                h=h, w=w, minutiae=virtual_minutiae,
                des=virtual_des, mask=mask)

            rolled_template.add_texture_template(texture_template)

        stop = timeit.default_timer()
        print('time for texture : %f' % (stop - start))

        return rolled_template

    def feature_extraction(self, image_dir, img_type='bmp', template_dir=None,
                           enhancement=False):
        """Feature extraction for a batch of images"""

        # Loading image names in input directory
        img_files = glob.glob(image_dir + '*.' + img_type)
        assert(len(img_files) > 0)
        img_files.sort()

        # Iterating over each image
        for i, img_file in enumerate(img_files):

            print(img_file)  # Printing image name
            start = timeit.default_timer()  # Start time

            # Checking if the feature template already exists and skipping
            img_name = os.path.basename(img_file)
            img_name = os.path.splitext(img_name)[0]
            fname = template_dir + img_name + '.dat'
            if os.path.exists(fname):
                continue

            # Extracting features without enhancement
            rolled_template = self.feature_extraction_single(
                img_file, output_dir=template_dir, enhancement=enhancement,
            )

            stop = timeit.default_timer()  # End time
            # Printing total execution time for one image
            print("Total time for extraction: %d" % (stop - start))

            # Writing binary template
            if template_dir is not None:

                # Template file name
                fname = template_dir + img_name + '.dat'
                print(fname)

                # Writing binary template
                template.Template2Bin_Byte_TF_C(
                    fname, rolled_template, isLatent=False
                )


def parse_arguments(argv):
    """Parse arguments"""

    # Parser
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        '--gpu', help='comma separated list of GPU(s) to use.', default='0'
    )

    parser.add_argument(
        '--N1', type=int, default=0,
        help='rolled index from which the enrollment starts'
    )

    parser.add_argument(
        '--N2', type=int, default=2000,
        help='rolled index from which the enrollment starts'
    )

    parser.add_argument(
        '--tdir', type=str,
        help='data path for minutiae descriptor and minutiae extraction'
    )

    parser.add_argument('--idir', type=str, help='data path for images')

    parser.add_argument('--itype', type=str, help='Image type', default="tif")

    parser.add_argument('--enhance', required=False, action='store_true',
                        help='Apply enhancement')

    # Parsing arguments
    return parser.parse_args(argv)


# Main execution
if __name__ == '__main__':

    # Parsing arguments
    args = parse_arguments(sys.argv[1:])

    # Configuring CUDA for GPUs
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Loading configuration file
    pwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(pwd + '/afis.config') as config_file:
        config = json.load(config_file)

    # Setting descriptor model names
    des_model_dirs = [
        config['DescriptorModelPatch2'],
        config['DescriptorModelPatch8'],
        config['DescriptorModelPatch11']
    ]

    # Setting patch types
    patch_types = [2, 8, 11]

    # Minutiae extraction model name
    minu_model_dir = config['MinutiaeExtractionModel']

    # Setting input and output directories
    img_dir = args.idir if args.idir else config['GalleryImageDirectory']
    temp_dir = args.tdir if args.tdir else config['GalleryTemplateDirectory']

    # enhancement model
    enhance_model_dir = config['EnhancementModel'] if args.enhance else None

    # Instantiating the feature extractor
    lf_rolled = FeatureExtractionRolled(
        patch_types=patch_types,
        des_model_dirs=des_model_dirs,
        minu_model_dir=minu_model_dir,
        enhancement_model_dir=enhance_model_dir
    )

    # Feature extraction
    print("Starting feature extraction (batch)...")
    lf_rolled.feature_extraction(
        image_dir=img_dir, template_dir=temp_dir, enhancement=args.enhance,
        img_type=args.itype
    )

    # Blocking this piece of code because it is buggy
    print("Finished feature extraction. Starting dimensionality reduction...")
    descriptor_DR.template_compression(
        input_dir=temp_dir, output_dir=temp_dir,
        model_path=config['DimensionalityReductionModel'],
        isLatent=False, config=None
    )

    print("Finished dimensionality reduction. Starting product quantization..")
    descriptor_PQ.encode_PQ(
        input_dir=temp_dir, output_dir=temp_dir, fprint_type='rolled')
    print("Finished product quantization. Exiting...")
