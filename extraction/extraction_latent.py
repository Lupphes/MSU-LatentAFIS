import glob
import math
import sys
import timeit
import json
import os
import argparse

from timeit import default_timer as timer

import matplotlib.pylab as plt
import numpy as np
import cv2
import scipy.spatial.distance

# from skimage import io

from skimage.morphology import binary_opening, binary_closing

import get_maps
import preprocessing
import filtering
import descriptor
import template
import minutiae_AEC
# import minutiae_AEC_modified as minutiae_AEC
import show
import enhancement_AEC
# import enhancement_AEC_npy as enhancement_AEC
import loggabor

import descriptor_PQ
import descriptor_DR

# Setting environment vars for tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Switching matplotlib backend
plt.switch_backend('agg')

# Global config
config = {}


def save_minutiae(mnt, fname):
    # Saving minutiae
    with open(fname, "w") as mf:
        mf.write("%d\n" % len(mnt))
        for m in mnt:
            m = tuple(m)
            mf.write("%d %d %f %f\n" % m)


class FeatureExtraction_Latent:
    def __init__(self, patch_types=None, des_model_dirs=None,
                 minu_model_dirs=None, enhancement_model_dir=None,
                 ROI_model_dir=None, coarsenet_dir=None, FineNet_dir=None):

        self.des_models = None
        self.patch_types = patch_types
        self.minu_model = None
        self.minu_model_dirs = minu_model_dirs
        self.des_model_dirs = des_model_dirs
        self.enhancement_model_dir = enhancement_model_dir
        self.ROI_model_dir = ROI_model_dir

        # Obtaining maps
        maps = get_maps.construct_dictionary(ori_num=60)
        self.dict, self.spacing, self.dict_all = maps[:3]
        self.dict_ori, self.dict_spacing = maps[3:]

        print("Loading models, this may take some time...")

        if self.minu_model_dirs is not None:
            self.minu_model = []
            for i, minu_model_dir in enumerate(minu_model_dirs):
                echo_info = (i + 1, len(minu_model_dirs), minu_model_dir)
                print("Loading minutiae model (%d of %d ): %s" % echo_info)
                model = minutiae_AEC.ImportGraph(minu_model_dir)
                self.minu_model.append(model)

        self.coarsenet_dir = coarsenet_dir

        patchSize = 160
        oriNum = 64
        self.patchIndexV = descriptor.get_patch_index(
            patchSize, patchSize, oriNum, isMinu=1
        )

        if self.des_model_dirs is not None:
            self.des_models = []
            for i, model_dir in enumerate(des_model_dirs):

                echo_info = (i + 1, len(des_model_dirs), model_dir)
                print("Loading descriptor model (%d of %d ): %s" % echo_info)

                dmodel = descriptor.ImportGraph(
                    model_dir, input_name="inputs:0",
                    output_name='embedding:0'
                )

                self.des_models.append(dmodel)

        if self.enhancement_model_dir is not None:
            print("Loading enhancement model: " + self.enhancement_model_dir)
            emodel = enhancement_AEC.ImportGraph(enhancement_model_dir)
            self.enhancement_model = emodel

        print("Finished loading models.")

    def feature_extraction_single_latent(self, img_file, output_dir=None,
                                         ppi=500, show_processes=False,
                                         show_minutiae=False, minu_file=None):
        # Params
        block = False
        block_size = 16

        show_minutiae = True

        # Loading images
        # img0 = io.imread(img_file, pilmode='L')  # / 255.0
        # img0 = io.imread(img_file, as_gray=True)
        img0 = cv2.imread(img_file, 0)
        img = img0.copy()

        # Img name
        img_name = tuple(os.path.basename(img_file).split("."))

        # REsizing to 500 ppi
        if ppi != 500:
            img = cv2.resize(img, (0, 0), fx=500.0 / ppi, fy=500.0 / ppi)

        # Adjusting image size to block size
        img = preprocessing.adjust_image_size(img, block_size)
        name = os.path.basename(img_file)
        root_name = output_dir + os.path.splitext(name)[0]

        # Starting timer for feature extraction
        start = timer()

        # input image shape after size adjustment
        h, w = img.shape

        # Do not process images larger than 1000x1000
        if h > 1000 and w > 1000:
            return None, None

        # cropping using two dictionary based approach
        if minu_file is not None:
            manu_minu = np.loadtxt(minu_file)
            # remove low quality minutiae points
            input_minu = np.array(manu_minu)
            input_minu[:, 2] = input_minu[:, 2] / 180.0 * np.pi
        else:
            input_minu = []

        # Preprocessing
        tex_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
        stft_texture_img = preprocessing.STFT(tex_img)

        ct_img_g = preprocessing.local_constrast_enhancement_gaussian(img)
        stft_img = preprocessing.STFT(img)
        ct_img_stft = preprocessing.STFT(ct_img_g)

        # Saving images
        fname = os.path.join(output_dir, "%s_tex.%s" % img_name)
        cv2.imwrite(fname, tex_img)
        fname = os.path.join(output_dir, "%s_texstft.%s" % img_name)
        cv2.imwrite(fname, stft_texture_img)
        fname = os.path.join(output_dir, "%s_lctg.%s" % img_name)
        cv2.imwrite(fname, ct_img_g)
        fname = os.path.join(output_dir, "%s_stft.%s" % img_name)
        cv2.imwrite(fname, stft_img)
        fname = os.path.join(output_dir, "%s_lctgstft.%s" % img_name)
        cv2.imwrite(fname, ct_img_stft)

        # step 1: enhance the latent based on our autoencoder
        aec_img = self.enhancement_model.run_whole_image(stft_texture_img)

        # saving enhanced image
        fname = os.path.join(output_dir, "%s_aec.%s" % img_name)
        cv2.imwrite(fname, aec_img)

        # Quality maps
        maps = get_maps.get_quality_map_dict(
            aec_img, self.dict_all, self.dict_ori, self.dict_spacing, R=500.0)
        quality_map_aec, dir_map_aec, fre_map_aec = maps

        # Obtaining mask
        bmask_aec = quality_map_aec > 0.45
        bmask_aec = binary_closing(bmask_aec, np.ones((3, 3))).astype(np.int)
        bmask_aec = binary_opening(bmask_aec, np.ones((3, 3))).astype(np.int)
        blkmask_ssim = get_maps.SSIM(stft_texture_img, aec_img, thr=0.2)
        blkmask = blkmask_ssim * bmask_aec
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

        # SAving AEC image with mask
        fname = os.path.join(output_dir, "%s_aec_mask.%s" % img_name)
        cv2.imwrite(fname, aec_img * mask)

        # Extracting minutiae
        minutiae_sets = []

        # Extracting minutiae from the STFT image
        mnt_stft = self.minu_model[0].run_whole_image(stft_img, minu_thr=0.05)
        minutiae_sets.append(mnt_stft)

        if show_minutiae:
            fname = root_name + '_stft_mnt.jpeg'
            show.show_minutiae_sets(stft_img, [input_minu, mnt_stft],
                                    mask=None, block=block, fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_stft_mnt.txt" % img_name[0])
        save_minutiae(mnt_stft, fname)

        # Extracting minutiae from the contrast enhanced STFT image
        mnt_stft = self.minu_model[0].run_whole_image(ct_img_stft,
                                                      minu_thr=0.1)
        minutiae_sets.append(mnt_stft)

        if show_minutiae:
            fname = root_name + '_ctstft_mnt.jpeg'
            show.show_minutiae_sets(ct_img_stft, [input_minu, mnt_stft],
                                    mask=None, block=block, fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_ctstft_mnt.txt" % img_name[0])
        save_minutiae(mnt_stft, fname)

        # Extracting minutiae from the enhanced image
        mnt_aec = self.minu_model[1].run_whole_image(aec_img, minu_thr=0.25)
        mnt_aec = self.remove_spurious_minutiae(mnt_aec, mask)
        minutiae_sets.append(mnt_aec)

        if show_minutiae:
            fname = root_name + '_aec_mnt.jpeg'
            show.show_minutiae_sets(aec_img, [input_minu, mnt_aec],
                                    mask=mask, block=block, fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_aec_mnt.txt" % img_name[0])
        save_minutiae(mnt_aec, fname)

        # Enhance gaussian contrast image
        enh_contrast_img = filtering.gabor_filtering_pixel2(
            ct_img_g, dir_map_aec + math.pi / 2,
            fre_map_aec, mask=np.ones((h, w)), block_size=16, angle_inc=3
        )

        # saving enhanced contrast gaussian image
        fname = os.path.join(output_dir, "%s_enhctg.%s" % img_name)
        cv2.imwrite(fname, enh_contrast_img)

        # Extracting minutiae from the enhanced contrast gaussian image
        mnt_contrast = self.minu_model[1].run_whole_image(
            enh_contrast_img, minu_thr=0.25)
        mnt_contrast = self.remove_spurious_minutiae(mnt_contrast, mask)
        minutiae_sets.append(mnt_contrast)

        if show_minutiae:
            fname = root_name + '_enhctg_mnt.jpeg'
            show.show_minutiae_sets(enh_contrast_img,
                                    [input_minu, mnt_contrast],
                                    mask=mask, block=block,
                                    fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_enhctg_mnt.txt" % img_name[0])
        save_minutiae(mnt_contrast, fname)

        # Enhance gaussian contrast image CENATAV
        gfilter = loggabor.LogGaborFilter(dir_map_aec + math.pi / 2,
                                          fre_map_aec, mask=np.ones((h, w)))
        enh_contrast_img, thr = gfilter.apply(ct_img_g)

        # saving enhanced contrast gaussian image CENATAV
        fname = os.path.join(output_dir, "%s_enhctg2.%s" % img_name)
        cv2.imwrite(fname, enh_contrast_img)

        # saving binarized enhanced contrast gaussian image CENATAV
        fname = os.path.join(output_dir, "%s_enhctg2_bin.%s" % img_name)
        bin_image = (enh_contrast_img >= thr).astype(np.uint8) * 255
        bin_image[mask == 0] = 0
        cv2.imwrite(fname, bin_image)

        # Extracting minutiae from the enhanced contrast gaussian image CENATAV
        mnt_contrast = self.minu_model[1].run_whole_image(
            enh_contrast_img, minu_thr=0.25)
        mnt_contrast = self.remove_spurious_minutiae(mnt_contrast, mask)

        if show_minutiae:
            fname = root_name + '_enhctg_mnt2.jpeg'
            show.show_minutiae_sets(enh_contrast_img,
                                    [input_minu, mnt_contrast],
                                    mask=mask, block=block,
                                    fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_enhctg_mnt2.txt" % img_name[0])
        save_minutiae(mnt_contrast, fname)

        # Enhance texture image
        enh_texture_img = filtering.gabor_filtering_pixel2(
            tex_img, dir_map_aec + math.pi / 2, fre_map_aec,
            mask=np.ones((h, w)), block_size=16, angle_inc=3
        )

        # saving enhanced texture image
        fname = os.path.join(output_dir, "%s_enhtext.%s" % img_name)
        cv2.imwrite(fname, enh_texture_img)

        # Extracting minutiae from the enhanced texture image
        mnt_texture = self.minu_model[1].run_whole_image(
            enh_texture_img, minu_thr=0.25)
        mnt_texture = self.remove_spurious_minutiae(mnt_texture, mask)
        minutiae_sets.append(mnt_texture)

        if show_minutiae:
            fname = root_name + '_enhtext_mnt.jpeg'
            show.show_minutiae_sets(enh_texture_img, [input_minu, mnt_texture],
                                    mask=mask, block=block, fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_enhtext_mnt.txt" % img_name[0])
        save_minutiae(mnt_texture, fname)

        # Enhance texture image CENATAV
        enh_texture_img, thr = gfilter.apply(tex_img)

        # saving enhanced texture image CENATAV
        fname = os.path.join(output_dir, "%s_enhtext2.%s" % img_name)
        cv2.imwrite(fname, enh_texture_img)

        # saving binarized enhanced texture image CENATAV
        fname = os.path.join(output_dir, "%s_enhtext2_bin.%s" % img_name)
        bin_image = (enh_texture_img >= thr).astype(np.uint8) * 255
        bin_image[mask == 0] = 0
        cv2.imwrite(fname, bin_image)

        # Extracting minutiae from the enhanced texture image CENATAV
        mnt_texture = self.minu_model[1].run_whole_image(
            enh_texture_img, minu_thr=0.25)
        mnt_texture = self.remove_spurious_minutiae(mnt_texture, mask)

        if show_minutiae:
            fname = root_name + '_enhtext_mnt2.jpeg'
            show.show_minutiae_sets(enh_texture_img, [input_minu, mnt_texture],
                                    mask=mask, block=block, fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_enhtext_mnt2.txt" % img_name[0])
        save_minutiae(mnt_texture, fname)

        # Latent template
        latent_template = template.Template()

        # template set 1: no ROI and enhancement are required
        # texture image is used for coarse segmentation
        descriptor_imgs = []

        # Descriptors
        descriptor_imgs.append(stft_img)
        descriptor_imgs.append(tex_img)
        descriptor_imgs.append(enh_texture_img)
        descriptor_imgs.append(enh_contrast_img)

        # Common minutiae sets
        mnt2 = self.get_common_minutiae(minutiae_sets, thr=2)
        mnt3 = self.get_common_minutiae(minutiae_sets, thr=3)

        minutiae_sets.append(mnt3)
        minutiae_sets.append(mnt2)

        if show_minutiae:
            fname = root_name + '_common1_mnt.jpeg'
            show.show_minutiae_sets(img, [input_minu, mnt2],
                                    mask=mask, block=block, fname=fname)

            fname = root_name + '_common2_mnt.jpeg'
            show.show_minutiae_sets(img, [input_minu, mnt3],
                                    mask=mask, block=block, fname=fname)

        # saving minutiae
        fname = os.path.join(output_dir, "%s_common1_mnt.txt" % img_name[0])
        save_minutiae(mnt2, fname)
        fname = os.path.join(output_dir, "%s_common2_mnt.txt" % img_name[0])
        save_minutiae(mnt3, fname)

        # End minutiae extraction
        end = timer()
        print('Time for minutiae extraction: %f' % (end - start))

        # Extracting minutiae descriptors
        start = timer()
        for mnt in minutiae_sets:
            for des_img in descriptor_imgs:
                des = descriptor.minutiae_descriptor_extraction(
                    des_img, mnt, self.patch_types, self.des_models,
                    self.patchIndexV, batch_size=128
                )
                minu_template = template.MinuTemplate(
                    h=h, w=w, blkH=blk_h, blkW=blk_w, minutiae=mnt,
                    des=des, oimg=dir_map_aec, mask=mask
                )
                latent_template.add_minu_template(minu_template)
        end = timer()
        print('Time for minutiae descriptor generation: %f' % (end - start))

        # Virtual minutiae
        start = timer()
        stride = 16
        x = np.arange(24, w - 24, stride)
        y = np.arange(24, h - 24, stride)
        virtual_minutiae = []
        dist_euclid = scipy.ndimage.morphology.distance_transform_edt(mask)
        for y_i in y:
            for x_i in x:
                if (dist_euclid[y_i][x_i] <= 16):
                    continue
                ofY = int(y_i / 16)
                ofX = int(x_i / 16)

                ori = -dir_map_aec[ofY][ofX]
                virtual_minutiae.append([x_i, y_i, ori])
                virtual_minutiae.append([x_i, y_i, math.pi + ori])
        virtual_minutiae = np.asarray(virtual_minutiae)

        # Texture templates
        texture_template = []
        if len(virtual_minutiae) > 3:
            virtual_des = descriptor.minutiae_descriptor_extraction(
                enh_contrast_img, virtual_minutiae, self.patch_types,
                self.des_models, self.patchIndexV, batch_size=128,
                patch_size=96
            )

            texture_template = template.TextureTemplate(
                h=h, w=w, minutiae=virtual_minutiae,
                des=virtual_des, mask=None
            )

            latent_template.add_texture_template(texture_template)

        end = timer()

        print('Time for texture template generation: %f' % (end - start))
        return latent_template, texture_template

    def get_common_minutiae(self, minutiae_sets, thr=3):
        """Return common minutiae among different minutiae sets"""
        nrof_minutiae_sets = len(minutiae_sets)

        init_ind = 3
        if len(minutiae_sets[init_ind]) == 0:
            return []
        mnt = list(minutiae_sets[init_ind][:, :4])
        count = list(np.ones(len(mnt),))
        for i in range(0, nrof_minutiae_sets):
            if i == init_ind:
                continue
            for j in range(len(minutiae_sets[i])):
                x2 = minutiae_sets[i][j, 0]
                y2 = minutiae_sets[i][j, 1]
                ori2 = minutiae_sets[i][j, 2]
                found = False
                for k in range(len(mnt)):
                    x1 = mnt[k][0]
                    y1 = mnt[k][1]
                    ori1 = mnt[k][2]
                    dist = math.sqrt(
                        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
                    )

                    ori_dist = math.fabs(ori1 - ori2)
                    if ori_dist > math.pi / 2:
                        ori_dist = math.pi - ori_dist
                    if dist <= 10 and ori_dist < math.pi / 6:
                        count[k] += 1
                        found = True
                        break
                if not found:
                    mnt.append([x2, y2, ori2, 1])
                    count.append(1)
        count = np.asarray(count)
        ind = np.where(count >= thr)[0]
        mnt = np.asarray(mnt)
        mnt = mnt[ind, :]
        mnt[:, 3] = 1
        return mnt

    def remove_spurious_minutiae(self, mnt, mask):
        """Remove spurious minutiae in the borders of a mask"""
        minu_num = len(mnt)
        if minu_num <= 0:
            return mnt
        flag = np.ones((minu_num,), np.uint8)
        h, w = mask.shape[:2]
        R = 10
        for i in range(minu_num):
            x = mnt[i, 0]
            y = mnt[i, 1]
            x = np.int(x)
            y = np.int(y)
            if x < R or y < R or x > w - R - 1 or y > h - R - 1:
                flag[i] = 0
            elif(mask[y - R, x - R] == 0 or mask[y - R, x + R] == 0 or
                 mask[y + R, x - R] == 0 or mask[y + R, x + R] == 0):
                flag[i] = 0
        mnt = mnt[flag > 0, :]
        return mnt

    def feature_extraction(self, image_dir, template_dir=None,
                           minu_path=None, N1=0, N2=258):
        """Feature extraction for a batch of images"""
        img_files = glob.glob(image_dir + '*.bmp')
        assert(len(img_files) > 0)

        if not os.path.exists(template_dir):
            os.makedirs(template_dir)

        img_files.sort()
        if minu_path is not None:
            minu_files = glob.glob(minu_path + '*.txt')
            minu_files.sort()

        for i, img_file in enumerate(img_files):
            print(i, img_file)
            img_name = os.path.basename(img_file)
            if template_dir is not None:
                fname = template_dir + os.path.splitext(img_name)[0] + '.dat'
                if os.path.exists(fname):
                    continue

            start = timeit.default_timer()
            if minu_path is not None and len(minu_files) > i:
                latent_t, texture_t = self.feature_extraction_single_latent(
                    img_file, output_dir=template_dir, show_processes=False,
                    minu_file=minu_files[i], show_minutiae=False
                )
            else:
                latent_t, texture_t = self.feature_extraction_single_latent(
                    img_file, output_dir=template_dir, show_processes=False,
                    show_minutiae=False
                )

            stop = timeit.default_timer()
            print(stop - start)

            fname = template_dir + os.path.splitext(img_name)[0] + '.dat'
            template.Template2Bin_Byte_TF_C(fname, latent_t,
                                            isLatent=True)


def get_feature_extractor():
    """Returns an instance of the latent feature extractor class"""
    global config
    if not config:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(dir_path + 'config') as config_file:
            config = json.load(config_file)

    des_model_dirs = []
    patch_types = []
    model_dir = config['DescriptorModelPatch2']
    des_model_dirs.append(model_dir)
    patch_types.append(2)
    model_dir = config['DescriptorModelPatch8']
    des_model_dirs.append(model_dir)
    patch_types.append(8)
    model_dir = config['DescriptorModelPatch11']
    des_model_dirs.append(model_dir)
    patch_types.append(11)

    # minutiae extraction model
    minu_model_dirs = []
    minu_model_dirs.append(config['MinutiaeExtractionModelLatentSTFT'])
    minu_model_dirs.append(config['MinutiaeExtractionModel'])

    # enhancement model
    enhancement_model_dir = config['EnhancementModel']

    return FeatureExtraction_Latent(
        patch_types=patch_types,
        des_model_dirs=des_model_dirs,
        enhancement_model_dir=enhancement_model_dir,
        minu_model_dirs=minu_model_dirs
    )


def main(image_dir, template_dir):
    """Main entry-point for processing directories"""

    lf_latent = get_feature_extractor()

    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

    print("Starting feature extraction (batch)...")
    lf_latent.feature_extraction(image_dir=image_dir,
                                 template_dir=template_dir,
                                 minu_path=config['MinuPath'])


def main_single_image(image_file, template_dir):
    """Main entry-point for processing a single image"""

    lf_latent = get_feature_extractor()

    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

    print("Latent query: " + image_file)
    print("Starting feature extraction (single latent)...")
    l_template, _ = lf_latent.feature_extraction_single_latent(
        image_file, output_dir=template_dir, show_processes=False,
        minu_file=None, show_minutiae=False
    )
    path = os.path.splitext(os.path.basename(image_file))[0]
    fname = template_dir + path + '.dat'
    template.Template2Bin_Byte_TF_C(fname, l_template, isLatent=True)

    return fname


def parse_arguments(argv):
    """Script argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu', help='comma separated list of GPU(s) to use.', default='0'
    )
    parser.add_argument(
        '--tdir', type=str,
        help='Path to location where extracted templates should be stored'
    )
    parser.add_argument(
        '--idir', type=str, help='Path to directory containing input images'
    )
    parser.add_argument('--i', type=str, help='Path to single input image')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # Parsing arguments
    args = parse_arguments(sys.argv[1:])

    # Working path
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Loading configuration file
    with open(dir_path + '/afis.config') as config_file:
        config = json.load(config_file)

    # Setting GPUs to use
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.i:  # Handling a single image

        # Setting template directory
        t_dir = args.tdir if args.tdir else config['LatentTemplateDirectory']
        template_fname = main_single_image(args.i, t_dir)

        print("Finished feature extraction. Starting dimensionality reduction")
        descriptor_DR.template_compression_single(
            input_file=template_fname, output_dir=t_dir,
            model_path=config['DimensionalityReductionModel'],
            isLatent=True, config=None
        )
        print("Finished dimensionality reduction. Starting product quantization...")
        descriptor_PQ.encode_PQ_single(
            input_file=template_fname,
            output_dir=t_dir, fprint_type='latent'
        )
        print("Finished product quantization. Exiting...")

    else:   # Handling a directory of images

        tdir = args.tdir if args.tdir else config['LatentTemplateDirectory']
        main(args.idir, tdir)

        print("Finished feature extraction. Starting dimensionality reduction...")
        descriptor_DR.template_compression(
            input_dir=tdir, output_dir=tdir,
            model_path=config['DimensionalityReductionModel'],
            isLatent=True, config=None
        )
        print("Finished dimensionality reduction. Starting product quantization...")
        descriptor_PQ.encode_PQ(
            input_dir=tdir, output_dir=tdir, fprint_type='latent'
        )
        print("Finished product quantization. Exiting...")
