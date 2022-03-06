import sys
import json
import os
from msu_latentafis.descriptor_DR import template_compression
from msu_latentafis.descriptor_PQ import encode_PQ
from msu_latentafis.extraction_rolled import parse_arguments, FeatureExtractionRolled

# Parsing arguments
args = parse_arguments(sys.argv[1:])

# Configuring CUDA for GPUs
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Working path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Loading configuration file
with open(dir_path + '/afis.config') as config_file:
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
    enhancement_model_dir=enhance_model_dir,
)

# Feature extraction
print("Starting feature extraction (batch)...")
lf_rolled.feature_extraction(
    image_dir=img_dir, template_dir=temp_dir, enhancement=args.enhance,
    img_type=args.itype, edited_mnt=args.edited_mnt
)

# Blocking this piece of code because it is buggy
print("Finished feature extraction. Starting dimensionality reduction...")
template_compression(
    input_dir=temp_dir, output_dir=temp_dir,
    model_path=config['DimensionalityReductionModel'],
    isLatent=False, config=None
)

print("Finished dimensionality reduction. Starting product quantization..")
encode_PQ(
    input_dir=temp_dir, output_dir=temp_dir, fprint_type='rolled')
print("Finished product quantization. Exiting...")
