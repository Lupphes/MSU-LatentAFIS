import sys
import json
import os
from msu_latentafis.descriptor_DR import template_compression_single, template_compression
from msu_latentafis.descriptor_PQ import encode_PQ, encode_PQ_single
from msu_latentafis.extraction_latent import main_single_image, parse_arguments, main

# Parsing arguments
args = parse_arguments(sys.argv[1:])

# Working path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Loading configuration file
with open(dir_path + '/afis.config') as config_file:
    config = json.load(config_file)

# Setting GPUs to use
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

tdir = args.tdir if args.tdir else config['LatentTemplateDirectory']
test = main(args.idir, tdir, args.edited_mnt)

print("Starting dimensionality reduction...")
template_compression(
    input_dir=tdir, output_dir=tdir,
    model_path=config['DimensionalityReductionModel'],
    isLatent=True, config=None
)
print("Starting product quantization...")
encode_PQ(
    input_dir=tdir, output_dir=tdir, fprint_type='latent'
)

print("Exiting...")
