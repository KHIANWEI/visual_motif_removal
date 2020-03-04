from __future__ import print_function
from utils.train_utils import load_globals, init_folders, init_nets
from utils.visualize_utils import generate_image
import torch
import sys
import time
import os


# consts
OUTPUT_FOLDER = './GeneratedOutputs'

def gen_output(input_path, net_folder, tag=''):
    my_device = torch.device('cuda:0')
    init_folders(OUTPUT_FOLDER)
    opt = load_globals(net_folder, {}, override=False)
    net = init_nets(opt, net_folder, my_device, tag)
    output_path = '%s/OUTPUT_%s.png' % (OUTPUT_FOLDER, time.strftime("%Y%m%d-%H%M%S"))
    print('Relative Path: %s' % output_path)
    generate_image(net, my_device, input_path, output_path)
    print('Absolute Path: %s' % os.path.abspath(output_path))
    return output_path

if __name__ == '__main__':
    gen_output(sys.argv[1], sys.argv[2], sys.argv[3])
