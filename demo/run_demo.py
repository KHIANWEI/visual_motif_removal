from __future__ import print_function
import os
import zipfile
import shutil
from urllib.request import urlopen
from utils.train_utils import load_globals, init_folders
from utils.visualize_utils import run_net
from utils.visualize_utils import my_save_test_images
from utils.train_utils import *
import torch


# consts
DEVICE = torch.device('cuda:0')
ROOT_PATH = '.'
NET_FOLDER_PATH = '%s/ManualTestNet' % ROOT_PATH
DATA_URL = 'http://pxcm.org/motif/demo.zip'
DATA_ZIP_FILE = '%s/demo.zip' % ROOT_PATH
NET_PATH = '%s/ManualTestNet/net_baseline_45.pth' % ROOT_PATH
TEST_PATH = '%s/ManualTest' % ROOT_PATH
RECONSTRUCTED_PATH = '%s/ManualTestOP' % ROOT_PATH


def download_data():
    if not os.path.exists(NET_PATH):
        if not os.path.exists(DATA_ZIP_FILE):
            print("Downloading zipped data to " + DATA_ZIP_FILE + " ...")
            resp = urlopen(DATA_URL)
            with open(DATA_ZIP_FILE, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            resp.close()
        print("Unzipping " + DATA_ZIP_FILE)
        with zipfile.ZipFile(DATA_ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall('./')
        print("... done unzipping")
    if os.path.exists(DATA_ZIP_FILE):
        os.remove(DATA_ZIP_FILE)
    print("Will use net_baseline in %s" % ROOT_PATH)


def run_demo():
#    download_data()
    init_folders(RECONSTRUCTED_PATH)
    opt = load_globals(NET_FOLDER_PATH, {}, override=False)
#    images_loader = my_loader(opt, cache_root=temp_images_path)
    MY_NET_PATH = './ManualTestNet'
    SAVE_IMAGE_NAME = './ManualTestOP/output.png'
    net = init_nets(opt, MY_NET_PATH, DEVICE, tag='60')

    my_save_test_images(net, TEST_PATH, DEVICE, SAVE_IMAGE_NAME)
#    run_net(opt, DEVICE, ROOT_PATH, TEST_PATH, RECONSTRUCTED_PATH, 'demo')
    print("Reconstructed images are at %s" % RECONSTRUCTED_PATH)

if __name__ == '__main__':
    run_demo()
