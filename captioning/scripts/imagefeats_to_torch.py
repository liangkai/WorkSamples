"""
Converts coco dataset image features into 
torch-compatible format
"""

from __future__ import print_function
import urllib2
import sys
import os
import shutil
import zipfile
import gzip
import cPickle as pickle

def convert_imagefeats(dirpath, feats_path, save_path):
	final_path = os.path.join(dirpath, feats_path)
	print(final_path)
	files = pickle.load(open(final_path))
	with open(save_path,"w") as f:
		f.write("\n".join(" ".join(map(str, x)) for x in (a,b)))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # data
    train_dir = os.path.join(base_dir, 'data/coco/train')
    test_dir = os.path.join(base_dir, 'data/coco/test')

    convert_imagefeats(test_dir, "googlenet_feats.cPickle", "googlenet_feats.txt")
    convert_imagefeats(train_dir, "googlenet_feats.cPickle", "googlenet_feats.txt")

