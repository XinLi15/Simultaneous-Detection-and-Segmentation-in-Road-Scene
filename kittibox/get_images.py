#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Anna Deichler


"""
Evaluates 


based on https://github.com/MarvinTeichmann/KittiBox 
Original author: Martin Teichmann
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageDraw


import json
import logging
import os
import sys
import argparse
import glob
import subprocess
import shutil
import collections
import pandas as pd
import  cv2

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.INFO,
					stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')
from utils import train_utils as kittibox_utils
from utils.data_utils import (annotation_jitter, annotation_to_h5)
from utils.annolist import AnnotationLib as al
from utils.annolist import AnnotationLib as AnnLib
import utils.train_utils

from utils.rect import Rect

try:
	# Check whether setup was done correctly
	import tensorvision.utils as tv_utils
	import tensorvision.core2 as core
except ImportError:
	# You forgot to initialize submodules
	logging.error("Could not import the submodules.")
	logging.error("Please execute:"
				  "'git submodule update --init --recursive'")
	exit(1)


flags.DEFINE_string('logdir', None,
					'Path to logdir.')
flags.DEFINE_string('input_image', None,
					'Image to apply KittiBox.')
flags.DEFINE_string('output_image', None,
					'Image to apply KittiBox.')


default_run = 'KittiBox_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
			   "pub/mttt2/models/KittiBox_pretrained.zip")



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--images_dir',required=True,help = '')
	parser.add_argument(
		'--outdir',required=True,help = '')
	parser.add_argument(
		'--val_out',required=True,help = '')
	parser.add_argument(
		'--groundtruth',required=True,help = '')
	args = parser.parse_args()
	return args


def save_labeled_image(row, outdir):
	im = cv2.imread(row[1]['file'])
	with open(row[1]['file_labelv']) as f:
		records = f.readlines()
		for record in records: 
			txt_data= record.rsplit(' ')
			if txt_data[0]=='Car':
				xmin,ymin,xmax,ymax = np.array(txt_data[4:8]).astype(float)
				cv2.rectangle(im, (int(float(xmin)),int(float(ymin))),(int(float(xmax)),int(float(ymax))), (255,0,0),1)
	with open(row[1]['file_labelg']) as f:
		records = f.readlines()
		for record in records: 
			txt_data= record.rsplit(' ')
			if txt_data[0]=='Car':
				xmin,ymin,xmax,ymax = np.array(txt_data[4:8]).astype(float)
				cv2.rectangle(im, (int(float(xmin)),int(float(ymin))),(int(float(xmax)),int(float(ymax))), (255,255,255),1)
	

		scp.misc.imsave(os.path.join(outdir, 'images' , row[1]['id']+'.png' ),im)
		
	
def get_df(fpath):
	paths = []
	ids = []
	_ = [paths.append(file) for file in glob.glob(fpath)]
	_ = [ids.append(os.path.basename(path).rsplit('.')[0]) for path in paths]
	return pd.DataFrame({'id': ids,'file': paths })


def main(_):
	args= parse_args()
	images = get_df(args.images_dir + '*png')
	labels_gr = get_df(args.groundtruth + '*txt')
	labels_vl = get_df(args.val_out  + '*txt')
	df = pd.merge(labels_vl, labels_gr, on=['id'],suffixes=('_labelv', '_labelg'))
	df = pd.merge(df, images, on=['id'])
	print(df.keys())
	[save_labeled_image(row, args.outdir) for row in df.iterrows()]

if __name__ == '__main__':
	tf.app.run()
