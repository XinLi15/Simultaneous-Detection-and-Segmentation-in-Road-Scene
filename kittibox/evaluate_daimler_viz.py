#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 


"""
responsible for acquiring bounding boxes on test images


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

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.INFO,
					stream=sys.stdout)


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
		'--labels_dir',required=True,help = '')
	parser.add_argument(
		'--weights_dir',required=True,help = '')
	parser.add_argument(
		'--hypes',required=True,help = '')
	args = parser.parse_args()
	return args

def write_rects(rects, filename):
	with open(filename, 'w') as f:
		for rect in rects:
			string = "Car 0 1 0 %f %f %f %f 0 0 0 0 0 0 0 %f" % \
				(rect.x1, rect.y1, rect.x2, rect.y2, rect.score)
			print(string, file=f)	

def _draw_rect(draw, rect, color):
	left = rect.cx-int(rect.width/2)
	bottom = rect.cy+int(rect.height/2)
	right = rect.cx+int(rect.width/2)
	top = rect.cy-int(rect.height/2)
	rect_cords = ((left, top), (left, bottom),
				  (right, bottom), (right, top),
				  (left, top))
	draw.line(rect_cords, fill=color, width=2)

def write_rects(rects, filename):
	with open(filename, 'w') as f:
		for rect in rects:
			string = "Car 0 1 0 %f %f %f %f 0 0 0 0 0 0 0 %f" % \
				(rect.x1, rect.y1, rect.x2, rect.y2, rect.score)
			print(string, file=f)

def maybe_download_and_extract(runs_dir):
	logdir = os.path.join(runs_dir, default_run)

	if os.path.exists(logdir):
		# weights are downloaded. Nothing to do
		return

	if not os.path.exists(runs_dir):
		os.makedirs(runs_dir)

	import zipfile
	download_name = tv_utils.download(weights_url, runs_dir)

	logging.info("Extracting KittiBox_pretrained.zip")

	zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

	return

def get_daimler_files(root_dir, file_ext):
	""" takes filenames and ids to dataframe"""
	lFiles = []
	print(root_dir)
	for file in glob.glob(root_dir+"*"+file_ext):
		lFiles.append(os.path.join(root_dir, file))

	return lFiles

def get_df(fpath):
	paths = []
	ids = []
	_ = [paths.append(file) for file in glob.glob(fpath)]
	_ = [ids.append(os.path.basename(path).rsplit('.')[0]) for path in paths]
	return pd.DataFrame({'id': ids,'file': paths })

def load_daimler_weights(checkpoint_dir,sess, saver):
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		logging.info(ckpt.model_checkpoint_path)
		file = os.path.basename(ckpt.model_checkpoint_path)
		checkpoint_path = os.path.join(checkpoint_dir, file)
		saver.restore(sess, checkpoint_path)
		return int(file.split('-')[1])

def main(_):
	args= parse_args()
	label_dir = args.labels_dir
	out_dir = args.outdir
	tv_utils.set_gpus_to_use()

	txt_out = os.path.join(out_dir,'val_out_txt')
	try:os.makedirs(txt_out)
	except OSError: 
		shutil.rmtree(txt_out)
		os.makedirs(txt_out)
	feat_out = os.path.join(out_dir,'val_out_features')
	try:os.makedirs(feat_out)
	except OSError: 
		shutil.rmtree(feat_out)
		os.makedirs(feat_out)

	if FLAGS.logdir is None:
		# Download and use weights from the MultiNet Paper
		if 'TV_DIR_RUNS' in os.environ:
			runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
									'KittiBox')
			print("FOUND")
		else:
			runs_dir = 'RUNS'
		maybe_download_and_extract(runs_dir)
		logdir = os.path.join(runs_dir, default_run)
	else:
		logging.info("Using weights found in {}".format(FLAGS.logdir))
		logdir = FLAGS.logdir

	# Loading hyperparameters from logdir
	hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

	logging.info("Hypes loaded successfully.")

	# Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
	modules = tv_utils.load_modules_from_logdir(logdir)
	logging.info("Modules loaded successfully. Starting to build tf graph.")

	images_path = get_daimler_files(args.images_dir, 'png')
	images = get_df(args.images_dir + '*png')
	labels = get_df(args.labels_dir  + '*txt')

	# Create tf graph and build module.
	with tf.Graph().as_default():
		# Create placeholder for input
		image_pl = tf.placeholder(tf.float32)
		image = tf.expand_dims(image_pl, 0)
		# build Tensorflow graph using the model from logdir
		intermediate, prediction = core.build_inference_graph(hypes, modules,
												image=image)

		logging.info("Graph build successfully.")

		# Create a session for running Ops on the Graph.
		sess = tf.Session()
		saver = tf.train.Saver()
		# checkpoint_dir = '/home/asus/checkpoints/atrous/'
		checkpoint_dir = args.weights_dir
		load_daimler_weights(checkpoint_dir,sess,saver)
		logging.info("Weights loaded successfully.")

	print([n.name for n in tf.get_default_graph().as_graph_def().node])

	logging.info("Starting inference using {} as input".format(images_path[0]))
	im_dir = os.path.join(out_dir,'images')
	try:os.makedirs(im_dir)
	except OSError: 
		shutil.rmtree(im_dir)
		os.makedirs(im_dir)
	# Load and resize input image

	for i,im in enumerate(images_path):
		orig_im = images_path[i]
		name = os.path.basename(orig_im).rsplit('.')[0]
		image = scp.misc.imread(orig_im)
		image = scp.misc.imresize(image, (hypes["image_height"],
										  hypes["image_width"]),
								  interp='cubic')
		feed = {image_pl: image}
		deep_feat = intermediate['unpooled']
		early_feat = intermediate['early_feat']

		early_feat_eval, deep_feat_eval = sess.run([early_feat, deep_feat],feed_dict = feed)

		np.save(os.path.join( feat_out, name + '_eval_earl'), early_feat_eval)
		np.save(os.path.join( feat_out, name + '_deep_feat'), deep_feat_eval)

		pred_boxes = prediction['pred_boxes_new']
		pred_confidences = prediction['pred_confidences']
		(np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
														 pred_confidences],
														feed_dict=feed)
		image = np.copy(image)
		rnn_len=1
		min_conf=0.50
		use_stitching = True
		tau=hypes['tau']
		tau = 0.25
		show_removed = False
		color_removed=(0, 0, 255)
		color_acc=(0, 255,0)
		H = json.load(open(args.hypes))
		num_cells = H["grid_height"] * H["grid_width"]
		boxes_r = np.reshape(np_pred_boxes, (-1,
									 H["grid_height"],
									 H["grid_width"],
									 rnn_len,
									 4))
		confidences_r = np.reshape(np_pred_confidences, (-1,
												 H["grid_height"],
												 H["grid_width"],
												 rnn_len,
												 H['num_classes']))

		cell_pix_size = H['region_size']
		all_rects = [[[] for _ in range(H["grid_width"])] for _ in range(H["grid_height"])]
		for n in range(rnn_len):
			for y in range(H["grid_height"]):
				for x in range(H["grid_width"]):
					bbox = boxes_r[0, y, x, n, :]
					abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
					abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
					w = bbox[2] 
					h = bbox[3] 
					conf = np.max(confidences_r[0, y, x, n, 1:])
					all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
		all_rects_r = [r for row in all_rects for cell in row for r in cell]
		if use_stitching:
			from utils.stitch_wrapper import stitch_rects
			acc_rects = stitch_rects(all_rects, tau)
		else:
			acc_rects = all_rects_r

		if not show_removed:
			all_rects_r = []

		pairs = [(all_rects_r, color_removed), (acc_rects, color_acc)]
		im = Image.fromarray(image.astype('uint8'))
		draw = ImageDraw.Draw(im)
		for rect_set, color in pairs:
			for rect in rect_set:
				if rect.confidence > min_conf:
					_draw_rect(draw, rect, color)

		image = np.array(im).astype('float32')
		rects = []
		for rect in acc_rects:
			r = al.AnnoRect()
			r.x1 = rect.cx - rect.width/2.
			r.x2 = rect.cx + rect.width/2.
			r.y1 = rect.cy - rect.height/2.
			r.y2 = rect.cy + rect.height/2.
			r.score = rect.true_confidence
			rects.append(r)

		pred_anno = AnnLib.Annotation()
		pred_anno = utils.train_utils.rescale_boxes((
			H["image_height"],
			H["image_width"]),
			pred_anno, image.shape[0],
			image.shape[1])
		val_file = os.path.join(txt_out, name + '.txt')
		write_rects(rects, val_file)


	

if __name__ == '__main__':
	tf.app.run()
