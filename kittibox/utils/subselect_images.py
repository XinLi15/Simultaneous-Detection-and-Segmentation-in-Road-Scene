import argparse
import os
import numpy as np
import pandas as pd
import re
import cv2
from collections import Counter
import glob
import shutil

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--labels_dir',required=True,help = '')
	parser.add_argument(
		'--images_dir',required=True,help = '')
	parser.add_argument(
		'--outdir', required=True, help = 'path/train.txt')
	parser.add_argument(
		'--N', required=True, help = 'number of images to select')
	args = parser.parse_args()
	return args

def get_daimler_files(root_dir, name_ext, file_ext):
	""" takes filenames and ids to dataframe"""
	lFiles = []
	print(root_dir)
	for file in glob.glob(root_dir+"*"+file_ext):
		lFiles.append(os.path.join(root_dir, file))
	if(len(lFiles)==0):raise Exception('no data found!') 
	idFiles=[re.findall(r"rDataset_(.*?)" +re.escape(name_ext), fname)[0] for fname in lFiles]
	dfFiles = pd.DataFrame(
	{     'id': idFiles,'file': lFiles })

	dfFiles = dfFiles.sort_index(ascending=True)
	return dfFiles


def main(**kwargs):
	label_dir = kwargs.pop('labels_dir')
	image_dir = kwargs.pop('images_dir')
	N = int(kwargs.pop('N'))
	outdir = kwargs.pop('outdir')
	if not os.path.exists(outdir):
		os.makedirs(outdir)	
	images = get_daimler_files(image_dir,'leftImg8bit','png')
	labels = get_daimler_files(label_dir,'labelData','json')
	df = pd.merge(labels, images, on=['id'],suffixes=('_label', '_image'))
	df['group']=[item.rsplit('_')[-3] for item in df['id']]
	dict_counter = Counter(df['group'])
	#select images randomly from all timestamps
	df_sample = pd.DataFrame(columns = df.columns)
	n = N/len(dict_counter.keys())
	for key in dict_counter.keys():
	    _ = df[df['group']==key]
	    _ = _.sample(min(n,len(_)))
	    df_sample = df_sample.append(_,ignore_index = True)
	if(len(df_sample)) < N:
		_ = df.sample(N-len(df_sample))
		df_sample = df_sample.append(_,ignore_index = True)

	for file_name in df_sample['file_image']:
	    if (os.path.isfile(file_name)):
	        shutil.copy(file_name, outdir)

	
if __name__ == '__main__':
	args_namespace = parse_args()
	kwargs = vars(args_namespace)
	main(**kwargs)