#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Transforms daimler training data for detection with KittiBox
images and labels are changed for match KITTI dataset ratio
images and labels are renamed (KittiBox assumes 6 digit numbe names)
Used for transforming Daimler dataset to KITTI format.
Usage:
python convert_daimler_data.py --labels_dir /path/to/daimler/labels/ --images_dir 
                /path/to/daimler/images --outdir path/desired/output/location/
""" 
import argparse
import os
import numpy as np
import pandas as pd
import re
import cv2
import glob
import json
import shutil
import copy
from sklearn.utils import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--labels_dir',required=True,help = '')
    parser.add_argument(
        '--images_dir',required=True,help = '')
    parser.add_argument(
        '--outdir', required=True, help = 'path/train.txt')

    args = parser.parse_args()
    return args

def get_df(fpath):
    """ 
    takes filenames in given directory to dataframe with IDs
    """
    paths = []
    ids = []
    _ = [paths.append(file) for file in glob.glob(fpath)]
    _ = [ids.append(os.path.basename(path).rsplit('.')[0]) for path in paths]
    return pd.DataFrame({'id': ids,'file': paths })

def save_resized_images(f_change, pad, row, image_folder):
    img = row['images']
    img = cv2.resize(img, (0,0), fx=float(f_change), fy=float(f_change)) 
    resized_img = cv2.copyMakeBorder(img, 0, 0, int(pad), int(pad), cv2.BORDER_CONSTANT)
    imagename=str(row['new_ID'])+'.png'
    image_file = os.path.join(image_folder, imagename )
    cv2.imwrite(image_file,resized_img)

def get_data_row(item):
    """
    convert daimler dataset label information to KITTI format 
    keeps occlusion, class label, bounding box information, others discarded
    """
    dict_occlusion = { '10':1, '40':2  }
    flatten = lambda *n: (e for a in n
        for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    row =[]
    if item['identity'] == 'cyclist': ID = 'Car'
    row.append(str(ID))
    row.append(str('%.2f' % 0.0))
    tags = item['tags']
    _ = [int(tags[i].rsplit('>')[-1]) for i,s in enumerate(item['tags']) if 'occluded' in s]
    if len(item.get('tags', '')) ==0:
        occlusion = 0
    else: 
        tags_data = ([int(item['tags'][i].rsplit('>')[-1]) for i,s in enumerate(item['tags']) if 'occluded' in s])
        if len(tags_data)==1: occlusion = dict_occlusion[str(tags_data[0])]
        else: occlusion = 0
    row.append(str(occlusion))
    alpha = 0
    row.append(str('%.2f' % 0.0))
    left, top, right, bottom = item['mincol'], item['minrow'], item['maxcol'], item['maxrow']
    [row.append(str('%.2f' % item))for item in [left,top,right,bottom]]
    row.append([str('%.2f' % 0.0) for x in range(6)])
    row.append([str('%.2f' % 0.0) for x in range(1)])
    row = list(flatten(row))
    return ' '.join(str(v) for v in row)

def create_dirs(outdir):
    """
    create output directory
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    label_folder = os.path.join(outdir,'labels')
    image_folder = os.path.join(outdir,'images')
    try:os.makedirs(label_folder)
    except OSError: 
        shutil.rmtree(label_folder)
        os.makedirs(label_folder)
    try:os.makedirs(image_folder)
    except OSError: 
        shutil.rmtree(image_folder)
        os.makedirs(image_folder)
    return label_folder, image_folder   


def modify_file_name(df,label_folder,image_folder):
    df['group']=[item.rsplit('_')[-3] for item in df['id']]
    df['images'] = [cv2.imread(item) for item in df['file_image']]
    for i in range(len(df['file_label'])):
        modify_labels(f_change, pad,df.iloc[i], label_folder)
        save_resized_images(f_change, pad, df.iloc[i],  image_folder)

def save_padded_data(f_change, pad,df,label_folder,image_folder):
    df['group']=[item.rsplit('_')[-3] for item in df['id']]
    df['images'] = [cv2.imread(item) for item in df['file_image']]
    for i in range(len(df['file_label'])):
        if get_roi_height(f_change,pad, df.iloc[i]['file_label']):
            rename_modify_labels(f_change, pad, df.iloc[i], label_folder)
            save_resized_images(f_change, pad, df.iloc[i],  image_folder)

def get_roi_height(f_change, pad, path):
    """
    Check roi size from daimler labels
    """
    json_data =json.load(open(path))
    _ = copy.deepcopy(json_data)
    for item in _['children']:
        item['maxcol'] = np.round((item['maxcol'])*f_change)+pad
        item['mincol'] = np.round((item['mincol'])*f_change)+pad
        item['maxrow'] = np.round((item['maxrow'])*f_change)
        item['minrow'] = np.round((item['minrow'])*f_change)
    return (item['maxrow']-item['minrow']>=50)

def remove_empty_groundtruth(f_change, pad, path):
    """
    Remove training image if no labels are provided
    """
    json_data =json.load(open(path))
    _ = copy.deepcopy(json_data)
    return len(_['children'])!=0

def rename_modify_labels(f_change, pad, row, label_folder):
    json_data =json.load(open(row['file_label']))
    name = row['file_label'].rsplit('/')[-1] 
    _ = copy.deepcopy(json_data)
    for item in _['children']:
        item['maxcol'] = np.round((item['maxcol'])*f_change)+pad
        item['mincol'] = np.round((item['mincol'])*f_change)+pad
        item['maxrow'] = np.round((item['maxrow'])*f_change)
        item['minrow'] = np.round((item['minrow'])*f_change)

    if _['children']!=[]:
        objects = [get_data_row(child) for child in _['children'] if child['identity']=='cyclist']
        # if len(objects)!=0:
        labelname=str(row['new_ID'])+'.txt'
        labelFile = os.path.join(label_folder, labelname)
        with open(labelFile, 'w') as f:
            for line in objects:
                f.write(str(line) + "\n")

def get_ratio(example):
    """
    Get conversion ratio and padding for transforming Daimler images to KITTI ratio
    """
    KITTI_HEIGHT = 384
    KITTI_WIDTH = 1248
    ratio_Kitti = float(KITTI_HEIGHT)/float(KITTI_WIDTH)
    img = cv2.imread(example)
    height, width, channels = img.shape
    f_change =round(float(KITTI_HEIGHT)/float(height),3)
    pad = np.ceil((KITTI_WIDTH - f_change*width)/2)
    return f_change, pad

def save_data(_df, type, _n, _image_folder, _label_folder, f_change, pad):
    """
    call subfunctions for saving transformed labels and images
    """
    label_folder_type = os.path.join(_label_folder, type)
    image_folder_type = os.path.join(_image_folder, type)
    print(image_folder_type)
    try:os.makedirs(label_folder_type)
    except OSError: 
        shutil.rmtree(label_folder_type)
        os.makedirs(label_folder_type)
    try:os.makedirs(image_folder_type)
    except OSError: 
        shutil.rmtree(image_folder_type)
        os.makedirs(image_folder_type)
    indices = range(0,len(_df))
    _df['new_ID']=["%06d" % (i,) for i in indices]

    _df_size = (len(_df))
    for k in range(_df_size/_n):
        print(str(k*_n) + " " + str((k+1)*_n-1))
        _df_s = _df.iloc[k*_n:(k+1)*_n-1]
        save_padded_data(f_change, pad, _df_s, label_folder_type, image_folder_type)
    _df_s = _df.iloc[(k+1)*_n-1:]
    save_padded_data(f_change, pad, _df_s, label_folder_type, image_folder_type)

def main(**kwargs):
    # processing data in chunks, memory reasons
    n = 100
    # read in locations of original daimler data and desired output location
    label_dir = kwargs.pop('labels_dir')
    image_dir = kwargs.pop('images_dir')
    outdir = kwargs.pop('outdir')
    # create folder structure for processed data
    label_folder, image_folder = create_dirs(outdir)
    # read in image and label paths to dataframe 
    print("reading images")
    images = get_df(image_dir + '*png')
    print("reading labels")
    labels = get_df(label_dir,+ '*json')
    print("mergins dataframe")
    # match images and labels in merged dataframe
    df = pd.merge(labels, images, on=['id'],suffixes=('_label', '_image'))
    # get necessary ratio change and padding size for KITTI dimensions
    f_change, pad = get_ratio((df['file_image'][0]))
    # filters: remove images with no labels
    mask_empty = [df.apply(lambda row: remove_empty_groundtruth(f_change, pad,row['file_label']),axis=1)]
    df = df[mask_empty[0]]
    mask = [df.apply(lambda row: get_roi_height(f_change, pad,row['file_label']),axis=1)]
    df = df[mask[0]]
    #separate into validation and training set
    val_percent = 0.2
    n_valid = (int(len(df)*val_percent))
    df = shuffle(df)
    # process and save daimler data
    save_data(df, 'test', n,image_folder, label_folder, f_change, pad)

if __name__ == '__main__':
    args_namespace = parse_args()
    kwargs = vars(args_namespace)
