
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Creates IDL files for daimler dataset, that are used during training KittiBox
Used for transforming Daimler dataset to KITTI format.
Usage:
python annotator.py --labels_dir /path/to/daimler/labels/ --images_dir 
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


def saveIDL(filename, annotations):
        [name, ext] = os.path.splitext(filename)

        if(ext == ".idl"):
                file = open(filename,'w')

        if(ext == ".gz"):
                file = gzip.GzipFile(filename, 'w')

        if(ext == ".bz2"):
                file = bz2.BZ2File(filename, 'w')

        i=0
        for annotation in annotations:
                annotation.writeIDL(file)
                if (i+1<len(annotations)):
                        file.write(";\n")
                else:
                        file.write(".\n")
                i+=1

        file.close()

def get_daimler_files(root_dir, file_ext):
        """ takes filenames and ids to dataframe"""
        lFiles = []
        print(root_dir)
        for file in glob.glob(root_dir+"*"+file_ext):
                print(file)
                lFiles.append(os.path.join(root_dir, file))
        if(len(lFiles)==0):raise Exception('no data found!') 
        idFiles=[fname.rsplit('/')[-1].rsplit('.')[-2] for fname in lFiles]
        dfFiles = pd.DataFrame(
        {     'id': idFiles,'file': lFiles })

        dfFiles = dfFiles.sort_index(ascending=True)
        return dfFiles

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
                '--images_dir',required=True,help = '')
        parser.add_argument(
                '--labels_dir',required=True,help = '')
        parser.add_argument(
                '--outdir',required=True,help = '')   
        args = parser.parse_args()
        return args

def main(**kwargs):
        # ONLY READ TRAIN - shuffle and val and train
        labels_dir = kwargs.pop('labels_dir')
        images_dir = kwargs.pop('images_dir')
        print("reading images")
        images = get_daimler_files(images_dir,'png')
        print("reading labels")
        labels = get_daimler_files(labels_dir,'txt')
        print("mergins dataframe")
        df = pd.merge(labels, images, on=['id'],suffixes=('_label', '_image'))
        
        df = shuffle(df)
        val_percent = 0.2
        n_valid = (int(len(df)*val_percent))
        df_valid = df.iloc[:n_valid]
        df_train = df.iloc[n_valid:]


        # function of writin - _df
        annotations_arr = []
        for row in df_valid.iterrows():
                with open(row[1]['file_label']) as f:
                        records = f.readlines()
                        annotation_data = Annotation()
                        # annotation_data.imageName = ""
                        annotation_data.imagePath ='/'.join(row[1]['file_image'].rsplit('/')[-4:])
                        print annotation_data.imagePath
                        for record in records:
                                rect = AnnoRect()
                                txt_data= record.rsplit(' ')
                                if txt_data[0]=='Car':
                                        xmin,ymin,xmax,ymax = np.array(txt_data[4:8]).astype(float)
                                rect.x1 = xmin
                                rect.y1 = ymin
                                rect.x2 = xmax
                                rect.y2 = ymax
                                annotation_data.rects.append(rect)
                        annotations_arr.append(annotation_data)
        saveIDL('/home/anna/valid.idl',annotations_arr)

        l = ['/'.join(str(x).rsplit('/')[-4:]) + " " +'/'.join(str(y).rsplit('/')[-4:]) for x, y in zip(df_valid.file_image, df_valid.file_label)]
        np.savetxt('/home/anna/val.txt', l, delimiter=" ", fmt="%s")

        # function of writin - _df
        annotations_arr = []
        for row in df_train.iterrows():
                with open(row[1]['file_label']) as f:
                        records = f.readlines()
                        annotation_data = Annotation()
                        # annotation_data.imageName = ""
                        annotation_data.imagePath ='/'.join(row[1]['file_image'].rsplit('/')[-4:])
                        for record in records:
                                rect = AnnoRect()
                                txt_data= record.rsplit(' ')
                                if txt_data[0]=='Car':
                                        xmin,ymin,xmax,ymax = np.array(txt_data[4:8]).astype(float)
                                rect.x1 = xmin
                                rect.y1 = ymin
                                rect.x2 = xmax
                                rect.y2 = ymax
                                annotation_data.rects.append(rect)
                        annotations_arr.append(annotation_data)
        saveIDL('/home/anna/train.idl',annotations_arr)

        l = ['/'.join(str(x).rsplit('/')[-4:]) + " " +'/'.join(str(y).rsplit('/')[-4:]) for x, y in zip(df_train.file_image, df_train.file_label)]
        np.savetxt('/home/anna/train.txt', l, delimiter=" ", fmt="%s")

if __name__ == '__main__':              
        args_namespace = parse_args()
        kwargs = vars(args_namespace)
        main(**kwargs)

