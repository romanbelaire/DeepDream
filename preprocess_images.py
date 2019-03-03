#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:07:35 2019

@author: roman
"""

import cv2
import os, glob, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir", help="directory containing images to process")
parser.add_argument("l", help = "desired length of resulting image",type=int)
parser.add_argument("w", help = "desired width of resulting image",type=int)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args() #arguments from user running the program

path = args.dir
targ_x = args.w
targ_y = args.l

numfiles = next(os.walk(path))[2] #dir is your directory path as string
numfiles = len(numfiles)

for i, img_path in enumerate(glob.iglob(path + "/*")):
    try:
        sys.stdout.write("\r" + str(i + 1) + "/" + str(numfiles))
        img = cv2.imread(img_path) #read image from path
        new_img = cv2.resize(img, (targ_x, targ_y)) #resize the image to specified dim
        #if(args.verbose):

        cv2.imwrite(img_path, new_img) #save to file

    except Exception as e:
        print(img_path + " not a resizeable image... removing")
        os.remove(img_path)
        print("Removed!")
print("Done!")