# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:30:36 2020

@author: leno
"""


import numpy as np
import pandas as pd
covid_data=pd.read_csv('metadata.csv')
covid_data.head()

covid_data.dropna(axis=1,inplace=True)

covid_data.groupby('finding').count()

import pandas as pd
import shutil
import os
# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
coronavirus = "COVID-19" # Virus to look for
x_ray = "PA" # View of X-Ray
metadata = "metadata.csv" # Metadata.csv Directory
imageDir = "images" # Directory of images
outputDir = 'Data//Covid' # Output directory to store selected images
metadata_csv = pd.read_csv(metadata)
# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
    if row["finding"] != coronavirus or row["view"] != x_ray:
        continue
    filename = row['filename'].split(os.path.sep)[-1]
    filePath = os.path.sep.join([imageDir, filename])
    shutil.copy2(filePath, outputDir)
print('Done')


import matplotlib.pyplot as plt
import argparse
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

INIT_LR = 1e-3
EPOCHS = 10
BS = 8
dataset = "C:\\Users\\leno\\Desktop\\Machine Learning A-Z Template Folder\\Hach-Off\\Data"

args={}
args["dataset"]=dataset

import numpy as np
import cv2
iPaths = list(paths.list_images(args["dataset"]))  #image paths
data = []
labels = []
for iPath in iPaths:
    label = iPath.split(os.path.sep)[-2]   #split the image paths
    image = cv2.imread(iPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert images into RGB Channel
    image = cv2.resize(image, (224, 224))  #Resizing the images
    data.append(image)
    labels.append(label)
data = np.array(data) / 255.0
labels = np.array(labels)
