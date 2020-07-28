import numpy # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plot
from pathlib import Path
import glob
import random

dataset_path = '../input/open-images-object-detection-rvc-2020/test'

#image_dataset = os.listdir(dataset_path)[:5]
#print(image_dataset)

def load_training_data(dataset_path):
    training_dataset = []
    for image in glob.glob(dataset_path + '/*.*'):
        try:
            #read the images one by one
            image_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (50, 50))
            training_dataset.append(image_array)
            #print(image_array)

            ####show the read data/images####
            #plot.imshow(image_array, cmap='gray')
            #plot.show()
        except Exception as e:
            print(e)

    return training_dataset

def further_preprocessing(training_dataset):
    shuffled_training_dataset = random.shuffle(training_dataset)
    
    return shuffled_training_dataset
    

training_dataset = load_training_data(dataset_path)
shuffled_training_dataset = further_preprocessing(training_dataset)
    
####prints calls
#print('Count Loading ...')
#print(len(training_dataset))
#print(len(shuffled_training_dataset))

