import os
from PIL import Image
from array import *
from random import shuffle
import glob 
from sklearn.model_selection import train_test_split 
import cv2

width = 64
height = 64

train_data = '/home/super/janjua/inv-layer-classification/URDU/train/'
test_data = '/home/super/janjua/inv-layer-classification/URDU/val/'

x = []
y = []

for i in glob.glob(train_data+'*/*'):
	img = cv2.imread(i)
	label = i.split('/')[-2]
	x.append(img)
	y.append(label)

for i in glob.glob(test_data+'*/*'):
	img = cv2.imread(i)
	label = i.split('/')[-2]
	x.append(img)
	y.append(label)

assert len(x) == len(y)

def get_split():
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
	return (x_train, x_test),(y_train, y_test)