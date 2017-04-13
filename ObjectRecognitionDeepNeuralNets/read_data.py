import os
import glob
from SimpleCV import *
from numpy import array
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import time
print __doc__

def read_data(start, stop, category, size=64, g=True):
	#Settings
	#my_images_path = "data/003.backpack/" #put your image path here if you want to override current directory
	my_images_path = "data/"+category+"/" #put your image path here if you want to override current directory
	extension = "*.jpg"

	#Program
	if not my_images_path:
	        path = os.getcwd() #get the current directory
	else:
	        path = my_images_path

	imgs = list() #load up an image list
	directory = os.path.join(path, extension)
	files = glob.glob(directory)
	out = []

	for file in files[start:stop+1]:
	        new_img = Image.open(file)
	        new_img.load()

	        if g:
	        	grey = new_img.greyscale()
	        else:
	        	grey = new_img	
	        
	        grey = grey.resize(size, size)
	        grey = np.resize(grey,(size*size))
	        out.append(data)
	        #grey.show()
	        #time.sleep(1) #wait for 1 second

	return out        


# Use case
#out = read_data(0, 10, "078.fried-egg", 64, False)
#out[0].show()
#time.sleep(1)

#print out[0][0]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

image = mpimg.imread('data/003.backpack/003_0001.jpg', 0)
image = rgb2gray(image)
print image.shape

