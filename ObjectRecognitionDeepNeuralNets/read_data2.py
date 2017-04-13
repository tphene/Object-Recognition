import os
import glob
import numpy as np
import matplotlib.image as mpimg

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

	for file in files[start-1:stop]:
		image = mpimg.imread(file)
		image = rgb2gray(image)
		image.resize((size*size))
		out.append(image)
	
	out = np.asarray(out)	
	return out


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# Use case
#out = read_data(1, 151, "003.backpack", 28, False)
#print out.shape
#out[0].show()
#time.sleep(1)
