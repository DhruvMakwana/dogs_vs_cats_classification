#importing libraries
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from hdf5 import hdf5datasetwriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
#--dataset is the path to the input dataset of Dogs vs. Cats images
ap.add_argument("-o", "--output", required=True,
	help="path to output HDF5 file")
#--output is the path to the output HDF5 file containing the features 
#extracted via ResNet
ap.add_argument("-b", "--batch-size", type=int, default=16,
	help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
	help="size of feature extraction buffer")
args = vars(ap.parse_args())

#batch_size
bs = args["batch_size"]

#grab images and shuffle them easy training and testing splits
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

#encode labels
labels = [p.split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

#load ResNet50 network
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)

dataset = hdf5datasetwriter.HDF5DatasetWriter((len(imagePaths), 100352),
	args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

#progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
	widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
	batchPaths = imagePaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	for (j, imagePath) in enumerate(batchPaths):
		#load the input image using the Keras helper utility
		#while ensuring the image is resized to 224x224 pixels
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)

		#preprocess the image by (1) expanding the dimensions and
		#(2) subtracting the mean RGB pixel intensity from the
		#ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

		#add the image to the batch
		batchImages.append(image)

	#pass the images through the network and use the outputs as
	#our actual features
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)

	#reshape the features so that each image is represented by
	#a flattened feature vector of the ‘MaxPooling2D‘ outputs
	features = features.reshape((features.shape[0], 100352))

	#add the features and labels to our HDF5 dataset
	dataset.add(features, batchLabels)
	pbar.update(i)

#close the dataset
dataset.close()
pbar.finish()