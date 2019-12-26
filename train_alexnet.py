#imorting libraries
#set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from config import dogs_vs_cats_config as config
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from preprocessing import patchpreprocessor
from preprocessing import meanpreprocessor
from callbacks import trainingmonitor
from hdf5 import hdf5datasetgenerator
from nn.conv import alexnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

#data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

#load RGB
means = json.loads(open(config.DATASET_MEAN).read())

#preprocessors
sp = simplepreprocessor.SimplePreprocessor(227, 227)
pp = patchpreprocessor.PatchPreprocessor(227, 227)
mp = meanpreprocessor.MeanPreprocessor(means["R"], means["G"], means["B"])
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

#training and validation dataset generator
trainGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TRAIN_HDF5, 64,
	aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.VAL_HDF5, 64,
	preprocessors=[sp, mp, iap], classes=2)

#compiling
print("[Info] compiling model...")
opt = Adam(lr=0.001)
model = alexnet.AlexNet.build(width=227, height=227, depth=3, classes=2, 
	reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, 
	metrics=["accuracy"])

#construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [trainingmonitor.TrainingMonitor(path)]

#training
print("[Info] training model...")
model.fit_generator(
	trainGen.generator(),
	epochs=75,
	steps_per_epoch=trainGen.numImages // 64,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // 64,
	max_queue_size=64 * 2,
	callbacks=callbacks, verbose=1)

#save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

#close the HDF5 datasets
trainGen.close()
valGen.close()