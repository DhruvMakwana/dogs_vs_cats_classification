#importing libraries
from config import dogs_vs_cats_config as config
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from preprocessing import meanpreprocessor
from preprocessing import croppreprocessor
from hdf5 import hdf5datasetgenerator
from utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

#load json
means = json.loads(open(config.DATASET_MEAN).read())

#preprocessor
sp = simplepreprocessor.SimplePreprocessor(227, 227)
mp = meanpreprocessor.MeanPreprocessor(means["R"], means["G"], means["B"])
cp = croppreprocessor.CropPreprocessor(227, 227)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

#loading
print("[Info] loading model...")
model = load_model(config.MODEL_PATH)

#testing
print("[INFO] predicting on test data (no crops)...")
testGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TEST_HDF5, 64,
	preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(),\
	steps=testGen.numImages // 64, max_queue_size=64 * 2)

#rank-1 and rank-5 accuracies
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()

testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
	preprocessors=[mp], classes=2)
predictions = []

#initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64,
	widgets=widgets).start()

#loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
	#loop over each of the individual images
	for image in images:
		#apply the crop preprocessor to the image to generate 10
		#separate crops, then convert them from images to arrays
		crops = cp.preprocess(image)
		crops = np.array([iap.preprocess(c) for c in crops],
			dtype="float32")

		#make predictions on the crops and then average them
		#together to obtain the final prediction
		pred = model.predict(crops)
		predictions.append(pred.mean(axis=0))

	#update the progress bar
	pbar.update(i)
#compute the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()