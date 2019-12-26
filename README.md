# dogs_vs_cats_classification

# Dataset: 

Here kaggle dogs_vs_cats dataset is used which contain 25000 images for dogs and cats. More data is found <a href="https://www.kaggle.com/c/dogs-vs-cats/data">here</a>

# Building HDF5 dataset

I have used `build_dogs_vs_cats.py` file for building hdf5 dataset. To run this file insert following command

`python build_dogs_vs_cats.py`

# Training AlexNet: 

I have used `train_alexnet.py` file for training AlexNet architecture on training images with 75 epochs. To run this file insert following command

`python train_alexnet.py`

# Evaluating AlexNet:

I have used `crop_accuracy.py` file for evaluating AlexNet model. To run this file insert following command

`python crop_accuracy.py`

# Training ResNet:

I have used `extract_features.py` file to extract resnet weights trained on ImageNet dataset and train on training images. To run this file insert following command

`python extract_features.py --dataset ../datasets/kaggle_dogs_vs_cats/train --output ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5`

# Train Logistic Regression classifier:

I have used `train_model.py` file to train logistic regression on extracted features. Using this file i get 96-98% accuracy and saving model for later use.
To run this file insert following command

`python train_model.py --db ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.pickle`
