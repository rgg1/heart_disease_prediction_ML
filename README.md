# Heart Disease Prediction

This repository contains a machine learning model to predict heart disease using a dataset of various health metrics. The model is implemented in PyTorch and achieves an accuracy of over 90% on the training dataset.

## Overview

The notebook includes:
- Data loading and manipulation using pandas and NumPy.
- Preprocessing, including one-hot encoding of categorical variables.
- Splitting the data into training and testing sets.
- Defining and training a neural network using PyTorch, with linear and ReLU activation functions.
- Evaluating the model on the testing set.

## Dataset

The dataset `heart.csv.xls` consists of different attributes such as age, sex, chest pain type, resting blood pressure, cholesterol level, etc. The target variable, 'HeartDisease', indicates the presence or absence of heart disease.
Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Model Architecture

The model consists of four linear layers, each followed by a ReLU activation function, except for the output layer. The architecture is as follows:

- Input Layer: Dimension based on input features.
- Hidden Layer 1: 40 neurons.
- Hidden Layer 2: 80 neurons.
- Hidden Layer 3: 60 neurons.
- Output Layer: 1 neuron (Binary classification).

## Requirements

The following libraries are required to run the notebook:

- torchmetrics
- torchinfo
- numpy
- pandas
- PyTorch
- scikit-learn
- tqdm

To install the specific libraries, use:

! pip install torchmetrics -q
! pip install torchinfo -q

If running on Google Colab, these should be the only libraries you need to install.

## Running the Code

You can run the notebook in a local environment that supports Jupyter Notebooks or directly in Google Colab. Ensure that the dataset is placed in the same directory as the notebook or modify the code to point to the correct location.

## Results

The model achieves over 90% accuracy on the training dataset, with slightly lower accuracy on the testing dataset but consistently over 80%. The notebook includes detailed output logs for the loss and accuracy at regular intervals during training.

## Future Work

Improvements could potentially be made by tuning hyperparameters, adjusting the model architecture, or using different preprocessing techniques.
