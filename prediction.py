# -*- coding: utf-8 -*-
"""prediction.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YNIO6AqkzVDwxPfyrgR47jo9VKQlBU_i
"""

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
