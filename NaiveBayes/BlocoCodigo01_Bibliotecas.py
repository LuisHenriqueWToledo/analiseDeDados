# Importe bibliotecas e modulos relevantes.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
