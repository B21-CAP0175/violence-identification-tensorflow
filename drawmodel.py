
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from keras_visualizer import visualizer

SAVED_MODEL_PATH = "training_mix_forthesecondtime.h5"

loaded=tf.keras.models.load_model(SAVED_MODEL_PATH)

dot_img_file = 'model_vis.png'
tf.keras.utils.plot_model(loaded, to_file=dot_img_file, show_shapes=True)