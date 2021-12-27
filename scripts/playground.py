# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
from tkinter import filedialog, Tk
import pandas as pd
import os
import pickle
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
import numpy as np


def playground():
    # load eeg data
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    raw_model = pd.read_pickle(fr'{file_path}')
    raw_model.offline_training(model_type='simple_svm')
    scores = raw_model.cross_val()
    (print(f"Prediction rate is: {np.mean(scores)*100}%"))


if __name__ == '__main__':
    playground()
