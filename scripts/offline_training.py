import copy
import os
import pickle
import sys

from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
import numpy as np


def offline_experiment(gui_folder_path=0,gui_keys=0):

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    gain = {"1": 0, "2":  1, "4": 2, "6": 3, "8": 4, "12": 5, "24": 6}
    configurations = ''.join([''.join(f"x{str(i + 1)}0{gain['6']}0110X") for i in range(8)]
                             +
                             [''.join(f"x{i}0{gain['6']}0110X") for i in ['Q', 'W', 'E']] + [
                ''.join(f"x{i}131000X") for i in ['R', 'T', 'Y', 'U', 'I']])

    eeg = EEG(board_id=CYTON_DAISY, config_json_converted=configurations)
    exp = OfflineExperiment(eeg=eeg, num_trials=21, trial_length=5,gui_folder_path=gui_folder_path,gui_keys=gui_keys, full_screen=True, audio=False,keys=(0,1,2))
    trials, labels = exp.run()
    session_directory = exp.session_directory
    unfiltered_model = MLModel(trials=trials, labels=labels, channel_removed=[])
    unfiltered_model.epochs_extractor(copy.deepcopy(eeg))
    pickle.dump(unfiltered_model, open(os.path.join(session_directory, 'pre_laplacian.pickle'), 'wb'))
    print('Finish!!')
    sys.exit(0)
if __name__ == '__main__':

    offline_experiment()

