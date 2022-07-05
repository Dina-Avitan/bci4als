import copy
import os
import pickle
import sys

from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
import numpy as np


def offline_experiment(pygame_gui_folder_path=0,pygame_gui_keys=0,advanced_gui ={}):

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2

    data_type = SYNTHETIC_BOARD
    if advanced_gui:
        if advanced_gui['use_synthetic']:
            data_type = SYNTHETIC_BOARD
        else:
            data_type = CYTON_DAISY
    data_type = CYTON_DAISY
    gain = {"1": 0, "2":  1, "4": 2, "6": 3, "8": 4, "12": 5, "24": 6}
    configurations = ''.join([''.join(f"x{str(i + 1)}0{gain['6']}0110X") for i in range(8)]
                             +
                             [''.join(f"x{i}0{gain['6']}0110X") for i in ['Q', 'W', 'E']] + [
                ''.join(f"x{i}131000X") for i in ['R', 'T', 'Y', 'U', 'I']])

    eeg = EEG(board_id=data_type, config_json_converted=configurations)
    if advanced_gui:
        exp = OfflineExperiment(eeg=eeg, num_trials=advanced_gui['num_trials'], trial_length=advanced_gui['trial_length'], pygame_gui_folder_path=pygame_gui_folder_path,
                                pygame_gui_keys=pygame_gui_keys, full_screen=True, audio=False,
                                keys=advanced_gui['classes_keys'], baseline_length=1)
    else:
        exp = OfflineExperiment(eeg=eeg, num_trials=27, trial_length=5, pygame_gui_folder_path=pygame_gui_folder_path,
                                pygame_gui_keys=pygame_gui_keys, full_screen=True,
                                audio=False, keys=(0,1,2), baseline_length=1)
    trials, labels = exp.run()
    session_directory = exp.session_directory
    unfiltered_model = MLModel(trials=trials, labels=labels, channel_removed=[], reference_to_baseline=1)
    unfiltered_model.epochs_extractor(copy.deepcopy(eeg))
    pickle.dump(unfiltered_model, open(os.path.join(session_directory, 'model.pickle'), 'wb'))
    print('Finish!!')
    sys.exit(0)


if __name__ == '__main__':
    offline_experiment()

