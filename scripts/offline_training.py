import os
import pickle
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
import numpy as np


def offline_experiment():

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    gain = {"1": 0, "2":  1, "4": 2, "6": 3, "8": 4, "12": 5, "24": 6}
    configurations = ''.join([''.join(f"x{str(i + 1)}0{gain['6']}0110X") for i in range(8)] +
                             [''.join(f"x{i}0{gain['6']}0110X") for i in ['Q', 'W', 'E']] + [
                ''.join(f"x{i}131000X") for i in ['R', 'T', 'Y', 'U', 'I']])

    eeg = EEG(board_id=CYTON_DAISY, config_json_converted=configurations)
    exp = OfflineExperiment(eeg=eeg, num_trials=60, trial_length=5, full_screen=True, audio=False)
    channel_removed = []
    trials, labels = exp.run()
    session_directory = exp.session_directory
    # remove outliers
    for i in range(len(trials)):
        std_col = trials[i].std(axis=0)
        channel_removed += std_col[std_col == 0].index.tolist()  # add outliers (bad electrodes) to remove

    # get and save raw unfiltered data
    unfiltered_model = MLModel(trials=trials, labels=labels, channel_removed=[])
    unfiltered_model.epochs_extractor(eeg)
    pickle.dump(unfiltered_model, open(os.path.join(session_directory, 'unfiltered_model.pickle'), 'wb'))

    # do Laplacian filter
    to_remove = []
    # trials, to_remove = eeg.laplacian(trials)
    # Delete repetitive elements in the list
    #channel_removed = list(set(channel_removed + to_removed))
    channel_removed = to_remove

    # Get model ready for classification
    model = MLModel(trials=trials, labels=labels, channel_removed=channel_removed)

    # save epochs
    model.epochs_extractor(eeg)
    pickle.dump(model, open(os.path.join(session_directory, 'raw_model.pickle'), 'wb'))

    # train model and classify
    model.offline_training(model_type='simple_svm')
    # Dump the MLModel
    pickle.dump(model, open(os.path.join(session_directory, 'trained_model.pickle'), 'wb'))

    print('Finish!!')

if __name__ == '__main__':

    offline_experiment()

