import os
import pickle
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
import numpy as np


def offline_experiment():

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=SYNTHETIC_BOARD)
    exp = OfflineExperiment(eeg=eeg, num_trials=10, trial_length=3,
                            full_screen=True, audio=False)
    channel_removed = []
    trials, labels = exp.run()
    session_directory = exp.session_directory

    # do Laplacian filter
    trials, channel_removed = eeg.laplacian(trials)
    pickle.dump(trials, open(os.path.join(session_directory, 'trials_after_laplacian.pickle'), 'wb'))
    # channel_removed = channel_removed.append() ##TODO: NOAM WILL MAKE OUTLIER CHANNELS DISAPPEAR

    # Get model ready for classification
    model = MLModel(trials=trials, labels=labels, channel_removed=channel_removed)
    model_test = MLModel(trials=trials, labels=labels, channel_removed=channel_removed)

    # save epochs
    model.epochs_extractor(eeg)
    pickle.dump(model, open(os.path.join(session_directory, 'raw_model.pickle'), 'wb'))

    # train model and classify
    model.offline_training(model_type='simple_svm')
    # Dump the MLModel
    pickle.dump(model, open(os.path.join(session_directory, 'trained_model.pickle'), 'wb'))

    # cross-validation
    scores = model_test.cross_val()
    (print(f"Prediction rate is: {np.mean(scores)*100}%"))


if __name__ == '__main__':

    offline_experiment()

