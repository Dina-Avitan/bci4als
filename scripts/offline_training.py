import os
import pickle
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment


def offline_experiment():

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=SYNTHETIC_BOARD)
    exp = OfflineExperiment(eeg=eeg, num_trials=5, trial_length=3,
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
    pickle.dump(model, open(os.path.join(session_directory, 'raw_data.pickle'), 'wb'))

    # save epochs
    epochs = model.epochs_extractor(eeg)
    pickle.dump(epochs, open(os.path.join(session_directory, 'epochs.pickle'), 'wb'))

    # train model and classify
    model.offline_training(eeg=eeg, model_type='csp_lda')
    features = model_test.offline_training(eeg=eeg, model_type='simple_svm')
    print(features.shape)
    pickle.dump(features, open(os.path.join(session_directory, 'features.pickle'), 'wb'))

    # Dump the MLModel
    pickle.dump(model, open(os.path.join(session_directory, 'model.pickle'), 'wb'))


if __name__ == '__main__':

    offline_experiment()

