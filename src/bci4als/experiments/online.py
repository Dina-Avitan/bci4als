import copy
import json
import os
import pickle
import random
import sys
import threading
import time
from typing import Dict, Union
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import playsound
from bci4als.eeg import EEG
from .experiment import Experiment
from bci4als.experiments.feedback import Feedback
from bci4als.ml_model import MLModel
from matplotlib.animation import FuncAnimation
from mne_features.feature_extraction import extract_features
from nptyping import NDArray
from psychopy import visual, core
from sklearn.preprocessing import StandardScaler
import pandas as pd


class OnlineExperiment(Experiment):
    """
    Class for running an online MI experiment.

    Attributes:
    ----------

        num_trials (int):
            Amount of trials in the experiment.

        buffer_time (float):
            Time in seconds for collecting EEG data before model's prediction.

        threshold (int):
            The amount the times the model need to be correct (predict = stim) before moving to the next stim.

    """

    def __init__(self, eeg: EEG, model: MLModel, num_trials: int,
                 buffer_time: float, threshold: int, co_learning: bool,skip_after: Union[bool, int] = False,
                 debug=False):

        super().__init__(eeg, num_trials)
        # experiment params
        self.experiment_type = "Online"
        self.threshold: int = threshold
        self.buffer_time: float = buffer_time
        self.model = model
        self.skip_after = skip_after
        # self.debug = self.model.debug
        self.debug = debug
        self.win = None
        self.co_learning: bool = co_learning

        # audio
        self.audio_success_path = os.path.join(r'../src/bci4als/experiments', 'audio', f'success.mp3')  # hope its generic

        # Model configs
        self.labels_enum: Dict[str, int] = {'right': 0, 'left': 1, 'idle': 2}  # , 'tongue': 3, 'legs': 4}
        self.label_dict: Dict[int, str] = dict([(value, key) for key, value in self.labels_enum.items()])
        self.num_labels: int = len(self.labels_enum)
        self.batch_stack = [[],[],[]]  # number of classes is number of empty lists


        # Hold list of lists of target-prediction pairs per trial
        # Example: [ [(0, 2), (0,3), (0,0), (0,0), (0,0) ] , [ ...] , ... ,[] ]
        self.results = []

    def _learning_model(self, feedback: Feedback, stim: int):

        """
        The method for learning the model from the current stim.

        A separate thread runs this method. The method responsible for the following steps:
            1. Collecting the EEG data from the board (according to the buffer time attribute).
            2. Predicting the stim using the current model and collected EEG data.
            3. Updating the feedback object according to the model's prediction.
            4. Updating the model according to the data and stim.

        :param feedback: feedback visualization for the subject
        :param stim: current stim
        :return:
        """

        timer = core.Clock()
        target_predictions = []
        num_tries = 0
        while not feedback.stop:
            # increase num_tries by 1
            print(f"num tries {num_tries}")

            # Sleep until the buffer full
            time.sleep(max(0, self.buffer_time - timer.getTime()))

            # Get data and channes from EEG
            data = copy.deepcopy(self.eeg.get_channels_data())

            # get data into epochs and filter it
            ch_names = self.eeg.get_board_names()
            # [ch_names.remove(bad_ch) for bad_ch in self.model.channel_removed if bad_ch in ch_names]
            ch_types = ['eeg'] * len(ch_names)
            sfreq: int = self.eeg.sfreq
            info = mne.create_info(ch_names, sfreq, ch_types)
            n_samples: int = min([t.shape[0] for t in data])  # get the minimum length of each elec
            epochs_array: np.ndarray = (np.stack([t[:self.model.epochs.get_data()[0].shape[1]] for t in data]))[np.newaxis]  # make the elecs same size
            # Get epochs object for prediction and another one for adding it to the data
            epochs = mne.EpochsArray(epochs_array, info)
            epochs.filter(1., 40., fir_design='firwin', skip_by_annotation='edge', verbose=False)
            # Make two epoch objects: one for prediction(will go ICA and laplace) and one for adding to data
            epochs_pred = copy.deepcopy(epochs)
            # Apply ICA
            epochs_pred = self.model.ica.apply(epochs_pred)
            # LaPlacian filter
            data, _ = self.eeg.laplacian(epochs_pred.get_data())
            # Predict the class
            if self.debug:
                # in debug mode, be correct 2/3 of the time and incorrect 1/3 of the time.
                prediction = stim if np.random.rand() <= 2 / 3 else (stim + 1) % len(self.labels_enum)
            else:
                # in normal mode, use the loaded model to make a prediction
                # squeeze is a plaster. you can later remove all the newaxis fom online predict
                prediction, test_features = self.model.online_predict(np.squeeze(epochs_pred.get_data()), eeg=self.eeg)
                prediction = int(prediction)

            # play sound if successful
            # todo: make this available to object params
            self.play_sound = True
            if self.play_sound:
                if prediction == stim:
                    playsound.playsound(self.audio_success_path)

            if self.co_learning:# and prediction == stim:  # maybe prediction doesnt have to be == stim
                self.batch_stack[stim].append(np.squeeze(epochs.get_data()))
                if all(self.batch_stack):
                    print('co-adaptive working')
                    data_batched = [self.batch_stack[0].pop(0), self.batch_stack[1].pop(0), self.batch_stack[2].pop(0)]
                    self.model.partial_fit(data_batched, [0,1,2], epochs, sfreq)
                    pickle.dump(self.model, open(os.path.join(self.session_directory, 'trained_model.pickle'), 'wb'))
            target_predictions.append((int(stim), int(prediction)))

            # Reset the clock for the next buffer
            timer.reset()

            if stim == prediction:
                num_tries = 0  # if successful, reset num_tries to 0
                print(num_tries)
            else:
                num_tries += 1

            # Update the feedback according the prediction
            feedback.update(prediction, skip=(num_tries >= self.skip_after))
            # feedback.update(stim)  # For debugging purposes

            # Debug
            print(f'Predict: {self.label_dict[prediction]}; '
                  f'True: {self.label_dict[stim]}')
        accuracy = sum([1 if p[1] == p[0] else 0 for p in target_predictions]) / len(target_predictions)
        print(f'Accuracy of last target: {accuracy}')
        self.results.append(target_predictions)

        # Save Results
        json.dump(self.results, open(os.path.join(self.session_directory, 'results.json'), "w"))

    def run(self, use_eeg: bool = True, full_screen: bool = False):

        # Init the current experiment folder
        self.subject_directory = self._ask_subject_directory()
        self.session_directory = self.create_session_folder(self.subject_directory)

        # Create experiment's metadata
        self.write_metadata()

        # Init experiments configurations
        self.win = visual.Window(monitor='testMonitor', fullscr=full_screen)

        # turn on EEG streaming
        if use_eeg:
            self.eeg.on()

        # For each stim in the trials list
        for stim in self.labels:

            # Init feedback instance
            feedback = Feedback(self.win, stim, self.buffer_time, self.threshold)

            # Use different thread for online learning of the model
            threading.Thread(target=self._learning_model,
                             args=(feedback, stim), daemon=True).start()

            # Maintain visual feedback on screen
            timer = core.Clock()

            while not feedback.stop:

                feedback.display(current_time=timer.getTime())

                # Reset the timer according the buffer time attribute
                if timer.getTime() > self.buffer_time:
                    timer.reset()

                # Halt if escape was pressed
                if 'escape' == self.get_keypress():
                    sys.exit(-1)

            # Waiting for key-press between trials
            self._wait_between_trials(feedback, self.eeg, use_eeg)

        # turn off EEG streaming
        if use_eeg:
            self.eeg.off()
