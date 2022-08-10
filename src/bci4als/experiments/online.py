import copy
import json
import os
import pickle
import random
import sys
import threading
import time
import tkinter
from tkinter import messagebox
from tkinter.filedialog import askopenfile
from typing import Dict, Union
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import playsound
import seaborn
from sklearn.metrics import confusion_matrix

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
import random


class OnlineExperiment(Experiment):
    """
    Class for running an online MI experiment.

    """

    def __init__(self, eeg: EEG, model: MLModel, num_trials: int,
                 buffer_time: float, threshold: int, co_learning: bool,skip_after: Union[bool, int] = False,
                 debug=False, mode='practice',stim_sound = False,keys=(0,1,2), baseline_length=0):
        """

        Args:
            eeg: The board
            model: A model from a previous recording. On the basis of this model the prediction is made
            num_trials: The number of trials you want. This number should be a multiple of the number of classes in the experiment
            buffer_time:  Time in seconds for collecting EEG data before model's prediction.
            threshold: The amount the times the model need to be correct (predict = stim) before moving to the next stim.
            co_learning: (bool)
            skip_after: The num of the same class repeat until it pass to the next trial
            debug: in debug mode, The prediction will be correct 2/3 of the time and incorrect 1/3 of the time
            mode:  If mode= 'practice': It will skip after skip_after errors. it will skip after threshold successes
                    If mode= 'test': It will not skip. It will run skip_after times whether you succeed or fail the trial
            stim_sound: (bool) If you want to add sound to the experiments
            keys:  the selected parameters: 0-right, 1-left, 2-idel, 3-tongue, 4-hands (if you are not use the GUI)
            baseline_length:
        """

        super().__init__(eeg, num_trials,keys)
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
        curr_path = os.getcwd()
        bci4als_path = os.path.dirname(curr_path)
        success_pathes = os.path.join(bci4als_path,'src','bci4als','experiments','audio', 'success_sounds')
        self.num_of_success_sounds = len(os.listdir(success_pathes))
        self.audio_success_path = [os.path.join(success_pathes, str(i)+'.mp3') for i in range(1,self.num_of_success_sounds+1)]
        self.audio_next_pathes = [os.path.join(r'../src/bci4als/experiments', 'audio', f'next_right.mp3'),
                                  os.path.join(r'../src/bci4als/experiments', 'audio', f'next_left.mp3'),
                                  os.path.join(r'../src/bci4als/experiments', 'audio', f'next_idle.mp3')]
        self.audio_first_stim_pathes = [os.path.join(r'../src/bci4als/experiments', 'audio', f'right.mp3'),
                                        os.path.join(r'../src/bci4als/experiments', 'audio', f'left.mp3'),
                                        os.path.join(r'../src/bci4als/experiments', 'audio', f'idle.mp3')]
        # Model configs
        self.all_labels= {0:'right',1: 'left', 2: 'idle', 3:'tongue', 4:'hands'}
        self.labels_enum = dict([(self.all_labels[i],i)for i in keys ])
        #self.labels_enum: Dict[str, int] = {'right': 0, 'left': 1, 'idle': 2, 'tongue': 3, 'hands': 4}
        self.label_dict: Dict[int, str] = dict([(value, key) for key, value in self.labels_enum.items()])
        self.num_labels: int = len(self.labels_enum)
        self.batch_stack = [[] for _ in range(len(keys))]  # number of classes is number of empty lists
        self.mode = mode
        self.stim_sound = stim_sound

        # Hold list of lists of target-prediction pairs per trial
        # Example: [ [(0, 2), (0,3), (0,0), (0,0), (0,0) ] , [ ...] , ... ,[] ]
        self.results = []
        # for labeling the predictions
        self.stack_order = dict([(keys[i],i) for i in range(len(keys))])
        # checking that baseline length matches the baseline length of loaded model
        if model.reference_to_baseline != baseline_length:
            print(f"Error: The baseline length you entered: {baseline_length} should match the baseline length of the model: {model.reference_to_baseline}")
            sys.exit(-1)
        # Parameter for referencing in respect to a baseline
        self.baseline_length = baseline_length  # in seconds. 0=no baseline.

    @staticmethod
    def ask_model_directory():
        """
        init the current subject directory
        :return: the subject directory
        """

        # get the CurrentStudy recording directory
        if not messagebox.askokcancel(title='bci4als - Online training',
                                      message="Select trained model path"):
            sys.exit(-1)

        # show an "Open" dialog box and return the path to the selected file
        model_path = askopenfile().name
        tkinter._default_root.destroy()
        if not model_path:
            sys.exit(-1)
        return model_path

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
        threshold_skip = 0
        while not feedback.stop:
            # increase num_tries by 1
            print(f"num tries {num_tries}")
            # Sleep until the buffer full
            time.sleep(max(0, self.buffer_time - timer.getTime()))

            # Parameters
            sfreq: int = self.eeg.sfreq
            ch_names = self.eeg.get_board_names()
            ch_types = ['eeg'] * len(ch_names)

            # Get data and channels from EEG
            data = copy.deepcopy(self.eeg.get_channels_data())
            if (num_tries > 0 or threshold_skip > 0) and self.baseline_length:
                # we need to append the baseline we sampled only in the first mini-trial
                print(data.shape)
                print(baseline.shape)
                data = np.concatenate((np.squeeze(baseline), data), axis=1)
                print(data.shape)
            # Prepare data for transformation to mne.Epoch object
            info = mne.create_info(ch_names, sfreq, ch_types)
            # Massage the data to fit the model size
            epochs_array: np.ndarray = (np.stack([t[:self.model.epochs.get_data().shape[2]] for t in data]))[np.newaxis]  # make the elecs same size
            # Get epochs object for prediction and another one for adding it to the data
            epochs = mne.EpochsArray(epochs_array, info)
            # Preprocessing
            epochs.filter(1., 40., fir_design='firwin', skip_by_annotation='edge', verbose=False)
            # Make two epoch objects: one for prediction and one for adding to data Because we dont want
            # filters on the raw data that we will be adding to the model.
            epochs_pred = copy.deepcopy(epochs)
            # Apply ICA
            epochs_pred = self.model.ica.apply(epochs_pred)
            # LaPlacian filter
            data_for_prediction, _ = self.eeg.laplacian(epochs_pred.get_data())
            # Separate the prediction object into data and baseline (only if num_tries is 0).

            # Because we sample the baseline once, we need to keep it throughout every "mini-trial" within each trial
            # If skip_after param is 0, this problem does not exist.
            if num_tries == 0:
                data_for_prediction, baseline = MLModel.baseline_extractor(data=data_for_prediction, fs=sfreq, baseline_length=self.baseline_length)
            else:
                pass

            # Predict the class
            if self.debug:
                # in debug mode, be correct 2/3 of the time and incorrect 1/3 of the time.
                prediction = stim if np.random.rand() <= 2 / 3 else (stim + 1) % len(self.labels_enum)
            else:
                # in normal mode, use the loaded model to make a prediction
                # squeeze is a plaster. you can later remove all the newaxis fom online predict
                prediction = self.model.online_predict(data=data_for_prediction, eeg=self.eeg,
                                                                      baseline=baseline)
                prediction = int(prediction)

            # play sound if successful
            # todo: make this available to object params
            self.play_sound = True
            if self.play_sound:
                if prediction == stim:
                    rand = random.randint(1, self.num_of_success_sounds)
                    playsound.playsound(self.audio_success_path[1])#(self.audio_success_path[rand])

            if self.co_learning:
                self.batch_stack[self.stack_order[stim]].append(np.squeeze(epochs.get_data()))
                if all(self.batch_stack):
                    data_batched = [data_stack.pop(0) for data_stack in self.batch_stack]
                    self.model.partial_fit(data_batched, self.keys, epochs, sfreq)
                    pickle.dump(self.model, open(os.path.join(self.session_directory, 'model.pickle'), 'wb'))
            target_predictions.append((int(stim), int(prediction)))

            # Reset the clock for the next buffer
            timer.reset()

            # skip according to mode
            if self.mode == 'practice':
                if stim == prediction:
                    num_tries = 0  # if successful, reset num_tries to 0
                    threshold_skip += 1
                else:
                    num_tries += 1
                # Update the feedback according the prediction
                skip_bool = True if num_tries >= self.skip_after or threshold_skip >= self.threshold else False
                feedback.update(predict_stim=prediction, skip=skip_bool, progress_criteria=self.threshold)

            if self.mode == 'test':
                num_tries += 1
                # Update the feedback according the prediction
                feedback.update(predict_stim=prediction, skip=(num_tries >= self.skip_after), progress_criteria=self.skip_after)

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
        self.session_directory = self.create_session_folder(self.subject_directory,experiment_type=self.experiment_type)

        # Create experiment's metadata
        self.write_metadata()

        # Init experiments configurations
        self.win = visual.Window(monitor='testMonitor', fullscr=full_screen)

        # turn on EEG streaming
        if use_eeg:
            self.eeg.on()

        # For each stim in the trials list
        for ind_stim, stim in enumerate(self.labels):
            # Init feedback instance
            # TODO: create a black screen feedback in the  baseline recording (the first 1 sec)
            # if ind_stim != 0:
            #     print('Roy is a fool!!!')
            #     print(feedback.black_screen_path)
            #     feedback.img_stim = visual.ImageStim(feedback.win, image=feedback.black_screen_path)
            #     feedback.img_stim.draw()
            time.sleep(self.baseline_length)  # sleep for baseline features

            feedback = Feedback(self.win, stim, self.buffer_time, self.skip_after)

            # Use different thread for online learning of the model
            threading.Thread(target=self._learning_model,
                             args=(feedback, stim), daemon=True).start()

            # Maintain visual feedback on screen
            timer = core.Clock()

            num_first_stim = -1  # for the first stim to be heard
            while not feedback.stop:
                num_first_stim += 1
                feedback.display(current_time=timer.getTime())
                if ind_stim == 0 and num_first_stim == 0: # for the first stim to be heard ( the other appears in "feedback.display" func)
                    if self.stim_sound:
                        playsound.playsound(self.audio_first_stim_pathes[self.labels[ind_stim]])
                # Reset the timer according the buffer time attribute
                if timer.getTime() > self.buffer_time:
                    timer.reset()

                # Halt if escape was pressed
                if 'escape' == self.get_keypress():
                    sys.exit(-1)

            # Waiting for key-press between trials
            next_stim = self.labels[ind_stim+1] if ind_stim < len(self.labels)-1 else 'end'
            self._wait_between_trials(feedback, self.eeg, use_eeg,next_stim,self.audio_next_pathes,stim_sound=self.stim_sound)

        # turn off EEG streaming
        if use_eeg:
            self.eeg.off()

    def plot_online_results(self):
        path = f"{self.session_directory}/results.json"
        with open(path) as f:
            data = json.load(f)
        rep_on_class = len(data[0])
        num_of_trials_class = len(data)/3
        results_dict = dict([(str(i),0) for i in self.keys])
        expected = []
        prediction = []
        for trial in data:
            for ind in trial:
                if ind[0]==ind[1]:
                    results_dict[str(ind[0])] += 1/(rep_on_class*num_of_trials_class)
                expected.append(ind[0])
                prediction.append(ind[1])

        # the bar plot
        labels = list(self.label_dict.values())
        classes = list(results_dict.keys())
        values = list(results_dict.values())
        plt.bar(classes,values,color = (0.5,0.1,0.5,0.6))
        plt.title('Online results - The prediction percentage for each class\n')
        plt.xlabel('Prediction percentage')
        plt.ylabel('Classes ')
        plt.xticks(classes,labels)
        plt.show()

        # the confusion matrix
        cm = confusion_matrix(expected,prediction)
        ax = seaborn.heatmap(cm/np.sum(cm),fmt='.2%', annot=True, cmap='Blues')
        ax.set_title('Online results - confusion matrix\n')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ')
        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()
        sys.exit(0)
