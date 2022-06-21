import pickle
import sys

import brainflow.board_shim

from bci4als.experiments.online import OnlineExperiment
from bci4als.eeg import EEG


def run_experiment(advanced_gui={}):

    model = pickle.load(open(OnlineExperiment.ask_model_directory(),'rb'))
    # re-fit model? (recommended)
    re_fit = True
    if re_fit:
        model.offline_training(model_type='simple_svm')

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2

    data_type = SYNTHETIC_BOARD
    if advanced_gui:
        if advanced_gui['use_synthetic']:
            data_type = SYNTHETIC_BOARD
        else:
            data_type = CYTON_DAISY

    # select buffer time
    # buffer_time = 5
    # if model.epochs.get_data()[0].shape[1]//125 != buffer_time:
    #     raise IndexError(f"Model buffer time must match online buffer time. change buffer time to"
    #                      f" {model.epochs.get_data()[0]//125} or change model")

    gain = {"1": 0, "2":  1, "4": 2, "6": 3, "8": 4, "12": 5, "24": 6}
    configurations = ''.join([''.join(f"x{str(i + 1)}0{gain['6']}0110X") for i in range(8)] +
                             [''.join(f"x{i}0{gain['6']}0110X") for i in ['Q', 'W', 'E', 'R']] + [
                ''.join(f"x{i}131000X") for i in ['T', 'Y', 'U', 'I']])
    eeg = EEG(board_id=data_type, config_json_converted=configurations)
    # If mode= 'practice': It will skip after skip_after errors. it will skip after threshold successes
    # If mode= 'test': It will not skip. It will run skip_after times whether you succeed or fail the trial
    if advanced_gui:
        exp = OnlineExperiment(eeg=eeg, model=model, num_trials=advanced_gui['num_trials'],
                               buffer_time=advanced_gui['trial_length'], threshold=2,
                               skip_after=advanced_gui['skip_after'], co_learning=True, debug=False,
                               mode='test', stim_sound=False, keys=advanced_gui['classes_keys'])
    else:
        exp = OnlineExperiment(eeg=eeg, model=model, num_trials=9, buffer_time=5, threshold=2, skip_after=1,
                               co_learning=True, debug=False, mode='test', stim_sound=False, keys=(0, 1, 2))


    exp.run(use_eeg=True, full_screen=True)
    exp.plot_online_results()

if __name__ == '__main__':
    try:
        run_experiment()
    except brainflow.board_shim.BrainFlowError as err:
        print('Please make sure the board is ON. Maybe try to restart the board.')
    sys.exit(0)


