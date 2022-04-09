import pickle

from bci4als.ml_model import MLModel
from bci4als.experiments.online import OnlineExperiment
from bci4als.eeg import EEG


def run_experiment(model_path: str):

    model = pickle.load(open(model_path, 'rb'))

    # re-fit model? (recommended)
    re_fit = True
    if re_fit:
        model.offline_training(model_type='simple_svm')

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2

    # select buffer time
    buffer_time = 5
    # if model.epochs.get_data()[0].shape[1]//125 != buffer_time:
    #     raise IndexError(f"Model buffer time must match online buffer time. change buffer time to"
    #                      f" {model.epochs.get_data()[0]//125} or change model")

    gain = {"1": 0, "2":  1, "4": 2, "6": 3, "8": 4, "12": 5, "24": 6}
    configurations = ''.join([''.join(f"x{str(i + 1)}0{gain['6']}0110X") for i in range(8)] +
                             [''.join(f"x{i}0{gain['6']}0110X") for i in ['Q', 'W', 'E', 'R']] + [
                ''.join(f"x{i}131000X") for i in ['T', 'Y', 'U', 'I']])
    eeg = EEG(board_id=SYNTHETIC_BOARD, config_json_converted=configurations)
    # If mode= 'practice': It will skip after skip_after errors. it will skip after threshold successes
    # If mode= 'test': It will not skip. It will run skip_after times whether you succeed or fail the trial
    exp = OnlineExperiment(eeg=eeg, model=model, num_trials=3, buffer_time=buffer_time, threshold=3, skip_after=4,
                           co_learning=True, debug=False, mode='test',stim_sound=False,keys=(2,3,4))

    exp.run(use_eeg=True, full_screen=True)
    exp.plot_online_results()

if __name__ == '__main__':

    model_path = r'../recordings/avi_2022/11/pre_laplacian.pickle'
    # model_path = None  # use if synthetic
    run_experiment(model_path=model_path)

# PAY ATTENTION!
# If synthetic - model Path should be none
# otherwise choose a model path
