import pickle

from bci4als.ml_model import MLModel
from bci4als.experiments.online import OnlineExperiment
from bci4als.eeg import EEG


def run_experiment(model_path: str):

    model = pickle.load(open(model_path, 'rb'))

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2

    gain = {"1": 0, "2":  1, "4": 2, "6": 3, "8": 4, "12": 5, "24": 6}
    configurations = ''.join([''.join(f"x{str(i + 1)}0{gain['6']}0110X") for i in range(8)] +
                             [''.join(f"x{i}0{gain['6']}0110X") for i in ['Q', 'W', 'E', 'R']] + [
                ''.join(f"x{i}131000X") for i in ['T', 'Y', 'U', 'I']])
    eeg = EEG(board_id=SYNTHETIC_BOARD, config_json_converted=configurations)

    exp = OnlineExperiment(eeg=eeg, model=model, num_trials=10, buffer_time=4, threshold=3, skip_after=3,
                           co_learning=False, debug=False)

    exp.run(use_eeg=True, full_screen=True)


if __name__ == '__main__':

    model_path = r'../recordings/synthetic_board/22/trained_model.pickle'
    # model_path = None  # use if synthetic
    run_experiment(model_path=model_path)

# PAY ATTENTION!
# If synthetic - model Path should be none
# otherwise choose a model path
