from offline_training import offline_experiment
from online_training import run_experiment
import GUI
from GUI import main
type, path ,gui_keys= main()

if type =="online":
    run_experiment()
elif type == "offline":
    offline_experiment(gui_folder_path =path,gui_keys=gui_keys)

