from offline_training import offline_experiment
from online_training import run_experiment
from bci4als.GUI import GUI_pygame

# # from GUI import main
type, path ,gui_keys = GUI_pygame.main()
print("main GUI worked")

if type =="online":
    run_experiment()
elif type == "offline":
    offline_experiment(pygame_gui_folder_path =path,pygame_gui_keys=gui_keys)

