from offline_training import offline_experiment
from online_training import run_experiment
import GUI
from GUI import main
type, path = main()
print(type)
print(path)
if type =="online":
    run_experiment()
elif type == "offline":
    offline_experiment()

