from bci4als.GUI.GUI_offline import *
from bci4als.GUI.GUI_online import *
from bci4als.GUI.choose_GUI import *
from offline_training import offline_experiment
from online_training import run_experiment

type = choose_GUI()
type_dict= type.run_gui()
if type_dict['type'] == 'Offline':
    g = GUI_offline()
    param = g.run_gui()
    print(param)
    offline_experiment(advanced_gui=param)
if type_dict['type'] == 'Online':
    g = GUI_online()
    param = g.run_gui()
    print(param)
    run_experiment(advanced_gui=param)
