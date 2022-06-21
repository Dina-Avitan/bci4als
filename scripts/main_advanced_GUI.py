from GUI_offline import GUI_offline
from GUI_online import GUI_online
from choose_GUI import choose_GUI
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

# g = GUI()
# param = g.run_gui()
# print(param)
