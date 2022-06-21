from tkinter import *
from tkinter.messagebox import showinfo
#from data_folder import create_session_folder, json_save


class GUI_online:
    def __init__(self):
        # window parameters
        self.win = Tk()
        self.win.geometry('500x300')
        self.Name = None
        self.entry_place = .4
        self.label_place = .1
        Label(self.win, text="Before we will start the session, enter here the details:", font=('Helvetica 13')) \
            .place(relx=.1, rely=.0)
        self.rec_params: dict = {}

        # name
        v = StringVar(self.win, value='avi')  # set default text
        self.entry_name = Entry(self.win, textvariable=v)
        self.entry_name.place(relx=self.entry_place, rely=.1)
        Label(self.win, text="Name:").place(relx=self.label_place, rely=.1)

        # synthtic
        self.use_synthetic = StringVar(self.win)
        synthetic_spin = Spinbox(self.win, values=['True', 'False'], textvariable=self.use_synthetic, width=20)
        self.use_synthetic.set('False')
        Label(self.win, text='Use synthetic board:').place(relx=self.label_place, rely=.2)
        synthetic_spin.place(relx=self.entry_place, rely=.2)


        # Classes
        v = StringVar(self.win, value='012') # set default text
        self.entry_classes = Entry(self.win, textvariable=v)
        self.entry_classes.place(relx=self.entry_place, rely=.4)
        Label(self.win, text="Classes: 0-right, 1-left, 2-idel, 3-tongue, 4-hands.").place(relx=self.label_place, rely=.3)

        # Trial length
        v = StringVar(self.win, value='5')  # set default value
        self.entry_trial_length = Entry(self.win, textvariable=v)
        self.entry_trial_length.place(relx=self.entry_place, rely=.5)
        Label(self.win, text="Trial length (sec):").place(relx=self.label_place, rely=.5)

        # num_trials
        v = StringVar(self.win, value='12')  # set default value
        self.entry_trials = Entry(self.win, textvariable=v)
        self.entry_trials.place(relx=self.entry_place, rely=.6)
        Label(self.win, text="Number of trials:").place(relx=self.label_place, rely=.6)

        # skip_after
        v = StringVar(self.win, value='1')  # set default value
        self.entry_skip_after = Entry(self.win, textvariable=v)
        self.entry_skip_after.place(relx=self.entry_place, rely=.7)
        Label(self.win, text="Skip_after:").place(relx=self.label_place, rely=.7)

        # use sound
        self.use_sound = StringVar(self.win)
        synthetic_spin = Spinbox(self.win, values=['True', 'False'], textvariable=self.use_sound, width=20)
        self.use_sound.set('False')
        Label(self.win, text='Use_sound:').place(relx=self.label_place, rely=.8)
        synthetic_spin.place(relx=self.entry_place, rely=.8)

    def submit_button(self):
        # def check_validity():  # todo: add validation

        def get_valid_data():
            # check_validty()
            classes = self.entry_classes.get()
            classes_list =  [int(i) for i in classes]
            self.rec_params = {
                'use_synthetic': self.use_synthetic.get(),
                'subject_name': self.entry_name.get(),
                'classes_keys': classes_list,
                'num_trials': int(self.entry_trials.get()),
                'skip_after': int(self.entry_skip_after.get()),
                'trial_length': int(self.entry_trial_length.get()),
                'stim_sound': self.use_sound.get()
            }
            # save to json
            # folder_path = create_session_folder(self.entry_name.get())
            # json_save(folder_path, "params.json", self.rec_params)
            # estimated_time = self.rec_params['blocks_N'] * self.rec_params['trials_N'] * self.rec_params['StimOnset'] \
            #                  * self.rec_params['interTime']
            # showinfo("Inforamtion", f"The session will open in a few seconds. \nEstimated time for "
            #                         f"{self.rec_params['blocks_N']} blocks:\n{float('{0:.2f}'.format(estimated_time))}")
            self.win.destroy()  # close the window

        Button(self.win, text="submit", command=get_valid_data).place(relx=.5, rely=.9)

    def find_stim_dict(self):
        stim_dict = {}
        if self.stim.get() == 'black_shapes':
            stim_dict = {"NON_TARGET": "black circle",
                         "TARGET_1": "black square",
                         "TARGET_2": "black triangle"
                        }
        elif self.stim.get() == 'blue_shpaes':
            stim_dict = { "NON_TARGET": "circle",
                          "TARGET_1": "square",
                          "TARGET_2": "triangular"
                        }
        return stim_dict

    def run_gui(self):
        self.submit_button()
        self.win.mainloop()
        return self.rec_params
