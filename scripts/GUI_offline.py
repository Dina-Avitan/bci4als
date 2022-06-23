from tkinter import *
from tkinter.messagebox import showinfo
#from data_folder import create_session_folder, json_save


class GUI_offline:
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

        # # name
        # v = StringVar(self.win, value='avi')  # set default text
        # self.entry_name = Entry(self.win, textvariable=v)
        # self.entry_name.place(relx=self.entry_place, rely=.1)
        # Label(self.win, text="Name:").place(relx=self.label_place, rely=.1)

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
        v = StringVar(self.win, value='27')  # set default value
        self.entry_trials = Entry(self.win, textvariable=v)
        self.entry_trials.place(relx=self.entry_place, rely=.6)
        Label(self.win, text="Number of trials:").place(relx=self.label_place, rely=.6)

        # use sound
        self.use_sound = StringVar(self.win)
        synthetic_spin = Spinbox(self.win, values=['True', 'False'], textvariable=self.use_sound, width=20)
        self.use_sound.set('False')
        Label(self.win, text='Use_sound:').place(relx=self.label_place, rely=.7)
        synthetic_spin.place(relx=self.entry_place, rely=.7)

    def submit_button(self):
        # def check_validity():

        def get_valid_data():
            # check_validty()
            classes = self.entry_classes.get()
            classes_list = [int(i) for i in classes]
            self.rec_params = {
                'use_synthetic': self.use_synthetic.get(),
                'classes_keys': classes_list,
                'num_trials': int(self.entry_trials.get()),
                'trial_length': int(self.entry_trial_length.get()),
                'stim_sound': self.use_sound.get()
            }#'subject_name': self.entry_name.get(),
            self.win.destroy()  # close the window

        Button(self.win, text="submit", command=get_valid_data).place(relx=.5, rely=.8)


    def run_gui(self):
        self.submit_button()
        self.win.mainloop()
        return self.rec_params
