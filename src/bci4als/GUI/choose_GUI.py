from tkinter import *
from tkinter.messagebox import showinfo
#from data_folder import create_session_folder, json_save


class choose_GUI:
    def __init__(self):
        # window parameters
        self.win = Tk()
        self.win.geometry('500x100')
        self.Name = None
        self.entry_place = .4
        self.label_place = .1
        Label(self.win, text="Choose session type:", font=('Helvetica 11')) \
            .place(relx=.1, rely=.0)
        self.rec_params: dict = {}

        # session type
        self.choose_type = StringVar(self.win)
        type_spin = Spinbox(self.win, values=['Offline', 'Online'], textvariable=self.choose_type, width=20)
        self.choose_type.set('Offline')
        Label(self.win, text='Type:').place(relx=self.label_place, rely=.2)
        type_spin.place(relx=self.entry_place, rely=.2)


    def submit_button(self):
        # def check_validity():  # todo: add validation

        def get_valid_data():
            # check_validty()
            self.rec_params = {
                'type': self.choose_type.get(),
            }
            # save to json
            # folder_path = create_session_folder(self.entry_name.get())
            # json_save(folder_path, "params.json", self.rec_params)
            # estimated_time = self.rec_params['blocks_N'] * self.rec_params['trials_N'] * self.rec_params['StimOnset'] \
            #                  * self.rec_params['interTime']
            # showinfo("Inforamtion", f"The session will open in a few seconds. \nEstimated time for "
            #                         f"{self.rec_params['blocks_N']} blocks:\n{float('{0:.2f}'.format(estimated_time))}")
            self.win.destroy()  # close the window

        Button(self.win, text="submit", command=get_valid_data).place(relx=.5, rely=.5)


    def run_gui(self):
        self.submit_button()
        self.win.mainloop()
        return self.rec_params
