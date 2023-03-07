import os
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

import sounddevice as sd
import soundfile as sf
from tkinter import *
from tkinter import messagebox



def create_data_folder():
    if not os.path.exists('data_from_recordings'):
        os.makedirs('data_from_recordings')
        messagebox.showinfo("showinfo", "Folder created")
    else:
        messagebox.showwarning("showwarning", "Folder already existing, can't overwrite it")


def is_in_dir_data():
    if 'data_from_recordings' in os.getcwd():
        return True
    else:
        os.chdir('data_from_recordings')


def Voice_rec():
    is_in_dir_data()

    fs = 16000
    # seconds
    duration = 30
    myrecording = sd.rec(int(duration * fs),
                         samplerate=fs, channels=1)
    sd.wait()
    filename = str(file_text_box.get(1.0, END))
    filename = filename.replace('\n', '')
    return sf.write(filename + ".wav", myrecording, fs)


def save_text():
    is_in_dir_data()

    name_of_file = file_text_box.get(1.0, END)
    name_of_file = name_of_file.replace('\n', '')
    text_file = open(name_of_file + ".txt", "w")
    text_file.write(my_text_box.get(1.0, END))
    text_file.close()


# Create an instance of tkinter window with a frame
win = Tk()
win.title("Txt and WAV generator for easy data recording and labeling")
win.geometry("500x500")
fm = Frame(win)

# Creating a label for creating folder
label_1 = Label(fm, text="If data_from_recordings folder does not exist, press the button to generate it")
label_1.pack(side=TOP, expand=YES)
# Create a button to start the recording and save it in wav
voice_recorder = Button(fm, text="Generate folder",
                        command=create_data_folder)
voice_recorder.pack(side=TOP, expand=YES)
fm.pack(fill=BOTH, expand=YES)

# Creating a label
label_1 = Label(fm, text="The textbox below is the name of the file\n It will be the same for the recording and the "
                         "text\n example : tristan_oui_3")
label_1.pack(side=TOP, expand=YES)

# Create a text box to input file name
file_text_box = Text(fm, height=1, width=30)
file_text_box.pack(side=TOP, expand=YES)

# Creating a label
label_2 = Label(fm, text="Here input the text you want to speak \n then press the save text file button\nexample : oui")
label_2.pack(side=TOP, expand=YES)

# Creating a text box widget with label
my_text_box = Text(fm, height=1, width=30)
my_text_box.pack(side=TOP, expand=YES)

# Create a button to save the text
save = Button(fm, text="Save text file", command=save_text)
save.pack(side=TOP, expand=YES)

# Creating a label
label_2 = Label(fm, text="Click the button below when you want to record \nIt will record for 5s and be saved "
                         "automatically after with the same name file")
label_2.pack(side=TOP, expand=YES)

# Create a button to start the recording and save it in wav
voice_recorder = Button(fm, text="Start and save voice recording", command=Voice_rec)
voice_recorder.pack(side=TOP, expand=YES)
fm.pack(fill=BOTH, expand=YES)

win.mainloop()
