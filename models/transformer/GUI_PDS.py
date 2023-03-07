# GUI for our project that will take an audio input in WAV then do a DIARIZATION
# Creating multiple audios of 10s length, then transcribe them with wav2vec2 model pretrained with french dataset
# After it will output each of the transcription in a textfield for the user to edit
# Then  analyze it a word cloud/TFIDF

import os
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
import tkinter as tk

import sounddevice as sd
import soundfile as sf
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import torch
import torchaudio

from transformers import AutoModelForCTC, Wav2Vec2ProcessorWithLM
from pyannote.database.util import load_rttm

import torch.utils
from torchaudio import *
from matplotlib import pyplot as plt
from pyannote.core import notebook


def do_transcription():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
    processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained("bhuang/asr-wav2vec2-french")
    model_sample_rate = processor_with_lm.feature_extractor.sampling_rate

    wav_path = browseFiles()
    # path to your audio file
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.squeeze(axis=0)  # mono

    # resample
    if sample_rate != model_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
        waveform = resampler(waveform)

    # normalize
    input_dict = processor_with_lm(waveform, sampling_rate=model_sample_rate, return_tensors="pt")

    with torch.inference_mode():
        logits = model(input_dict.input_values.to(device)).logits

    predicted_sentence = processor_with_lm.batch_decode(logits.cpu().numpy()).text[0]
    print(predicted_sentence)
    #clear textfox before inserting new prediction
    TRANSCRIPTION_text_box.delete(1.0,END)
    TRANSCRIPTION_text_box.insert(END, predicted_sentence)

# Function for opening the
# file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File")

    # Change label contents
    label_file_wav_explorer.configure(text="File Opened: " + filename)
    return filename


def load_RTTM_PLOT():
    filename = browseFiles()
    rttm_file = filename
    _, timecodes = load_rttm(rttm_file).popitem()
    figure, ax = plt.subplots()
    notebook.plot_annotation(timecodes, ax=ax, time=True, legend=True)
    plt.show()

def do_NLP_WORDCLOUD():
    print("yo")

# Create an instance of tkinter window with a frame
win = Tk()
win.title("WAV transcription ")
#setting tkinter window size
win.geometry("800x800")

fm = Frame(win)
fm.pack(fill=BOTH, expand=1)

# canvas
my_canvas = Canvas(fm)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# scrollbar
my_scrollbar = tk.Scrollbar(fm, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

# configure the canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind(
    '<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all"))
)

second_frame = Frame(my_canvas)

my_canvas.create_window((0, 0), window=second_frame, anchor="nw")

# Creating a label for creating folder
label_1 = Label(second_frame, text="This GUI does: \n WAV transcription \n RTTM annotation plot \n NLP and Word Cloud")
label_1.pack(side=TOP, expand=YES)

# ______________________ RTTM

# Creating a label RTTM
label_button_RTTM = Label(second_frame, text="Search and plot RTTM annotation of an audio")
label_button_RTTM.pack(side=TOP, expand=YES)

# Create a button to search and plot RTTM file
button_RTTM = Button(second_frame, text="Browse and PLOT RTTM", command=load_RTTM_PLOT)
button_RTTM.pack(side=TOP, expand=YES)
# ______________________ RTTM end

# ______________________ TRANSCRIPTION

# Create a File Explorer label
label_file_wav_explorer = Label(second_frame,
                                text="Find the WAV file you want to translate",
                                width=100, height=4,
                                fg="blue")
label_file_wav_explorer.pack(side=TOP, expand=YES)

# Create a button to launch TRANSCRIPTION
button_TRANSCRIPTION = Button(second_frame, text="Browse and do TRANSCRIPTION of WaV file", command=do_transcription)
button_TRANSCRIPTION.pack(side=TOP, expand=YES)

# Creating a label for TRANSCRIPTION textbox
TRANSCRIPTION_text_box_label = Label(second_frame,
                                     text="Here is the output of the TRANSCRIPTION \n you can correct it before doing NLP")
TRANSCRIPTION_text_box_label.pack(side=TOP, expand=YES)

# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# ______________________ TRANSCRIPTION end

# ______________________ NLP and WordCloud
# Creating a label NLP
label_button_RTTM = Label(second_frame, text="Do the NLP and Wordcloud")
label_button_RTTM.pack(side=TOP, expand=YES)

# Create a button to search and plot NLP WordCloud
button_RTTM = Button(second_frame, text="Do NLP of TRANSCRIPTION", command=do_NLP_WORDCLOUD)
button_RTTM.pack(side=TOP, expand=YES)


# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = Text(second_frame, height=7, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# ______________________ NLP and WordCloud end

win.mainloop()
