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

# for wordCloud
import re
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.data.path += ['/mnt/share/nltk_data']
from wordcloud import WordCloud

# for NLP
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from spacy.lang.fr.examples import sentences
import numpy as np
import re
import random
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

regex = r'[a-zA-ZÀ-ÖØ-öø-ÿ]+'


def do_transcription():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
    processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained("bhuang/asr-wav2vec2-french")
    model_sample_rate = processor_with_lm.feature_extractor.sampling_rate

    wav_path_dir = browseDir()
    for root, dirs, files in os.walk(wav_path_dir):
        for f in files:
            prefix, suffix = os.path.splitext(f)

            # path to your audio file
            waveform, sample_rate = torchaudio.load(wav_path_dir + '//'+ f)
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
            # clear textfox before inserting new prediction
            # TRANSCRIPTION_text_box.delete(1.0, END)
            TRANSCRIPTION_text_box.insert(END, predicted_sentence + '.\n\n')


# Function for opening the
# file explorer window
def browseFiles():
    curr_directory = os.getcwd()
    filename = filedialog.askopenfilename(initialdir=curr_directory,
                                          title="Select a File")

    # Change label contents
    label_file_wav_explorer.configure(text="File Opened: " + filename)
    return filename


def browseDir():
    curr_directory = os.getcwd()
    folder_path = filedialog.askdirectory(initialdir=curr_directory,
                                          title="Select a Directory")
    return folder_path


def load_RTTM_PLOT():
    filename = browseFiles()
    rttm_file = filename
    _, timecodes = load_rttm(rttm_file).popitem()
    figure, ax = plt.subplots()
    notebook.plot_annotation(timecodes, ax=ax, time=True, legend=True)
    plt.show()


def do_TFIDF_POSITIVITY():
    # TFIDF part
    nlp = spacy.load('fr_core_news_lg')
    transcript_of_textfield = str(TRANSCRIPTION_text_box.get(1.0, "end-1c"))
    doc = nlp(transcript_of_textfield)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_text = tfidf_vectorizer.fit_transform([' '.join(lemma_punct_stop(doc))])

    df_tfidf = pd.DataFrame(tfidf_text[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["TF-IDF"])
    df_tfidf = df_tfidf.sort_values("TF-IDF", ascending=False)
    plot_tfidf(df_tfidf, 30)
    # POSITIVITY part
    sid = SentimentIntensityAnalyzer()

    sentences = [sentence.text for sentence in doc.sents]

    analyzer = pipeline(
        task='text-classification',
        model="cmarkea/distilcamembert-base-sentiment",
        tokenizer="cmarkea/distilcamembert-base-sentiment"
    )
    result = analyzer(
        "J'aime me promener en forêt même si ça me donne mal aux pieds.",
        return_all_scores=False
    )

    df_camembert = pd.DataFrame(columns=["phrase", "label", "score"])

    for sentence in sentences:
        result = analyzer(sentence, return_all_scores=False)
        df_camembert = df_camembert.append(
            {"phrase": sentence, "label": result[0]['label'], "score": result[0]['score']}, ignore_index=True)
    plot_camembert(df_camembert['label'])


def lemma_punct_stop(doc):
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]


def plot_tfidf(df_tfidf, size):
    max_value = float(df_tfidf.max().values)
    min_value = float(df_tfidf.min().values)
    x_labels = df_tfidf.index[:size]
    y_values = df_tfidf.values.reshape(len(df_tfidf))[:size]

    fig, ax = plt.subplots()
    ax.bar(x_labels, y_values)
    ax.set_xlabel("Mots")
    ax.set_ylabel("Fréquence (max={0:.4f} ; min={1:.4f})".format(max_value, min_value))
    ax.set_title("TF-IDF (Term Frequency-Inverse Document Frequency)")
    plt.xticks([x for x in x_labels], rotation=90)
    plt.show()
    return


def plot_camembert(df):
    x_labels = df.index
    y_values = df.values.reshape(len(df))

    d = np.array(['1 star', '2 stars', '3 stars', '4 stars', '5 stars'])

    y_Idx = np.where(np.unique(d) == np.expand_dims(y_values,-1))[1]

    fig,ax = plt.subplots()
    ax.plot(x_labels, y_Idx)
    ax.set_xlabel("phrase")
    ax.set_ylabel("Score")
    ax.set_title("Positivité par phrases")
    plt.yticks(np.arange(len(d)),d)
    plt.show()
    return


def do_WC():
    path = browseDir()
    print("yo")

    txts = []

    txts.extend(read_txt(path + "/"))

    df = pd.DataFrame(txts, columns=['txt_content'])
    df["clean_txt"] = df.txt_content.apply(clean_text)

    text_to_use = df['clean_txt'].values

    wordcloud = WordCloud(background_color='white').generate(str(text_to_use))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def read_txt(folder):
    txt_list = []
    files_list = os.listdir(folder)
    for file_name in files_list:
        file_content = open(folder + file_name, 'r', encoding='utf-8')
        txt_list.append(file_content.read())
    file_content.close()
    return txt_list


def clean_text(text):
    # Remove figures
    stop_punctuation = [':', '(', ')', '/', '|', ',',
                        '.', '*', '#', '"', '&', '~',
                        '-', '_', '@', '?', '!']
    stoplist = stopwords.words('french')
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenization
    words = text.lower().split()
    stemmer = SnowballStemmer('french')
    # lemmatizer = FrenchLefffLemmatizer()
    words = [stemmer.stem(w) for w in words if w not in stoplist and not w in stop_punctuation]
    return ' '.join(words)


# Create an instance of tkinter window with a frame
win = Tk()
win.title("WAV transcription ")
# setting tkinter window size
win.geometry("1000x800")

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
label_1 = Label(second_frame, text="This GUI does: \n WAV transcription \n RTTM annotation plot \n TFIDF, POSITIVITY "
                                   "and Word Cloud")
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
                                     text="Here is the output of the TRANSCRIPTION \n you can correct it before doing "
                                          "TFIDF and POSITIVITY on it")
TRANSCRIPTION_text_box_label.pack(side=TOP, expand=YES)
import tkinter.scrolledtext as tkscrolled

# Creating a text box for TRANSCRIPTION
TRANSCRIPTION_text_box = tkscrolled.ScrolledText(second_frame, height=30, width=120)
TRANSCRIPTION_text_box.pack(side=TOP, expand=YES)
# ______________________ TRANSCRIPTION end

# ______________________ NLP

# Create a button to search and plot NLP
button_do_TFIDF_POSITIVITY= Button(second_frame, text="Do NLP of TRANSCRIPTION \n tfidf and positivity", command=do_TFIDF_POSITIVITY)
button_do_TFIDF_POSITIVITY.pack(side=TOP, expand=YES)

# ______________________ NLP end

# ______________________ WordCloud

# Create a button to search and plot WordCloud
button_WC = Button(second_frame, text="Browse dir and do WordCloud", command=do_WC)
button_WC.pack(side=TOP, expand=YES)
# ______________________ WordCloud end


win.mainloop()
