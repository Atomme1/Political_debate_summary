from pydub import AudioSegment
from pydub.silence import split_on_silence

# Variables for the audio file You need to download this file from here:
# https://etc.usf.edu/lit2go/1/alices-adventures-in-wonderland/1/chapter-i-down-the-rabbit-hole/
file_path = "audio_chunk//test_diari_tristan.wav"
file_name = file_path.split('/')[-1]
audio_format = "wav"

# Reading and splitting the audio file into chunks
sound = AudioSegment.from_file(file_path, format=audio_format)
audio_chunks = split_on_silence(sound
                                , min_silence_len=100
                                , silence_thresh=-45
                                , keep_silence=50
                                )

# Putting the file back together
combined = AudioSegment.empty()
i = 0
for chunk in audio_chunks:
    chunk.export(f'./audio_chunk/{i}_{file_name}', format=audio_format)
    i += 1
