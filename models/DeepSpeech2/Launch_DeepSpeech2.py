import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from jiwer import wer
from sklearn.model_selection import train_test_split
from pathlib import Path


# from IPython import display


def do_everything(PATH_WAV, PATH_TXT, PATH_LOGS, PATH_SAVED_MODEL, epochs, freq_of_save):
    df_wer = pd.DataFrame(columns=['WER', 'Epochs'])
    df_wer.to_csv("WER_CSV", sep=';', encoding='utf-8', index=False)

    # CHECK GPU RUNNING WITH CUDA
    gpus = tf.config.list_physical_devices('GPU')
    gpu = gpus[0]

    tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # All wav files have been transform to 16bits 16kHz for 256 000 bits/sec

    # Prepare the data for a DF to match Keras tutorial
    wav_files = [f for f in os.listdir(PATH_WAV)]
    print(wav_files)

    txt_files = [f for f in os.listdir(PATH_TXT)]
    # print(txt_files)
    wav_labels = []
    print(txt_files)

    for txt in txt_files:
        myfile = open(PATH_TXT + txt, "rt", encoding="utf-8")  # open lorem.txt for reading text
        text = myfile.read()
        # text = text.replace("é", "e")
        # text = text.replace("è", "e")
        # text = text.replace("à", "a")
        # text = text.replace("û", "u")
        # text = text.replace("ê", "e")
        # text = text.replace("’", "'")
        # text = text.replace("ô", "o")
        # text = text.replace("ï", "i")
        # text = text.replace("ë", "e")
        # text = text.replace("ù", "u")
        # text = text.replace("ö", "o")
        # text = text.replace("-", " ")
        # text = text.replace(",", " ")
        wav_labels.append(text)  # read the entire file to string
        myfile.close()  # close the file
    print(wav_labels)  # print string contentsm

    ## put that in pandas DF
    data_wav = {'wav': wav_files, 'labels': wav_labels}
    df_one_word = pd.DataFrame(data=data_wav)

    # Split train validate with shuffle

    X = data_wav['wav']
    y = data_wav['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=104,
                                                        train_size=0.8, shuffle=True)

    # Create 2 df for train and validate
    df_train = {'wav': X_train, 'labels': y_train}
    df_val = {'wav': X_test, 'labels': y_test}

    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the training set: {len(df_val)}")

    # The set of characters accepted in the transcription.
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    print(
        f"The vocabulary is: {char_to_num.get_vocabulary()} "
        f"(size ={char_to_num.vocabulary_size()})"
    )

    # An integer scalar Tensor. The window length in samples.
    frame_length = 256
    # An integer scalar Tensor. The number of samples to step.
    frame_step = 160
    # An integer scalar Tensor. The size of the FFT to apply.
    # If not provided, uses the smallest power of 2 enclosing frame_length.
    fft_length = 384

    # wavs_path = "C://Users//trist//PycharmProjects//AudioMNIST//Open_SLR_data//wav_all_16_bits//"

    def encode_single_sample(wav_file, label):
        ###########################################
        ##  Process the Audio
        ##########################################
        # 1. Read wav file
        file = tf.io.read_file(PATH_WAV + wav_file)
        # 2. Decode the wav file
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        # 3. Change type to float
        audio = tf.cast(audio, tf.float32)
        # 4. Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        ###########################################
        ##  Process the label
        ##########################################
        # 7. Convert label to Lower case
        label = tf.strings.lower(label)
        # 8. Split the label
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        # 9. Map the characters in label to numbers
        label = char_to_num(label)
        # 10. Return a dict as our model is expecting two inputs
        return spectrogram, label

    # test a single encoding to see if it works
    encode_single_sample("0a7e5f3a-faa4-4e03-84ed-0719d20ec79d.wav", "oui")

    batch_size = 10
    # Define the trainig dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["wav"]), list(df_train["labels"]))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["wav"]), list(df_val["labels"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # fig = plt.figure(figsize=(8, 5))
    ## Print spectrogram
    # for batch in train_dataset.take(1):
    #     spectrogram = batch[0][0].numpy()
    #     spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    #     label = batch[1][0]
    #     print(label)
    #     # Spectrogram
    #     label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    #     ax = plt.subplot(2, 1, 1)
    #     ax.imshow(spectrogram, vmax=1)
    #     ax.set_title(label)
    #     ax.axis("off")
    #     # Wav
    #     file = tf.io.read_file(PATH_WAV + list(df_train["wav"])[0])
    #     audio, _ = tf.audio.decode_wav(file)
    #     audio = audio.numpy()
    #     ax = plt.subplot(2, 1, 2)
    #     plt.plot(audio)
    #     ax.set_title("Signal Wave")
    #     ax.set_xlim(0, len(audio))
    #     display.display(display.Audio(np.transpose(audio), rate=16000))
    # plt.show()

    def CTCLoss(y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
        """Model similar to DeepSpeech2."""
        # Model's input
        input_spectrogram = layers.Input((None, input_dim), name="input")
        # Expand the dimension to use 2D CNN.
        x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
        # Convolution layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        # Convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        # Model
        model_DeepSpeech_2 = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model_DeepSpeech_2.compile(optimizer=opt, loss=CTCLoss)
        return model_DeepSpeech_2

    # Get the model
    model_DeepSpeech_2 = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=512,
    )
    model_DeepSpeech_2.summary(line_length=110)

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        # Iterate over the results and get back the text
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    def decode_predictions(pred):
        input_len = np.ones(pred.shape[0])
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        # Iterate over the results and get back the text
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    # A callback class to output a few transcriptions during training
    class CallbackEval(keras.callbacks.Callback):
        """Displays a batch of outputs after every epoch."""

        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def on_epoch_end(self, epoch: int, logs=None):
            if epoch % 10 == 1:
                predictions = []
                targets = []
                for batch in self.dataset:
                    X, y = batch
                    batch_predictions = model_DeepSpeech_2.predict(X)
                    batch_predictions = decode_batch_predictions(batch_predictions)
                    predictions.extend(batch_predictions)
                    for label in y:
                        label = (
                            tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                        )
                        targets.append(label)
                wer_score = wer(targets, predictions)
                print("-" * 100)
                print(f"Word Error Rate: {wer_score:.4f}")
                print("-" * 100)

                df_wer = pd.read_csv("WER_CSV", sep=';', encoding='utf-8')
                df_wer = df_wer.append({"WER": wer_score, "Epochs": epoch}, ignore_index=True)
                df_wer.to_csv("WER_CSV", sep=';', encoding='utf-8', index=False)
                with open('output_prediction.txt', 'a') as file_output:
                    file_output.write(f"Epochs : {epoch}\n")
                    file_output.write(f"Word Error Rate : {wer_score}\n")
                    for i in np.random.randint(0, len(predictions), 2):
                        file_output.write(f"Target    : {targets[i]}\n")
                        file_output.write(f"Prediction    : {predictions[i]}\n")
                        file_output.write("-----------------------------------\n")
                        # print(f"Target    : {targets[i]}")
                        # print(f"Prediction: {predictions[i]}")
                        # print("-" * 100)
                    file_output.write("----------------------------------------------------------------------\n")
                    file_output.close()

    # Define the number of epochs.
    # Callback function to check transcription on the val set.
    validation_callback = CallbackEval(validation_dataset)

    # checkpoint_filepath = 'C://Users//trist//PycharmProjects//AudioMNIST//Saved_Model_OPEN_SLR//model_51_epochs.hdf5'
    # Save model every 10 epochs
    checkpoint_path = PATH_SAVED_MODEL + "/cp-{epoch:04d}.ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_freq=freq_of_save)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOGS)
    # Train the model
    history = model_DeepSpeech_2.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[tensorboard_callback, validation_callback, model_checkpoint_callback],
    )

    model_DeepSpeech_2.save_weights(PATH_SAVED_MODEL + 'last_saved_model.hdf5')

    ## PARTIE VALIDATION PRÉDICTION
    # model_DeepSpeech_2 = build_model(
    #     input_dim=fft_length // 2 + 1,
    #     output_dim=char_to_num.vocabulary_size(),
    #     rnn_units=512,
    # )
    # on recharge les poids du model ainsi qu'une ou plusieurs donnée jamais vu pour voir si le model arrive à re transcrire

    # checkpoint_filepath = 'C://Users//trist//PycharmProjects//AudioMNIST//Saved_Model_OPEN_SLR//model_51_epochs.hdf5'
    #
    # model.load_weights(checkpoint_filepath)
    #
    # predictions = []
    # targets = []
    # for batch in validation_dataset:
    #     X, y = batch
    #     batch_predictions = model.predict(X)
    #     batch_predictions = decode_batch_predictions(batch_predictions)
    #     predictions.extend(batch_predictions)
    #     for label in y:
    #         label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    #         targets.append(label)
    # wer_score = wer(targets, predictions)
    # print("-" * 100)
    # print(f"Word Error Rate: {wer_score:.4f}")
    # print("-" * 100)
    # for i in np.random.randint(0, len(predictions), 5):
    #     print(f"Target    : {targets[i]}")
    #     print(f"Prediction: {predictions[i]}")
    #     print("-" * 100)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def escape_slashes(s):
    return s.replace('//', '\\')


def preprocess_labels_before_training(PATH_TXT):
    txt_files = [f for f in os.listdir(PATH_TXT)]
    # print(txt_files)

    for txt in txt_files:
        print(txt_files.index(txt))
        myfile_read = open(PATH_TXT + txt, "r+", encoding="utf-8")  # open lorem.txt for reading text
        text = myfile_read.read()
        print("BEFORE: " + text)
        text = text.replace("é", "e")
        text = text.replace("è", "e")
        text = text.replace("à", "a")
        text = text.replace("û", "u")
        text = text.replace("ê", "e")
        text = text.replace("’", "'")
        text = text.replace("ô", "o")
        text = text.replace("ï", "i")
        text = text.replace("ë", "e")
        text = text.replace("ù", "u")
        text = text.replace("ö", "o")
        text = text.replace("-", " ")
        text = text.replace(",", " ")
        print("AFTER: " + text)
        myfile_read.close()  # close the file

        myfile_replace = open(PATH_TXT + txt, "w+", encoding="utf-8")  # open lorem.txt for reading text
        myfile_replace.write(text)

        myfile_replace.close()  # close the file


if __name__ == "__main__":
    # change parameters here
    PATH_WAV = "wav_all_16k_16bit_mono//wav//"
    # PATH_TXT = "wav_all_16k_16bit_mono//txt//"
    PATH_TXT = "C://Users//trist//Desktop//txt_processed//"
    PATH_LOGS = "logs//"
    PATH_SAVED_MODEL = "saved_models_DeepSpeech2//"
    # saved_model_16k_16bit_mono
    epochs = 15
    freq_of_save = 50
    print("CURRENT WORKING DIRECTORY IS : " + os.getcwd())
    # do_everything(PATH_WAV, PATH_TXT, PATH_LOGS, PATH_SAVED_MODEL, epochs,freq_of_save)
    preprocess_labels_before_training(PATH_TXT)
    print("DONE yeepee")