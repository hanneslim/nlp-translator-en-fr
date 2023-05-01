# basic libs
import numpy as np
import pandas as pd

# cleaning data
import re
import nltk

# tokenization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model
from tensorflow.keras.layers import LSTM, Embedding, Input, Dense, SpatialDropout1D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model

# read in dataSet for training
df = pd.read_csv("./dataset/eng_-french.csv")
df.columns = ["english", "frensh"]
# print(df.head())
# print(df.info())
data = df[:1000]
# print(data.info())


# clean english column
def clean_english(text):
    text = text.lower()  # lower case
    # remove any characters not a-z and ?!,'
    text = re.sub(u"[^a-z!?',]", " ", text)
    # word tokenization
    text = nltk.word_tokenize(text)
    # join text
    text = " ".join([i.strip() for i in text])
    return text


# clean frensh language
def clean_frensh(text):
    text = text.lower()  # lower case
    # remove any characters not a-z and ?!,'
    # characters a-z and (éâàçêêëôîû) chars of frensh lang which contain accent
    text = re.sub(u"[^a-zéâàçêêëôîû!?',]", " ", text)
    return text

# print(data.iloc[4,1],clean_frensh(data.iloc[4,1]))


# apply cleaningFunctions to dataframe
data["english"] = data["english"].apply(lambda txt: clean_english(txt))
data["frensh"] = data["frensh"].apply(lambda txt: clean_frensh(txt))

# add <start> <end> token to decoder sentence (Frensh)
data["frensh"] = data["frensh"].apply(lambda txt: f"<start> {txt} <end>")

# english tokenizer
english_tokenize = Tokenizer(filters='#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
english_tokenize.fit_on_texts(data["english"])
num_encoder_tokens = len(english_tokenize.word_index)+1
# print(num_encoder_tokens)
encoder = english_tokenize.texts_to_sequences(data["english"])
# print(encoder[:5])
max_encoder_sequence_len = np.max([len(enc) for enc in encoder])
# print(max_encoder_sequence_len)

# frensh tokenizer
french_tokenize = Tokenizer(filters="#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n")
french_tokenize.fit_on_texts(data["frensh"])
num_decoder_tokens = len(french_tokenize.word_index)+1
# print(num_decoder_tokens)
decoder = french_tokenize.texts_to_sequences(data["frensh"])
# print(decoder[:5])
max_decoder_sequence_len = np.max([len(dec) for dec in decoder])
# print(max_decoder_sequence_len)


def make_references():
    # Load the saved model
    model = load_model('./model-saves/Translate_Eng_FR.h5')
    # Load the saved weights into the reference models
    model.load_weights('./model-saves/model_NMT')

    # Get the encoder and decoder layers from the model by name
    encoder_input = model.input[0]
    decoder_input = model.input[1]
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_lstm_2_layer').output
    encoder_states = [state_h_enc, state_c_enc]
    decoder_lstm = model.get_layer('decoder_lstm_layer')
    decoder_embedding = model.get_layer('decoder_embedding_layer')(decoder_input)
    decoder_dense = model.get_layer('decoder_final_layer')

    encoder_reference_model=Model(encoder_input,encoder_states)

    decoder_state_h=Input(shape=(256,))
    decoder_state_c=Input(shape=(256,))
    decoder_input_states=[decoder_state_h,decoder_state_c]

    decoder_outputs,state_h,state_c=decoder_lstm(decoder_embedding,initial_state=decoder_input_states)

    decoder_state=[state_h,state_c]
    decoder_outputs=decoder_dense(decoder_outputs)
    decoder_reference_model=Model([decoder_input]+decoder_input_states,[decoder_outputs]+decoder_state)

    return encoder_reference_model,decoder_reference_model


# prepare Text
def prepare_text(text):
    text = clean_english(text)

    res = [english_tokenize.word_index[i] for i in text.split(" ")]
    pad = pad_sequences([res], maxlen=max_encoder_sequence_len, padding="post")
    return pad

# print(prepare_text("How are you"))


for i in range(20):  # throws error when word is not in vocabulary...!
    enc_model, dec_model = make_references()

    states_value = enc_model(prepare_text(input("Enter text :- ")))

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = french_tokenize.word_index["start"]

    stop_condition = False
    decoded_translaition=""

    while not stop_condition:
        dec_output, h, c = dec_model.predict([empty_target_seq]+states_value)
        sampled_word_index = np.argmax(dec_output[0, -1, :])
        sampled_word = None

        for word, index in french_tokenize.word_index.items():
            if sampled_word_index == index:
                decoded_translaition+=' {}'.format(word)
                sampled_word = word

            if sampled_word == "end" or len(decoded_translaition.split(" ")) >= max_decoder_sequence_len:
                stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_value = [h, c]
    print(decoded_translaition)
