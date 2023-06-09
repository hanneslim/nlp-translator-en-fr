# source: https://www.kaggle.com/code/ahmedgamal12/model-machine-translation-from-eng-fr/notebook

# basic libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# cleaning data
import re
import os
import nltk
# nltk.download("stopwords")
# nltk.download('punkt')

# save vocabulary in files
import pickle

# tokenization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Model
from tensorflow.keras.layers import LSTM, Embedding, Input, Dense, SpatialDropout1D, Activation
from tensorflow.keras.models import Model, Sequential

# training model dependanices
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

print(len(tf.config.experimental.list_physical_devices('GPU')) > 0)

# read in dataSet for training
df = pd.read_csv("./dataset/eng_-french.csv")
df.columns = ["english", "french"]
# print(df.head())
# print(df.info())
data = df[:170000]
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


# print(data.iloc[1, 0], clean_english(data.iloc[1, 0]))

# clean french language
def clean_french(text):
    text = text.lower()  # lower case
    # remove any characters not a-z and ?!,'
    # characters a-z and (éâàçêêëôîû) chars of french lang which contain accent
    text = re.sub(u"[^a-zéâàçêêëôîû!?',]", " ", text)
    return text

# print(data.iloc[4,1],clean_french(data.iloc[4,1]))


# apply cleaningFunctions to dataframe
data["english"] = data["english"].apply(lambda txt: clean_english(txt))
data["french"] = data["french"].apply(lambda txt: clean_french(txt))

# add <start> <end> token to decoder sentence (french)
data["french"] = data["french"].apply(lambda txt: f"<start> {txt} <end>")

# print(data.sample(10))

# english tokenizer
english_tokenize = Tokenizer(filters='#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
english_tokenize.fit_on_texts(data["english"])
num_encoder_tokens = len(english_tokenize.word_index)+1
# print(num_encoder_tokens)
encoder = english_tokenize.texts_to_sequences(data["english"])
# print(encoder[:5])
max_encoder_sequence_len = np.max([len(enc) for enc in encoder])
# print(max_encoder_sequence_len)

# french tokenizer
french_tokenize = Tokenizer(filters="#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n")
french_tokenize.fit_on_texts(data["french"])
num_decoder_tokens = len(french_tokenize.word_index)+1
# print(num_decoder_tokens)
decoder = french_tokenize.texts_to_sequences(data["french"])
# print(decoder[:5])
max_decoder_sequence_len = np.max([len(dec) for dec in decoder])
# print(max_decoder_sequence_len)


idx_2_txt_decoder = {k: i for i, k in french_tokenize.word_index.items()}
# print(idx_2_txt_decoder[1])
idx_2_txt_encoder = {k: i for i, k in english_tokenize.word_index.items()}
# print(idx_2_txt_encoder[2])

idx_2_txt_decoder[0] = "<pad>"
idx_2_txt_encoder[0] = "<pad>"

# save idx_2_txt_encoder and idx_2_txt_decoder , english_tokenize.word_index , french_tokenize.word_index
pickle.dump(english_tokenize.word_index, open("./saves/word_2_idx_input.txt", "wb"))
pickle.dump(french_tokenize.word_index, open("./saves/word_2_idx_target.txt", "wb"))
pickle.dump(idx_2_txt_encoder, open("./saves/idx_2_word_input.txt", "wb"))
pickle.dump(idx_2_txt_decoder, open("./saves/idx_2_word_target.txt", "wb"))

# pad sequences
encoder_seq = pad_sequences(encoder, maxlen=max_encoder_sequence_len, padding="post")
# print(encoder_seq.shape)
decoder_inp = pad_sequences([arr[:-1] for arr in decoder], maxlen=max_decoder_sequence_len, padding="post")
# print(decoder_inp.shape)
decoder_output = pad_sequences([arr[1:] for arr in decoder], maxlen=max_decoder_sequence_len, padding="post")
# print(decoder_output.shape)

# print([idx_2_txt_decoder[i] for i in decoder_output[0]])
# print([idx_2_txt_encoder[i] for i in encoder_seq[0]])

# decoder_output_categorical = to_categorical(decoder_output, num_classes=num_decoder_tokens+1)
# print(decoder_output_categorical.shape)

# Design LSTM NN (Encoder & Decoder)
# encoder model
encoder_input=Input(shape=(None,),name="encoder_input_layer")
encoder_embedding=Embedding(num_encoder_tokens,300,input_length=max_encoder_sequence_len,name="encoder_embedding_layer")(encoder_input)
encoder_lstm=LSTM(256,activation="tanh",return_sequences=True,return_state=True,name="encoder_lstm_1_layer")(encoder_embedding)
encoder_lstm2=LSTM(256,activation="tanh",return_state=True,name="encoder_lstm_2_layer")(encoder_lstm)
_,state_h,state_c=encoder_lstm2
encoder_states=[state_h,state_c]

# decoder model
decoder_input=Input(shape=(None,),name="decoder_input_layer")
decoder_embedding=Embedding(num_decoder_tokens,300,input_length=max_decoder_sequence_len,name="decoder_embedding_layer")(decoder_input)
decoder_lstm=LSTM(256,activation="tanh",return_state=True,return_sequences=True,name="decoder_lstm_layer")
decoder_outputs,_,_=decoder_lstm(decoder_embedding,initial_state=encoder_states)
decoder_dense=Dense(num_decoder_tokens+1,activation="softmax",name="decoder_final_layer")
outputs=decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], outputs)
# print(model.summary())

# train model
loss = tf.losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(
    [encoder_seq, decoder_inp],
    decoder_output,
    epochs=300,  # 80
    batch_size=450,  # 450
    # validation_split=0.2,
    # callbacks=[callback]
)
# adam 20000 samples epoch 200 batch size 60 loss resulted in 0.059 ->rmsprop loss resulted in 0.2
# adam 170000 samples epochs 80 batch size 450 loss resulted in 0.0437
# rmsprop 170000 samples epochs 300 batch size 450 resulted in 0.0415

# save model
model.save("./model-experimental/Translate_Eng_FR.h5")
model.save_weights("./model-experimental/model_NMT")
