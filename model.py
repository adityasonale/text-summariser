import numpy as np
import pandas as pd
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import keras

import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K 
from tensorflow.python.keras.layers import Layer

from utils.attention import AttentionLayer

from keras.models import load_model
from keras.utils import plot_model


# Paths

DATASET_PATH = r"D:\Datasets\news_summary_more.csv"
GLOVE_PATH = r"D:\Datasets\glove\glove.42B.300d.txt\glove.42B.300d.txt"



class Abstractive_Summariser:

    def __init__(self):
        self.input_lengths = []
        self.output_lengths = []
        self.embedding_matrix = {}
        self.word_corpus = set()
        self.inter_words = None
        self.max_text_len = 0
        self.max_text_len = 0

    def preprocess(self,text):
        text = text.lower()  # lowering the text
        text = [contractions.fix(word) for word in text.split(" ")]  # word contraction
        text = " ".join(text)
        text = [word for word in text.split(" ") if word not in stopwords.words('english')]
        text = " ".join(text)
        text = re.sub(r"\'s ","",text)   # remove 's from the sentence
        text =  re.sub(r"\(.*\)","",text) # remove words written in ()
        text = re.sub(r'[^a-zA-Z0-9. ]','',text)
        text = re.sub(r'\.','. ',text)
        text = re.sub(r'\s+',' ',text)
        return text
    
    def unique_words(self,dataset):
        for line in dataset['text']:
            for word in line.split():
                if word not in self.word_corpus:
                    self.word_corpus.add(word)
                    
        unique_words = len(self.word_corpus)

        self.inter_words = set(self.embedding_matrix.keys()).intersection(self.word_corpus)

        print("total words present {}".format(len(self.word_corpus)))
        print('total interleaving words present are {} %'.format(np.round((float(len(self.inter_words))/len(self.word_corpus))*100)))

    def num(self,text):
        unique_w = [w for w in text.split() if w not in self.inter_words]
        return len(unique_w)
    
    def summarise(self,sentence):
        sentence = sentence.reshape(1,self.max_text_len)
        encoder_out,encoder_states = model_enc(sentence)
        
        target_sentence = ''
        target_seq = np.zeros((1,1))
        target_seq[0,0] = target_word_index['start']
        stop_condition = False
        
        enc_out,states_enc = model_enc.predict(sentence)
        
        decoder_stat = states_enc
        
        while not stop_condition:
            
            word,s_h,s_c = decoder_model.predict([target_seq] + [enc_out,decoder_stat])
            
            token = np.argmax(word[0,-1,:])
            
            target_word = reverse_target_word[token]
            
            if token != 1:
                target_sentence = target_sentence + target_word + ' '
                
            if token == 1 or len(target_sentence.split()) >= (self.max_headline_len) - 1:
                stop_condition = True
                
            target_seq = np.zeros((1,1))
            target_seq[0,0] = token
            decoder_stat = [s_h,s_c]
            
        return target_sentence
    
    def seq2summary(self,input_seq):
        newString=''
        for i in input_seq:
            if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
                newString=newString+reverse_target_word[i]+' '
        return newString
    
    def seq2text(input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+reverse_input_word[i]+' '
        return newString

summariser = Abstractive_Summariser()

# loading dataset
dataset = pd.read_csv(DATASET_PATH)

# data cleaning
dataset.drop_duplicates(inplace=True)

# applying preprocessing function on text and headlines

dataset['headlines'] = dataset['headlines'].apply(summariser.preprocess)
dataset['text'] = dataset['text'].apply(summariser.preprocess)

# adding _start_ and _end_ token in the headlines.
dataset['headlines'] = dataset["headlines"].apply(lambda x: "_start_ " + x +" _end_")

# getting lengths of input and output

for i,j in zip(dataset["headlines"],dataset['text']):
    summariser.input_lengths.append(len(j.split()))
    summariser.output_lengths.append(len(i.split()))

# loading embedding matrix
with open(GLOVE_PATH,encoding="utf8") as file:
    for line in file:
        values =  line.split()
        word = values[0]
        vector = np.array(values[1:],'float32')
        summariser.embedding_matrix[word] = vector

# counting unique words from input text since the output is the 
# summary and the words would be present.

summariser.unique_words(dataset)

# Getting some more information


# unique words in each sentence

dataset['unique_words'] = dataset['text'].apply(summariser.num)

dataset = dataset[dataset['unique_words'] < 4]

dataset.reset_index(inplace=True,drop=True)

summariser.max_text_len = max(summariser.input_lengths)
summariser.max_headline_len = max(summariser.input_lengths)

# Splitting the dataset into train,test and validation set.

X_train,X_val,y_train,y_val = train_test_split(dataset['text'],dataset['headlines'],test_size=5,random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size = 0.5, random_state = 20)


# Tokenization

x_t = Tokenizer()

x_t.fit_on_texts(dataset['text'] + dataset['headlines'])

x_vocab_size = len(x_t.word_index) + 1

encoded_x_train = x_t.texts_to_sequences(X_train)
encoded_x_val = x_t.texts_to_sequences(X_val)
encoded_x_test = x_t.texts_to_sequences(X_test)

padded_X_train = pad_sequences(encoded_x_train, maxlen=summariser.max_text_len,padding='post')
paddad_X_val = pad_sequences(encoded_x_val,maxlen=summariser.max_text_len,padding='post')
padded_X_test = pad_sequences(encoded_x_test,maxlen=summariser.max_text_len,padding='post')



y_t = Tokenizer()
y_t.fit_on_texts(dataset['headlines'])
y_vocab_size = len(y_t.word_index) + 1

encoded_y_train = y_t.texts_to_sequences(y_train)
encoded_y_val = y_t.texts_to_sequences(y_val)
encoded_y_test = y_t.texts_to_sequences(y_test)

paddad_y_train = pad_sequences(encoded_y_train,maxlen=summariser.max_headline_len,padding='post')
paddad_y_test = pad_sequences(encoded_y_test,maxlen=summariser.max_headline_len,padding='post')
padded_y_val = pad_sequences(encoded_y_val,maxlen=summariser.max_headline_len,padding='post')


# Embedding matrix
embedding_dim = 300
embedding_matrix_n = np.zeros((x_vocab_size,embedding_dim),dtype=np.float32)

for word,i in x_t.word_index.items():
    embedding_vector = summariser.embedding_matrix.get(word)
    if embedding_vector is not None:
        summariser.embedding_matrix[i] = embedding_vector

embedding_matrix_m = np.zeros((x_vocab_size,embedding_dim),dtype=np.float32)

for word,i in x_t.word_index.items():
    embedding_vector = summariser.embedding_matrix.get(word)
    if embedding_vector is not None:
        embedding_matrix_m[i] = embedding_vector

encoder_input_layer = Input(shape=(summariser.max_text_len,), name="encoder_input_layer")

# The Embedding Layer

embedding_layer_input = Embedding(input_dim=x_vocab_size, output_dim=300, trainable=False, input_length=summariser.max_text_len, weights=[summariser.embedding_matrix], name="encoder_embedding_layer")(encoder_input_layer)

# print(embedding_layer_input.shape)

# encoder

encoder_lstm_1,state_h1,state_c1 = LSTM(500, return_sequences= True,return_state=True, name="encoder_lstm_1")(embedding_layer_input)
encoder_lstm_2,state_h2,state_c2 = LSTM(500, return_sequences= True,return_state=True, name="encoder_lstm_2")(encoder_lstm_1)
encoder_lstm_3,state_h3,state_c3 = LSTM(500, return_sequences= True,return_state=True, name="encoder_lstm_3")(encoder_lstm_2)

output_encoder, state_h, state_c = LSTM(500, return_state=True, return_sequences=True, name="encoder_output_layer")(encoder_lstm_3)
encoder_states = [state_h,state_c]


decoder_input_layer = Input(shape=(None,), name="decoder_input_layer")

embedding_layer_output = Embedding(input_dim=y_vocab_size, output_dim=300, trainable=False, input_length=summariser.max_text_len, weights=[summariser.embedding_matrix], name="decoder_embedding_layer")(decoder_input_layer)

# decoder


output_decoder, state_h, state_c = LSTM(500, return_state=True, return_sequences=True, name="decoder_output_layer")(embedding_layer_output,initial_state=encoder_states)

## Attention Layer
attn_layer = AttentionLayer(name = 'attention_layer')
attn_out, attn_states = attn_layer([output_encoder, output_decoder])   # based on the encoder and decoder outputs, determine which nodes are more important

# ## Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([output_decoder, attn_out])

## Dense layer
decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax'))(decoder_concat_input)

# decoder_states = [state_h,state_c]
# output_dense = Dense(len(tokens_output), activation='softmax',name="output_dense")(output_decoder)

model = Model(inputs=[encoder_input_layer,decoder_input_layer], outputs=[decoder_dense])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
checkpoint_filepath = './model.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True, save_freq = "epoch")
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
history=model.fit([padded_X_train,paddad_y_train[:,:-1]], paddad_y_train.reshape(paddad_y_train.shape[0],paddad_y_train.shape[1], 1)[:,1:] ,epochs=10,batch_size=512, validation_data=([paddad_X_val,padded_y_val[:,:-1]], padded_y_val.reshape(padded_y_val.shape[0],padded_y_val.shape[1], 1)[:,1:]))








custom_objects = {'AttentionLayer': AttentionLayer}
with keras.utils.custom_object_scope(custom_objects):
    loaded_model = load_model(r"\test_model.h5")


loaded_model.summary()

plot_model(loaded_model,show_shapes=True,show_layer_names=True)

for i, layer in enumerate(loaded_model.layers):
    print(f"Layer at index {i}: {layer.name}, Type: {type(layer)}")

input_encoder = loaded_model.layers[0].input   # taking input

embedding_inp = loaded_model.layers[1](input_encoder)

# Encoder
lstm_enc_1,h1,c1 = loaded_model.layers[2](embedding_inp) # encoder lstm 1
lstm_enc_2,h2,c2 = loaded_model.layers[3](lstm_enc_1) # encoder lstm 2
lstm_enc_3,h3,c3 = loaded_model.layers[5](lstm_enc_2) # encoder lstm 3
lstm_enc_output,h,c = loaded_model.layers[7](lstm_enc_3) # encoder output layer
encoder_states = [h,c]

model_enc = Model(input_encoder,[lstm_enc_output,encoder_states])  # final encoder model

decoder_initial_state_h = Input(shape=(500,))
decoder_initial_state_c = Input(shape=(500,))

encoder_out = Input(shape=(summariser.max_text_len,500,))

decoder_states = [decoder_initial_state_h,decoder_initial_state_c]

input_decoder = loaded_model.layers[4].input # taking input

embedding_out = loaded_model.layers[6](input_decoder)

# Decoder
lstm_dec_output,h_d,c_d = loaded_model.layers[8](embedding_out,initial_state=decoder_states) # decoder output layer

attention_output,attention_states = loaded_model.layers[9]([encoder_out,lstm_dec_output]) # implementing attention mechanism

concatenation_output = loaded_model.layers[10]([lstm_dec_output,attention_output]) # concatenation layer

time_distributed_layer = loaded_model.layers[11](concatenation_output) # time distributed layer that provides the predicted word

decoder_model = Model([input_decoder] + [encoder_out,decoder_states], [time_distributed_layer] + [h_d,c_d]) # final decoder model


reverse_target_word = y_t.index_word
reverse_input_word = x_t.index_word
target_word_index = y_t.word_index

for i in range(10):
  print("Review:",summariser.seq2text(padded_X_test[i]))
  print("Original summary:",summariser.seq2summary(paddad_y_test[i]))
  print("Predicted summary:",summariser.summarise(padded_X_test[i]))
  print("\n")