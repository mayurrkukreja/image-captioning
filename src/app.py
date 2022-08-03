from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2

vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

#load the trained model to classify sign
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import dump, load
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# base_model = InceptionV3(weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
# vgg_model = Model(base_model.input, base_model.layers[-2].output)

# def preprocess_img(img_path):
#     #inception v3 excepts img in 299*299
#     img = load_img(img_path, target_size = (299, 299))
#     x = img_to_array(img)
#     # Add one more dimension
#     x = np.expand_dims(x, axis = 0)
#     x = preprocess_input(x)
#     return x

# def encode(image):
#     image = preprocess_img(image)
#     vec = vgg_model.predict(image)
#     vec = np.reshape(vec, (vec.shape[1]))
#     return vec


pickle_in = open("wordtoix.pkl", "rb")
wordtoix = load(pickle_in)
pickle_in = open("ixtoword.pkl", "rb")
ixtoword = load(pickle_in)
max_length = 74

def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            e = image
            preds = model.predict([e, np.array(par_caps)])
            
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


model = load_model('new-model-1.h5')

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')


resnet = load_model('new-model.h5')

print("="*150)
print("RESNET MODEL LOADED")




app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")


    
    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    
    
    incept = resnet.predict(image).reshape(1,2048)

    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)



    return render_template('after.html', data=final)

if __name__ == "__main__":
    app.run(debug=True)


