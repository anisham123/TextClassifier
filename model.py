import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from collections import Counter
from keras.layers import Embedding,Input, LSTM, Dense, Conv1D, Conv2D, MaxPool2D, MaxPooling1D, Dropout, Activation, Reshape, Concatenate, Flatten
from keras.models import Sequential, model_from_json
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils



# read the input
f = open("LabelledData.txt",'r')
Input = f.read()
Input = Input.split("\n")
Input_len = len(Input)
Input_data = {}
# seperating label and data
InputData = {}
InputLabel = {}
for j in range(0,Input_len-2):
    b = Input[j].split(",,, ")
    Input_data[j] = b
    InputData[j] = Input_data[j][0]
    InputLabel[j] = Input_data[j][1]

d = InputLabel.values()
tokenizerlabel = Tokenizer(num_words=20000)
tokenizerlabel.fit_on_texts(d)
sequenceslabel = tokenizerlabel.texts_to_sequences(d)


# As the data has imbalanced classifiers
# Iterating the dataset to make a balanced dataset

count1 = sequenceslabel.count([1])
count2 = sequenceslabel.count([2])
count3 = sequenceslabel.count([3])
count4 = sequenceslabel.count([4])
count5 = sequenceslabel.count([5])
values_count = [count1, count2, count3,count4,count5]
max_count = values_count.index(max(values_count))
a = ""
i = 1482
for k in range(1,len(tokenizerlabel.word_index)+1):
    if(values_count[max_count]/sequenceslabel.count([k]) > 1):
          c = values_count[max_count]/sequenceslabel.count([k])
          for value1,key1 in tokenizerlabel.word_index.iteritems():
             if key1 == k:
                a = value1
          for j in range(c-1):
            for n in range(len(sequenceslabel)):
               if(d[n] == a):
                    InputLabel[i] = InputLabel[n]
                    InputData[i] = InputData[n]
                    i = i+1


#tokenizing LabelData Y_train
d = InputLabel.values()
tokenizerlabel = Tokenizer(num_words=20000)
tokenizerlabel.fit_on_texts(d)
decoder = {}
decoder =  tokenizerlabel.word_index
sequenceslabel = tokenizerlabel.texts_to_sequences(d)
len(sequenceslabel)
target = list()
for i in range(len(InputLabel)):
    a = sequenceslabel[i][0]
    letter = [0 for _ in range(len(tokenizerlabel.word_index))]
    letter[a-1] = 1
    target.append(letter)


Y_train = pad_sequences(target, maxlen=len(tokenizerlabel.word_index))[:len(InputLabel)]


#Tokenizing InputData X_train

d1 = InputData.values()
tokenizerData = Tokenizer(num_words=20000)
tokenizerData.fit_on_texts(d1)
sequencesData = tokenizerData.texts_to_sequences(d1)
max_len = 0;
for i in range(len(sequencesData)):
   if (len(sequencesData[i]) > max_len):
      max_len = len(sequencesData[i])

X_train = pad_sequences(sequencesData, max_len)[:len(InputData)]





'''sequence_length = 30
vocabulary_size = 20000
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)'''
'''model.add(Embedding(20000, 128, input_length=20))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='softmax'))'''

# Built a CNN-LSTM model using Keras'''

vocabulary_size=20000
embedding_dim = 128
model = Sequential()
model.add(Embedding(vocabulary_size,embedding_dim, input_length=30))
model.add(Dropout(0.2))
model.add(Conv1D(128, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=6))
model.add(LSTM(64))
model.add(Dense(5, activation='softmax'))
'''print(model.summary())'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,Y_train,shuffle=True,batch_size=64,validation_split=0.40,epochs=10,verbose=True)

a = model.predict(X_train)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


