from model import model
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# prediction decoder
def decode(value):
    value = value+1
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
    for value1,key1 in tokenizerlabel.word_index.iteritems():
        if key1 == value:
            print value1

# load the model by opening the model saved in the form of json file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

# Enter input and classify the text
a = " "
while(a != "STOP"):
    Input = raw_input("Enter Input / If you want to stop the classifier enter STOP ---- ")
    if (Input == "STOP" or Input == "stop" or Input == "Stop"):
       break
    else:
        Input = np.array([Input])
        tokenizerTest = Tokenizer(num_words=20000)
        tokenizerTest.fit_on_texts(Input)
        sequencesTest = tokenizerTest.texts_to_sequences(Input)
        X_test = pad_sequences(sequencesTest, maxlen=30)[:len(Input)]
        predOutput = model.predict(X_test)
        #print(predOutput)
        preds = np.argmax(predOutput, axis=1)
        decode(preds)
