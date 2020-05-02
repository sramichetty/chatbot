import nltk
import nltk.stem.lancaster 
import lancasterstemmer
stemmer = lancaster()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json", mode='r', encoding='utf-8') as json_data:
    data = json.load(json_data)

try:

with open("data.pickel","rb") as f:
    words,labels,training,output = pickel.oad(f)


except:

words = []
labels = []
docs = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
    
    if intent["tag"]not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(classes))]

for x,doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in words :
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = np.array(output)

with open ("data.pickel","rb") as f:
    pickel.dump((words,labels,training,output))

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0],activation= "softmax"))

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nlkt.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w == se
              bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("start talking to the bot")

    while True:
        inp = input("you: ")
        if inp.lower() == "quit"
           break
        results = model.predict([bag_of_words(inp,words)])
        results_index = num.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"];
           if tg['tag'] == tag:
               responses = tg['reponses']
        
        print(random.choice(responses))


chat()