from flask import Flask,request
import numpy as np
import pickle
import random
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer




intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')


lm=WordNetLemmatizer()
def bow(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[lm.lemmatize(word.lower())for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
    return np.array(bag)

def chat(question):
    print("Start talking with the bot")
    while True:
        inp=question
        res=model.predict(np.array([bow(inp,words)]))
        res_index=np.argmax(res)
        tag=classes[res_index]
        for tg in intents['intents']:
            if tg['tag']==tag:
                responses=tg['responses']
        msg=(random.choice(responses))
        print(msg)
        return msg


# def prob():
#     while True:
#         inp=input("You:",)
#         if inp.lower()=="quit":
#             break
#         res=model.predict(np.array([bow(inp,words)]))
#         op=np.argmax(res)
#         print(res[0][op])
#         if res[0][op]>0.9995064:
#             tag=classes[op]
#             for tg in intents['intents']:
#                 if tg['tag']==tag:
#                     responses=tg['responses']
#             msg1=random.choice(responses)
#             print(msg1)
#         else:
#             inp=inp.replace(" ","_")
#             # res1=wikipedia.page(wikipedia.search(inp))
#             # msg2=res1.summary
#             # my_speak(msg2)


app = Flask(__name__)
@app.route("/")
def hello_world():
    return "<p>Hey what are you doing!</p>"


@app.route('/handle_chat', methods=['POST'])
def handle_post():
    if request.method == 'POST':
        question = request.form['question']
        print(question)
        return chat(question)

