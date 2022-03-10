import emoji
import pandas as pd
import numpy as np
from keras.models import model_from_json

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                        "1": ":baseball:",
                        "2": ":beaming_face_with_smiling_eyes:",
                        "3": ":downcast_face_with_sweat:",
                        "4": ":fork_and_knife:",
                    }

with open("services/emojifier/model.json", "r") as file:
    model = model_from_json(file.read())
model.load_weights("services/emojifier/model.h5")


embeddings = {}
with open('services/emojifier/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        try:
            values = line.split()
            word = values[0]
                #coeffs = np.asarray(values[1:])
            coeffs = np.asarray(values[1:],dtype='float32')
                #print(coeffs)
        except:
            f.__next__()
            
        embeddings[word] = (coeffs)
        #print(type(coeffs))
        #print(coeffs)
        #print(embeddings[word])
def getOutputEmbeddings(X):
        
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
                
        return embedding_matrix_output

def predict(x):
    X = pd.Series([x])
    emb_X = getOutputEmbeddings(X)
    p = np.argmax(model.predict((emb_X)), axis=-1)
    print(' '.join(X[0]))
    return (emoji.emojize(emoji_dictionary[str(p[0])]))



if __name__ =="__main__":
    print(predict("i am  sad"))

