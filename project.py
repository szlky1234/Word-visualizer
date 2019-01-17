import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))



class Run:
    '''
    variables
    '''
    corpus = []
    corpusWords = []
    
    cLength = 0
    lookUpDict = {}
    cDim = 0
    batch_size = 100
    X_train = []
    Y_train = []
    data = []
    y_label = []
    oneHotDict= {}
    
    
    def __init__(self):
        #get raw text and clean into list of words without stopwords
        
        self.corpus = self.cleanInput()
        neighbours = self.generateNeighbours(self.corpus,2) #windowSize = 2
        
        
        #neighbours = self.generateNeighbor(self.corpus)
        print(neighbours)
        s = list(set(self.corpusWords)) # set of unique words without duplicates
        print(s)
        
        self.cDim = len(s)
        
        for wordIndex in range(0,len(s)):
            self.lookUpDict.update({s[wordIndex]:wordIndex})
        
        data = []
        X = []
        Y = []        
        for key in neighbours:
            data.append([self.lookUpDict[key[0]],key[0],self.toOneHot(self.lookUpDict[key[0]]),
                        self.lookUpDict[key[1]],key[1]])
            X.append(self.toOneHot(self.lookUpDict[key[0]]))
            Y.append(self.toOneHot(self.lookUpDict[key[1]])) 
            self.oneHotDict.update({key[0]:list(self.toOneHot(self.lookUpDict[key[0]]))})
        df = pd.DataFrame(data, columns = ['id','input','oneHot','id','output'])
        print(df)
        
        self.X_train = np.asarray(X)
        self.Y_train = np.asarray(Y)
        
        self.data = tf.placeholder(tf.float32, shape=(None, self.cDim))
        self.y_label = tf.placeholder(tf.float32, shape=(None, self.cDim))            

        
    '''
    generates a list of lists of size 2, containing word and its neighbour depending
    on window_size
    '''
    def cleanInput(self):
        f = open('jfk.txt')
        corpus = f.read()
        f.close()
        token = nltk.word_tokenize(corpus)
        sentenceChunk = ' '.join(token).split('.')
        truncated = []
        wnl = nltk.stem.WordNetLemmatizer()
        
        for chunk in sentenceChunk:
            part = nltk.word_tokenize(chunk)
            temp = []
            for w in part:
                word = w.lower()
                if word != ',' and word != '?' and word != '!' and word != "’" and word != "'":
                    if word not in stopwords:
                        temp.append(wnl.lemmatize(word,'n'))
                        self.corpusWords.append(wnl.lemmatize(word,'n'))
            truncated.append(temp)

        return truncated
        
    def generateNeighbours(self,alist,windowSize):
        
        neighbours = []
        
        for sentence in alist:
            for idx, word in enumerate(sentence):
                for neighbor in sentence[max(idx - windowSize, 0) : min(idx + windowSize, len(sentence)) + 1] : 
                    if neighbor != word:
                        neighbours.append([word, neighbor])            
        return neighbours
    '''
    Consumes an index and returns a one hot vector with length of corpLengthWoutDupe
    at that index
    '''
    def toOneHot(self,index):
        oneHot = np.zeros(self.cDim)
        oneHot[index] = 1
        return oneHot

    '''
    neural network with corpLengthWoutDupe inputs, one hidden layer of 2 neurons, and
    output layer of corpLengthWoutDupe outputs
    '''
    def train_network(self):
        #data = tf.placeholder(tf.float32, shape=(None, self.corpLengthWoutDupe))
        #y_label = tf.placeholder(tf.float32, shape=(None, self.corpLengthWoutDupe))        
 
        
        hidden1 = {'weights':tf.Variable(tf.random_normal([self.cDim, 2])),
                   'biases':tf.Variable(tf.random_normal([1]))}
        
        l1 = tf.add(tf.matmul(self.data, hidden1['weights']),hidden1['biases'])
        
        outputLayer = {'weights':tf.Variable(tf.random_normal([2, self.cDim])),
                   'biases':tf.Variable(tf.random_normal([1]))}  
        
        prediction = tf.nn.softmax(tf.add(tf.matmul(l1, outputLayer['weights']),outputLayer['biases']))
        
        loss = tf.reduce_mean(-tf.reduce_sum(self.y_label * tf.log(prediction), axis=[1]))
        train_op = tf.train.GradientDescentOptimizer(1.6).minimize(loss) 
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 4000
            for i in range(0,iteration):
                sess.run(train_op, feed_dict={self.data: self.X_train, self.y_label: self.Y_train})
                if i % 30 == 0:
                    print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={self.data: self.X_train, self.y_label: self.Y_train}))
            vectors = sess.run(hidden1['weights'] + hidden1['biases'])
            print(vectors)
            
            x = []
            y = []
            mylist = list(self.lookUpDict.keys())
            for coord in vectors:
                x.append(coord[0])
                y.append(coord[1])
            for x1,y1,word in zip(x,y,mylist):
                plt.annotate(word,(x1,y1))
                
            plt.plot(np.asarray(x),np.asarray(y),'g.')
            plt.show()            
        
run = Run()
run.train_network()