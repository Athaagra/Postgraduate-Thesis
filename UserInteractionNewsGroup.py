from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from scipy import sparse
from scipy.sparse import csc_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
random.seed(0)
#results=[]
#for ire in range(0,100):
newsgroups_train = fetch_20newsgroups(subset='train')
pprint(list(newsgroups_train.target_names))
categories = ['alt.atheism', 'talk.religion.misc',
                'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                     categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=categories)
newsgroups = np.hstack((newsgroups_train.data,newsgroups_test.data))
    vectorizer = TfidfVectorizer()
    sc = StandardScaler(with_mean=False)
    vectors = vectorizer.fit_transform(newsgroups)
    #vectors_train=vectors_train[0:-1,0:20000]
    print(vectors.shape)
    vectorssc=sc.fit_transform(vectors)
    vectorTrain = vectorssc[0:2034, 0:-1]
    vectorTest = vectorssc[2034:3387, 0:-1]
    #newsgroups_train.target
    #vectorizer = TfidfVectorizer()
    index = np.arange(0,len(newsgroups_test.target))
    #print(len(newsgroups_train.target))
    #print(index)

    index = np.arange(0, len(catzero))
    fig = plt.figure()
    plt.figure(figsize=(18,18))

    plt.plot(index, catzero, '-', label="category one")
    plt.plot(index, catone, '-', label="category two")
    plt.plot(index, cattwo, '-', label="category three")
    plt.plot(index, catthree, '-', label="category four")
    #plt.plot(kind='bar',x='index',y='newsgroups_train.target')
    cat=[]
    for i in range(len(catzero)):
        if catzero[i]==1:
            cat.append(catzero[i],'green')
        if catzero[i]==0:
            cat.append(catzero[i],'yellow')
        else:
            cat.append(catzero[i],'red')


    #vectors_test = vectorizer.fit_transform(newsgroups_test.data)
    #vectors_test=vectors_test[0:-1,0:20000]
    #vectors_test.shape
    #print(vectors_test)
    #vectors_testsc=sc.fit_transform(vectors_test)
    #vectors_test.shape
    labels = np.unique(newsgroups_train.target)

    cat=[]
    for j in labels:
        rd=[]
        for i in range(0,30):
            x = random.randint(-1, 1)
            rd.append(np.array(x))
        cat.append(np.array(rd))
    UIA=np.array(cat)
    vectorTraan= vectorTrain.toarray()
    vectorTr= vectorTrain.toarray()

    for i in range(len(newsgroups_train.target)):
        uia =cat[newsgroups_train.target[i]]
        #print(uia)
        vectorOfd = vectorTr[i,:]
        #print(vectorOfd)
        vOd=vectorOfd
        stfofDist=np.std(vOd)
        indvOd = vOd.argsort()
        HighestTerms = indvOd[-30:len(indvOd)]
        for j in range(len(HighestTerms)):
            if uia[j] == 1:
                print(vectorTr[i,:][HighestTerms[j]])
                vectorTr[i,:][HighestTerms[j]] = vectorTr[i,:][HighestTerms[j]]+stfofDist
                print(vectorTr[i,:][HighestTerms[j]])
                #M[i, :].data[HighestTerms[j]] = vOd[HighestTerms[j]]
                #print(M[i, :].data[HighestTerms[j]])
            if uia[j] == -1:
                print(vectorTr[i,:][HighestTerms[j]])
                vectorTr[i,:][HighestTerms[j]] = vectorTr[i,:][HighestTerms[j]]-stfofDist
                print(vectorTr[i, :][HighestTerms[j]])
                #M[i, :].data[HighestTerms[j]] = vOd[HighestTerms[j]]
                #print(M[i, :].data[HighestTerms[j]])
            if uia[j] == 0:
                print(vectorTr[i,:][HighestTerms[j]])
                vectorTr[i,:][HighestTerms[j]] = vectorTr[i,:][HighestTerms[j]]*0
                print(vectorTr[i, :][HighestTerms[j]])
                #M[i, :].data[HighestTerms[j]] = vOd[HighestTerms[j]]
                #print(M[i, :].data[HighestTerms[j]])
            else:
                vectorTr[i,:][HighestTerms[j]] = vectorTr[i,:][HighestTerms[j]]

    clf = make_pipeline(StandardScaler(with_mean=False),
                        SGDClassifier(max_iter=1000, tol=1e-3))
    clfn = make_pipeline(StandardScaler(with_mean=False),
                        SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(vectorTraan,newsgroups_train.target)
    clfn.fit(vectorTr,newsgroups_train.target)

    y_pred = clf.predict(vectorTest)
    y_predn = clfn.predict(vectorTest)

    # calculate recall
    recall = recall_score(newsgroups_test.target, y_pred, average='macro')
    print('Recall: %.3f' % recall)
    # calculate prediction
    precision = precision_score(newsgroups_test.target, y_pred , average='macro')
    print('Precision: %.3f' % precision)
    #################################################
    # calculate recall
    rrecall = recall_score(newsgroups_test.target, y_predn, average='macro')
    print('Recall Re-weighting: %.3f' % recall)
    # calculate prediction
    rprecision = precision_score(newsgroups_test.target, y_predn , average='macro')
    print('Precision Re-weighting: %.3f' % precision)
    results.append([recall,precision,rrecall,rprecision,cat])
    results = np.array(results)

    atheism = np.where(newsgroups_test.target==3)

    recall=[]
    precision=[]
    rrecall=[]
    rprecision=[]
    for i in range(len(results)):
        recall.append(results[i][0])
        precision.append(results[i][1])
        rrecall.append(results[i][2])
        rprecision.append(results[i][3])
    recall = np.array(recall)
    indrec = recall.argsort()
    precision = np.array(precision)
    indpre = precision.argsort()
    rrecall = np.array(rrecall)
    indrre = rrecall.argsort()
    rprecision = np.array(rprecision)
    indrpr = rprecision.argsort()
    #fig = plt.figure()
    #plt.figure(figsize=(18,18))
    #indexofDocument = np.arange(0,len(vectorTr[1,:]))
    #plt.plot(indexofDocument, vectorTr[1,:], '-',label="tf-idf Re-weighting")
    #plt.plot(indexofDocument, vectorTraan[1,:], '-',label="tf-idf")
    #plt.title("Tf-idf and Re-weighting vector")
    #plt.xlabel("Number of words")
    #plt.ylabel("Rating of each word")
    #plt.legend(loc="upper left")
    #plt.grid()
    #plt.show()

    fig = plt.figure()
    plt.figure(figsize=(18,18))
    indexofDocument = np.arange(0,len(recall))
    plt.plot(indexofDocument, recall, '-',label="tf-idf recall")
    plt.plot(indexofDocument, precision, '-',label="tf-idf precision")
    plt.plot(indexofDocument, rrecall, '-', label="tf-idf Re-weighting recall")
    plt.plot(indexofDocument, rprecision, '-', label="tf-idf Re-weighting precision")
    plt.title("Results after 100 runs")
    plt.xlabel("Training")
    plt.ylabel("Scores")
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()










# load
clf2 = joblib.load("model.pkl")







