#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This carrys on the work from pre_process_articles.py that does the 
word embeeding. Now we create an list that stores the information of 
every articles in terms of their mapping from vectors to labels.
"""

def label_and_classification():
    import gensim
    import numpy as np
    import tensorflow as tf
    word_vector=gensim.models.KeyedVectors.load_word2vec_format('wv.txt')
    token_list=np.load('token_list.npy',allow_pickle=True)
    token_list=token_list.tolist()
    """Two list below contain the words' vectors of X and the one-hot encoding labels of Y"""
    X,Y=[],[]
    def see_if_token_in_label_list(token,article_label):     
        lst=[w[1] for w in article_label]
        try:
            frt_occurence=lst.index(token[1])
        except ValueError:
            Y.append([0 if i !=18 else 1 for i in range(19)])
            return 18
        Y.append([0 if i !=label_mapping[article_label[frt_occurence][4]] else 1 for i in range(19)])
        return label_mapping[article_label[frt_occurence][4]]
    
    label_list=[]
    label_mapping={'name':0,'location':1,'time':2,'contact':3,'ID':4,'profession':5,'biomarker':6,
               'family':7,'clinical_event':8,'special_skills':9,'unique_treatment':10,'account':11,
               'organization':12,'education':13,'money':14,'belonging_mark':15,'med_exam':16,'others':17,'useless':18}
    
    
    for article in token_list:
        for l in article[1]:
            for i in range(3):
                l[i]=int(l[i])
                
        for token in article[0]:
            label_list.append(see_if_token_in_label_list(token,article[1])) 
            try:
                X.append(word_vector.get_vector(token[0]))
            except TypeError:
                X.append(np.zeros(300))
                
    '''Below tryies to build an classifier for final classification'''
    number_of_split_train=int(len(X)*0.8)   
    train_X,train_Y,test_X,test_Y=X[:number_of_split_train],Y[:number_of_split_train],X[number_of_split_train:],Y[number_of_split_train:] #for banary classifier
    '''
    Since the label classes are extremely imbalanced, 
    we adjust the weights of each class by passing in the weight list to the model.
    '''
    number_of_label_class=[0 for _ in range(19)]
    for i in label_list:
        number_of_label_class[i]+=1
        
    weights_class_dict={i:len(label_list)/number_of_label_class[i] if number_of_label_class[i]!=0 else len(label_list) for i in range(19)}
    """adjust the weight of 'useless' class so that we don't value other classes too much."""
    weights_class_dict[18]=20        
# =============================================================================
#     METRICS = [
#           tf.keras.metrics.TruePositives(name='tp'),
#           tf.keras.metrics.FalsePositives(name='fp'),
#           tf.keras.metrics.TrueNegatives(name='tn'),
#           tf.keras.metrics.FalseNegatives(name='fn'), 
#           tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#           tf.keras.metrics.Precision(name='precision'),
#           tf.keras.metrics.Recall(name='recall'),
#           tf.keras.metrics.AUC(name='auc')]
# =============================================================================

    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(30,activation=tf.nn.relu,input_shape=(300,)))
    model.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(19,activation=tf.nn.softmax))
    model.compile(loss='CategoricalCrossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model.summary()
    
    EPOCHS,BATCH_SIZE=2,10
    model.fit(np.array(train_X),np.array(train_Y),epochs=EPOCHS,batch_size=BATCH_SIZE,class_weight=weights_class_dict)
    test_loss,accuracy=model.evaluate(np.array(test_X),np.array(test_Y))
    predictions=model.predict(np.array(X))
    class_preds = np.argmax(predictions, axis=-1)
    
    """see what words are not in 'useless' category"""
    lst=[]
    for article,predict in zip(token_list,class_preds):
        for token in article[0]:
            if predict!=18:
                lst.append(token[0])
    return lst


if __name__=='__main__':
    p=label_and_classification()
    


