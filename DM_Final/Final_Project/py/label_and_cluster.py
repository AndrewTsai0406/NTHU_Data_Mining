#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:32:31 2020

@author: andrew
"""
import gensim
import numpy as np

word_vector=gensim.models.KeyedVectors.load_word2vec_format('wv.txt')
word_vec_for_classes={'name':[],'location':[],'time':[],'contact':[],'ID':[],'profession':[],'biomarker':[],
               'family':[],'clinical_event':[],'special_skills':[],'unique_treatment':[],'account':[],
               'organization':[],'education':[],'money':[],'belonging_mark':[],'med_exam':[],'others':[],'useless':[]}

token_list=np.load('token_list.npy',allow_pickle=True)
token_list=np.array(token_list.tolist())[:,1]
token_set=set()
for article in token_list:
    for token in article:
        if not token[3] in token_set:
            token_set.add(token[3])
            try:
                word_vec_for_classes[token[4]].append(word_vector.get_vector(token[3]))
            except KeyError:
                pass
word_vec_for_classes_centers={key:sum(word_vec_for_classes[key])/len(word_vec_for_classes[key]) if len(word_vec_for_classes[key]) !=0 else np.zeros(300) for key in word_vec_for_classes.keys()}












