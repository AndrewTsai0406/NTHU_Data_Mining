# -*- coding: utf-8 -*-
"""
With this one, it reads several files(stop_words.txt,self_defined_words.txt,train_1_update.txt) 
for the purpose of word embedding using Jieba aad Word2Vec.

Jieba segamentation---->
Word2Vec embbeding which retrains the pre-trained with our tokens
(By the way, all the labeled data are garanteed to be vectorized.)--->
The model of words' vectors are created for the classification later on.'

"""
def preprocess_articles():
    import jieba
    import gensim 
    import numpy as np
    r=open('stop_words.txt',encoding='utf-8')
    stop_words=r.read()
    r.close()
    
    jieba.load_userdict('self_defined_words.txt') 
    
    r=open('train_1_update.txt',encoding='utf-8')
    text=r.read()
    r.close()
    
    article_lists=text.split('\n--------------------\n')
    del(article_lists[-1])
    processed_ariticles= list(map(lambda x:[x[0],'\narticle_id\t'+x[1]],[article.split('\narticle_id\t') for article in article_lists]))
    
    for article in processed_ariticles:
        article[0]=article[0].replace("\n醫師：",'醫師：').replace("\n民眾：",'民眾：').replace("\n家屬：",'家屬：').replace("\n個管師：",'個管師：').replace("\n護理師：",'護理師：')
    
    token_list=[]
    for number,art in enumerate(processed_ariticles):
        token_list.append([[t for t in jieba.tokenize(art[0]) if t[0] not in stop_words]])
        label_of_current_article=[s.split() for s in art[1].split('\n')]
        del(label_of_current_article[0:2])
        del(label_of_current_article[-1])
        token_list[number].append(label_of_current_article)
    token_list=np.array(token_list)
    np.save('token_list.npy',token_list)
    
    train_list=[[w[0] for articles in token_list[:,0] for w in articles]]
    
    model=gensim.models.Word2Vec.load('zh.bin')
    model.min_count=1
    model.build_vocab(train_list, update=True)
    model.train(train_list,epochs=50,total_examples=model.corpus_count)
    model.wv.save_word2vec_format('wv.txt')
    print(f'There are now {len(model.wv.vocab)} words in the model.\nPlease feel free to try out the words in it, you will be disappointed so sure!')
word_vec_for_classes={'name':0,'location':0,'time':0,'contact':0,'ID':0,'profession':0,'biomarker':0,
               'family':0,'clinical_event':0,'special_skills':0,'unique_treatment':0,'account':0,
               'organization':0,'education':0,'money':0,'belonging_mark':0,'med_exam':0}
for w in a[:,1]:
    for i in w:
        word_vec_for_classes[i[4]]+=1
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,4,1])
langs = word_vec_for_classes.keys()
students = word_vec_for_classes.values()
ax.bar(langs,students)
plt.show()

if __name__=='__main__':
    preprocess_articles()

