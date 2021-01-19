#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This one is going to create a text file that contains all the labeled words
"""
import pandas 

r=open('train_1_update.txt',encoding='utf-8')
text=r.read()
r.close()

article_lists=text.split('\n--------------------\n')
del(article_lists[-1])
processed_ariticles= list(map(lambda x:[x[0],'article_id\t'+x[1]],[article.split('\narticle_id\t') for article in article_lists]))
df_list=[]

for i,text in enumerate(processed_ariticles):
    df_list.append(pandas.DataFrame([x.split('\t') for x in processed_ariticles[i][1].split('\n')]))

results=pandas.concat(df_list)
results.set_axis(results.iloc[0],inplace=True,axis=1)
results.drop(0,inplace=True)

keys=set()
for w in results['entity_text']:
    keys.add(w)
keys.discard(None)


r= open('self_defined_words.txt','w')
r.write('\n'.join(sorted(list(keys),key=len,reverse=True)))
r.close()
#count =2256 len(set)=892
