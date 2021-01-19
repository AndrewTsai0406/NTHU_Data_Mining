# DM_Final
-----------------------------------------------
Processes for the final project:

tokenize articles-->trains the pre_trained word2vec model-->(1)classification or (2)clustering

-----------------------------------------------

Codes:

>collect_the_keys.ipynb

(save all the labeled words as self_defined_words.txt)
	
>pre_process_articles.ipynb

(use Jieba and Gensim.word2vev to tokenized articles and save it as wv.txt and token_list.npy)

>label_and_cluster.ipynb

(This tries to cluster the selected tokens by they means among their own groups. Also, it visualizes the word2vec space for the tokens)

>label_and_classification.ipynb

(A simple classifier using Keras)

-----------------------------------------------
  
materials:
>  train_1_update.txt

>  development_1.txt
  
pre_processed_data:
>  self_defined_words.txt

>  stop_words.txt

>  token_list.npy
  
pre_trained_wv:
>  wv.txt

>  zh.bin

>  zh.bin.syn0.npy

>  zh.bin.syn1neg.npy

>  zh.tsv
  











