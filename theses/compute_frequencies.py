import pickle
import pandas as pd
from nltk.util import ngrams
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import os
from os.path import isfile,join

from preprocess import clean_doc, save_list, load_doc, process_docs, add_doc_to_vocab

#########################################################

# This script precomputes the frequencies of part-of-speach ngrams and function words,
# The script produces a csv-file with one row for each document, where the first column
# contains the identifier of the document, and the remaining columns represent the frequencies of all pos-tag unigrams, 
# the 100 most common pos-tag bigrams, the 500 most common pos-tag trigrams, and the 200 most common tokens (function words) and pronouns.

#########################################################

# csv-file containing a column named 'id', listing the identifiers of the theses to include
#included_documents = "df_swedish_downsampled.csv"
included_documents = "df_english_downsampled.csv"

# Where to save the frequencies
#frequency_file = "frequencies_swe.csv"
frequency_file = "frequencies_eng.csv"

# File path to the "chunk files" that contain the pos-tagged texts
#file_path = "/data/annl/csv/swe/"
file_path = "/data/annl/csv/eng/"

# Compute the frequencies for this part-of-speech tag

tag_kind = 'pos' # for english
#tag_kind = 'msd' # for swedish

#file_for_saving_vocabulary = "vocab_swe.txt"
file_for_saving_vocabulary = "vocab_eng.txt"

# for swedish : 
#pos_columns = ['id','word', 'baseform' , 'lemgram', 'sense',   'sentiment_label', 'sentiment_score', 'pos',  'msd',     'upos',    'ufeats']
# for english
pos_columns = ["id", "word",   "pos",     "upos",    "baseform"]


#########################################################

 # from https://www.thefreedictionary.com/List-of-pronouns.htm
pronouns_english = ["all","another","any","anybody","anyone","anything","as",
               "aught","both","each","each other","either","enough","everybody","everyone",
               "everything","few","he","her","hers","herself","him","himself","his","I","idem",
               "it","its","itself","many","me","mine","most","my","myself","naught","neither",
               "no one","nobody","none","nothing","nought","one","one another","other","others",
               "ought","our","ours","ourself","ourselves","several","she","some","somebody","someone",
               "something","somewhat","such","suchlike","that","thee","their","theirs","theirself",
               "theirselves","them","themself","themselves","there","these","they","thine","this",
               "those","thou","thy","thyself","us","we","what","whatever","whatnot","whatsoever",
               "whence","where","whereby","wherefrom","wherein","whereinto","whereof","whereon",
               "wherever","wheresoever","whereto","whereunto","wherewith","wherewithal","whether",
               "which","whichever","whichsoever","who","whoever","whom","whomever","whomso","whomsoever",
               "whose","whosever","whosesoever","whoso","whosoever","ye","yon","yonder","you","your",
               "yours","yourself","yourselves"]

pronouns_swedish = ["jag","du","han","hon","mig","dig","henne","honom","de","dem","vi","oss","ni","er","era","våra","min","din",
           "mina","dina","hennes","hans","deras","sig","den","det","varandra","tillsammans",
           "mitt","ditt","sin","sitt","sina","dess","dens","dets","någon","något","några","sådan","sådant","sådana","våran","vårt",
            "vår","mej","dej","sej","hen","dom","nån","nåra","sånt","såna","eran","ert","densamma","densamme","dessa","detta",
            "denna","vem","vad","vilken","vilka","vilket","inget","ingen","inga","varför", "hur", "när", "var","deras"
            "som", "vars","vilkas", "vars","vems","ingens","någons","alla","allas","inga","ingas","man","en","ens","envar"         
           ]   

pronouns = pronouns_english

################################################################################



def get_ngrams_by_doc(chunkfile,ids,tagtype,unigram_tokens,bigram_tokens,trigram_tokens) :
    # Compute frequencies and cleaned text for all texts in chunkfile, whose id is in ids
    fun_vocab = load_doc(file_for_saving_vocabulary)
    file = open(chunkfile, 'r')
    lines = file.readlines()
    chunks = get_the_chunks(lines,ids)
    all_frequencies = []
    all_cleandocs = []
    all_ids = []

    for i in range(len(chunks)) :

        df = pd.DataFrame([row.split("\t") for row in chunks[i]])
   
        df.columns = pos_columns 
        df = df.dropna(subset=['word'])

        bigrams_i  = list(ngrams(df[tagtype].values,2))
        trigrams_i = list(ngrams(df[tagtype].values,3))
        unigrams_i = list(df[tag_kind].values)
        bigram_frequencies_i  = [(bigrams_i.count(bigram) / len(bigrams_i)) for bigram in bigram_tokens]
        trigram_frequencies_i = [(trigrams_i.count(trigram) / len(trigrams_i)) for trigram in trigram_tokens]
        unigram_frequencies_i = [unigrams_i.count(unigram) / len(unigrams_i) for unigram in unigram_tokens]
        all_frequencies_i = ( unigram_frequencies_i + bigram_frequencies_i + trigram_frequencies_i)
        cleandoc = clean_doc(df['word'].values,fun_vocab)
        
        all_ids.append(df['id'].values[0])
        all_cleandocs.append(cleandoc)
        
        all_frequencies.append(all_frequencies_i)
      
    return(all_ids,all_frequencies,all_cleandocs)


def get_the_chunks(lines,ids) : 
    add_chunk = False
    chunk = []
    chunks = []
    for i,line in enumerate(lines) :
        if line.startswith('# text.id =') : 
            
            if chunk : 
                chunks.append(chunk)
                chunk = []
                   
            if line.startswith('# text.id = gu') : 
                theid = line.split('# text.id = ')[1][3:-1]
            else : theid = line.split('# text.id = ')[1][4:-1]
        
            if theid in [str(val) for val in ids] : 
                      
                add_chunk = True
            else : add_chunk = False
        elif line.startswith('#') and not '|' in line : ()

        elif add_chunk : chunk.append(theid + "\t" + line)
        else : ()
    if chunk : chunks.append(chunk)
    return chunks


def process_chunk(chunk_file,ids) :
    # Sums up the occurrences of ngrams of pos-tags
    file = open(chunkfile, 'r')
    lines = file.readlines()
        
    chunks = get_the_chunks(lines,ids)

    for i in range(len(chunks)) :
        print("chunk: " + str(i))
        df = pd.DataFrame([row.split("\t") for row in chunks[i]])
       
        df.columns = pos_columns 
        df = df.dropna(subset=['word'])  
        unigram_tokens_msd.update(set(df[tag_kind].values))
        bigrams_msd        = list(ngrams(df[tag_kind].values, 2)) 
        trigrams_msd       = list(ngrams(df[tag_kind].values, 3)) 
        bigram_occurrences_msd.update(bigrams_msd)
        trigram_occurrences_msd.update(trigrams_msd)
        add_doc_to_vocab(df['word'].values, vocab)
    del chunks


def flatten(listoflists) :
    return [item for sublist in listoflists for item in sublist]
    
##############################################

bigram_occurrences_msd  = Counter()
trigram_occurrences_msd = Counter()
vocab = Counter()
unigram_tokens_msd = set()

if __name__ == "__main__": 
    
    df = pd.read_csv(included_documents)
    ids = df['id'].values
    chunk_files = [join(file_path,f) for f in os.listdir(file_path) if isfile(join(file_path, f))]
    
    for chunkfile in chunk_files :
        # Sums up the occurrences of ngrams of pos-tags
        process_chunk(chunkfile,ids)
    
    # Pick the most common n-grams and tokens
    bigram_tokens_msd  = [k for k,c in bigram_occurrences_msd.most_common(100)]
    trigram_tokens_msd = [k for k,c in trigram_occurrences_msd.most_common(500)]  
    tokens = [k for k,c in vocab.most_common(200)]
   
    tokens = list(set(tokens + pronouns))

    save_list(tokens,file_for_saving_vocabulary)
    
    unigram_tokens_msd = list(unigram_tokens_msd)
    # Compute all frequencies 
    all_frequencies_ = [(get_ngrams_by_doc(chunkfile,ids,tag_kind,unigram_tokens_msd,bigram_tokens_msd,trigram_tokens_msd)) for chunkfile in chunk_files]

    all_frequencies = [freq[1] for freq in all_frequencies_] 
    # flatten list
    all_frequencies = flatten(all_frequencies)
     
    # Get the id's of the theses in the same order as the frequency listings
    ids = [freq[0] for freq in all_frequencies_]
    ids = flatten(ids)
   
    docs            = [freq[2] for freq in all_frequencies_] 
    docs = flatten(docs)
   
    del all_frequencies_
    ngram_freqs = pd.DataFrame(all_frequencies)
    ngram_freqs.columns = unigram_tokens_msd + bigram_tokens_msd + trigram_tokens_msd
    ngram_freqs['id'] = ids

    # tokenize and get frequency counts of all tokens in the cleaned documents
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    frequencies = tokenizer.texts_to_matrix(docs, mode='freq')

    fun_tokens = [a for (a,b) in tokenizer.word_index.items()]
    fun_freqs = pd.DataFrame(frequencies)
    
    fun_freqs = fun_freqs.drop(fun_freqs.columns[[0]], axis=1)
    
    fun_freqs.columns = fun_tokens
    joined = ngram_freqs.join(fun_freqs)
    
    joined.to_csv(frequency_file)
    
