
import pandas as pd
from nltk.util import ngrams as nltk_ngrams
from nltk import FreqDist
from collections import Counter
import gzip
import sys
import json
from pathlib import Path
from tqdm import tqdm

#########################################################

# This script precomputes the frequencies of part-of-speach ngrams and function words,
# The script produces a csv-file with one row for each document, where the first column
# contains the identifier of the document, and the remaining columns represent the frequencies of all pos-tag unigrams, 
# all pos-tag unigrams, the 100 most common pos-tag bigrams, the 500 most common pos-tag trigrams, 
# and all function words (from closed word classes).

max_tokens   = 1_000_000  # keep all tokens (from closed classes)
max_unigrams = 1_000_000  # keep all unigrams
max_bigrams  = 100
max_trigrams = 500

min_count = 10  # only keep features with this many occurrences

#########################################################

languages = "sv en".split()
word_kinds = "word baseform".split()
tag_kinds = "upos pos msd".split()

try:
    lang, word_kind, tag_kind = sys.argv[1:]
    assert lang in languages
    assert word_kind in word_kinds
    assert tag_kind in tag_kinds
except:
    sys.exit(f"Usage: python {sys.argv[0]} [{'|'.join(languages)}] [{'|'.join(word_kinds)}] [{'|'.join(tag_kinds)}]")

pos_columns_lists = {
    'sv': "word baseform lemgram sense sentiment_label sentiment_score pos msd upos ufeats".split(),
    'en': "word pos upos baseform".split(),
}

pos_columns = pos_columns_lists[lang]

# csv-file containing a column named 'id', listing the identifiers of the theses to include
included_documents = Path(f"data/theses-{lang}-downsampled.csv")

# Where to save the frequencies
frequency_file = Path(f"data/frequencies-{lang}-{word_kind}-{tag_kind}.csv.gz")
frequency_file_jsonl = Path(f"data/frequencies-{lang}-{word_kind}-{tag_kind}.jsonl.gz")

# Where to save the vocabulary
file_for_saving_vocabulary = f"data/vocab-{lang}-{word_kind}-{tag_kind}.txt"

# File path to the "chunk files" that contain the pos-tagged texts
file_path = Path(f"data/parsed-csv-{lang}/")


#########################################################

closed_tags_sets = {}
closed_tags_sets['sv'] = set("""
#upos#  #suc#
ADP     PP
AUX
CCONJ   KN
DET     DT HD HS PS
#NUM#   #RG#          #_NUM_is_closed_but_we_don't_include_it_#
PART    IE PL
PRON    HP PN
SCONJ   SN
""".split())

closed_tags_sets['en'] = set("""
#upos#  #penn-treebank#
ADP     IN RP
AUX
CCONJ   CC
DET     DT PDT PRP$ WDT WP$
#NUM#   #CD#                    #_NUM_is_closed_but_we_don't_include_it_#
PART    POS TO
PRON    EX PRP WP
SCONJ
""".split())

closed_tags = closed_tags_sets[lang]


################################################################################

null_tag = "?"

def ngrams(tokens, n):
    return ['+'.join(null_tag if x is None else x for x in ngr) for ngr in nltk_ngrams(tokens, n)]


def get_the_chunks(chunkfiles, ids): 
    for chunkfil in chunkfiles:
        id = ""
        add_chunk = False
        chunk = []
        with gzip.open(chunkfil, 'rt') as file:
            for i, line in enumerate(file):
                if line.startswith('# text.id ='): 
                    if chunk: 
                        # print(id, end=" ", flush=True)
                        yield id, chunk
                        chunk = []
                    _, id = line.split('# text.id = ')
                    _, id = id.split('-')
                    id = id.strip()
                    assert id.isdigit(), (line, id)
                    add_chunk = (id in ids)
                elif line.startswith('#') and not '|' in line: 
                    pass
                elif add_chunk:
                    chunk.append(line)
        if chunk: 
            yield id, chunk



##############################################

df = pd.read_csv(included_documents)
ids = set(str(val) for val in df['id'])

chunk_files = list(file_path.glob('*.csv.gz'))
# chunk_files = chunk_files[:5]

print("# Read chunks to tokens and n-grams")

all_chunks = []
for id, chunk in tqdm(get_the_chunks(chunk_files, ids), total=len(ids)):
    df = pd.DataFrame([row.split("\t") for row in chunk], columns=pos_columns)
    df = df.dropna(subset=[word_kind])  

    # Only keep tokens from closed categories
    tokens = [
        word.lower() 
        for word0, tag in zip(df[word_kind], df[tag_kind]) 
        if tag in closed_tags
        for word in [word0.strip('|')]
        if word.isalpha() and len(word) > 1
    ]
 
    unigrams = ngrams(df[tag_kind], 1)
    bigrams  = ngrams(df[tag_kind], 2)
    trigrams = ngrams(df[tag_kind], 3)

    all_chunks.append((id, FreqDist(tokens), FreqDist(unigrams), FreqDist(bigrams), FreqDist(trigrams)))


print("# Count token and n-gram occurrences")

word_occurrences = Counter()
unigram_occurrences = Counter()
bigram_occurrences  = Counter()
trigram_occurrences = Counter()

for id, tokens, unigrams, bigrams, trigrams in all_chunks:
    word_occurrences.update(tokens)
    unigram_occurrences.update(unigrams)
    bigram_occurrences .update(bigrams)
    trigram_occurrences.update(trigrams)


print("# Pick the most common tokens and n-grams")

def most_important(occurrences, max_n, min_c):
    selected = [k for k, c in occurrences.most_common() if c >= min_c]
    return sorted(selected[:max_n])

word_tokens = most_important(word_occurrences, max_tokens, min_count)
unigram_tokens = most_important(unigram_occurrences, max_unigrams, min_count)
bigram_tokens  = most_important(bigram_occurrences, max_bigrams, min_count)
trigram_tokens = most_important(trigram_occurrences, max_trigrams, min_count)

info = {
    'words'   : len(word_tokens), 
    'unigrams': len(unigram_tokens),
    'bigrams' : len(bigram_tokens),
    'trigrams': len(trigram_tokens),
}

with open(file_for_saving_vocabulary, 'w') as out:
    print(json.dumps(info, indent=4))

    print(json.dumps(info), file=out)
    print(file=out)

    print("# words:", len(word_tokens), file=out)
    for t in word_tokens: print(t, word_occurrences[t], file=out)
    print(file=out)
    
    print("# unigrams:", len(unigram_tokens), file=out)
    for t in unigram_tokens: print(t, unigram_occurrences[t], file=out)
    print(file=out)
    
    print("# bigrams:", len(bigram_tokens), file=out)
    for t in bigram_tokens: print(t, bigram_occurrences[t], file=out)
    print(file=out)
    
    print("# trigrams:", len(trigram_tokens), file=out)
    for t in trigram_tokens: print(t, trigram_occurrences[t], file=out)
    print(file=out)


print("# Compute token and n-gram frequencies")
all_docs = []
for id, tokens, unigrams, bigrams, trigrams in all_chunks:
    token_frequencies = [tokens.freq(tok) for tok in word_tokens]
    unigram_frequencies = [unigrams.freq(unigram) for unigram in unigram_tokens]
    bigram_frequencies  = [bigrams.freq(bigram) for bigram in bigram_tokens]
    trigram_frequencies = [trigrams.freq(trigram) for trigram in trigram_tokens]

    all_docs.append([id] + token_frequencies + unigram_frequencies + bigram_frequencies + trigram_frequencies)

df_docs = pd.DataFrame(all_docs, columns = ['ID'] + word_tokens + unigram_tokens + bigram_tokens + trigram_tokens)

df_docs.to_csv(frequency_file)
df_docs.to_json(frequency_file_jsonl, lines=True, orient="records")
