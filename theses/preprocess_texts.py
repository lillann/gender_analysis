
# turn a doc into clean tokens
def clean_doc(doc,vocab):
 tokens = [word.lower() for word in doc if word in vocab ]
 return ' '.join(tokens)


def save_list(lines, filename):
 # convert lines to a single blob of text
 data = '\n'.join(lines)
 file = open(filename, 'w')
 file.write(data)
 file.close()

 
# load doc into memory
def load_doc(filename):
 file = open(filename, 'r')
 text = file.read()
 text = text.split("\n")
 file.close()
 return text
 
def process_docs(df, vocab): 
 all_ids = df['id'].unique()     
 processed_docs = list() 
 for ident in all_ids :
     # make all tokens in document lowercase, and remove those that are not in the vocabulary
     cleandoc = clean_doc(df[df['id'] == ident]['word'].values,vocab)
     processed_docs.append(cleandoc)
 return processed_docs


def add_doc_to_vocab(doc, vocab):
 tokens = [word.lower() for word in doc if word.isalpha() and len(word) > 1 ] 
 # update counts
 vocab.update(tokens)


def add_docs(docs, vocab):
 for doc in docs : 
     if not pd.isna(doc) : 
         add_doc_to_vocab(doc, vocab)