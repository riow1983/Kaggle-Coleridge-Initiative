import os
import shutilu90
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")



class SentenceGetter(object):  
    def __init__(self, df, use_pos=False):
        self.n_sent = 1
        self.df = df
        self.empty = False
        if use_pos:
            agg_func = lambda s: [(w,p,f,t) for w,p,f,t in zip(s["word"].values.tolist(),
                                                        s["pos"].values.tolist(),
                                                        s["isTrain"].values.tolist(),
                                                        s["tag"].values.tolist())]
        else:
            agg_func = lambda s: [(w,f,t) for w,f,t in zip(s["word"].values.tolist(),
                                                        s["isTrain"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.df.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None



def convert_tokens(row, m, max_len, train=False, use_pos=False, verbose=False):
	"""
    Args:
        row: df row
        m: df row index
        max_len: Int
    Returns:
        df: pd.DataFrame
    
    ex) convert_tokens(row,i, MAX_LEN)
    reference: https://www.kaggle.com/shahules/coleridge-initiative-data-to-ner-format
    """
    df = pd.DataFrame()
    if use_pos:
        text = row["tok"]
        pos = row["pos"]
    else:
        #text = x["text"].split()
        text = row['text'].replace('\uf0b7','').split()
        
    
    if train:
        entity = row['dataset_label']
        
        ## main
        tokens=[]
        k=0
        for i,x in enumerate(text):

            if k==0:
                if x==entity.split()[0]:
                    entity_len = len(entity.split())
                    if entity == ' '.join(text[i:i+entity_len]):
                        tokens.extend(['o-dataset']*len(entity.split()))
                        k = entity_len
                        #print('k updated')
                    else:
                        #print(x,'o1')
                        tokens.append('o')
                else:
                    #print(x,'o2')
                    tokens.append('o')


            k = max(0,k-1)
                
    k=0
    sentence_hash=[]
    for i in range(0,len(text), max_len):
        if verbose:
            print("Is length of text[i:i+max_len] 290?", len(text[i:i+max_len]))
        sentence_hash.extend([f'sentence#{k}']* len(text[i:i+max_len]))
        k+=1
      
    #df['token'] = list(map(str,text))
    df['word'] = text
    if verbose:
        print(list(map(str,text)))
        print("length of token:", len(list(map(str,text))))
        print(pos)
        print("length of pos:", len(pos))
    if use_pos:
        df['pos'] = pos
    else:
        df['pos'] = None
    df['sentence'] = f'sentence{m}'
    df['sentence#'] = sentence_hash
    if train:
        df['tag'] = tokens

    return df



def df2dataset(df, max_len, use_pos=False, make_cv=False, verbose=False):
    """
    Args:
    	df: pd.DataFrame
    	max_len: Int
    	use_pos: Bool
    	make_cv: Bool
    	verbose: Bool
    Returns:
    	dataset: pd.DataFrame
    """

	# POS tagging
	if use_pos:
	    tok, pos = [], []
	    bar = tqdm(total = df.shape[0])
	    for doc in nlp.pipe(df['text'].values, batch_size=50, n_process=-1):
	        if doc.is_parsed:
	            tok.append([n.text for n in doc])
	            pos.append([n.pos_ for n in doc])
	        else:
	            # We want to make sure that the lists of parsed results have the
	            # same number of entries of the original Dataframe, so add some blanks in case the parse fails
	            tok.append(None)
	            pos.append(None)
	        bar.update(1)
	    df["tok"] = tok
	    df["pos"] = pos
	else:
	    df["tok"] = None
	    df["pos"] = None

	# process
	dataset = pd.DataFrame()
	bar = tqdm(total = df.shape[0])
	for i,row in tqdm(df.iterrows()):
	    _df = convert_tokens(row,i, max_len, verbose=verbose)
	    dataset.append(_df,ignore_index=True)
	    bar.update(1)
	    
	dataset["sentence_idx"] = dataset["sentence"] + dataset["sentence#"]
	#dataset = dataset[["sentence", "sentence_idx", "token", "pos"]].copy()
	#dataset.rename(columns={"token":"word"}, inplace=True)

	return dataset


def get_cv(dataset, num_splits=5):
	"""
	Args:
		dataset: pd.DataFrame
		num_splits: Int
	Returns:
		folds: pd.DataFrame
	"""
	X = dataset.index.values
	y = dataset["tag"].values
	groups = dataset["sentence"].values

	group_kfold = GroupKFold(n_splits=num_splits)
	group_kfold.get_n_splits(X, y, groups)

	#res = {}
	folds = pd.DataFrame()
	for i, (_, test_index) in enumerate(group_kfold.split(X, y, groups)):
	    #X_train, X_test = X[train_index], X[test_index]
	    X_test = X[test_index]
	    #y_train, y_test = y[train_index], y[test_index]
	    #X_train = dataset[dataset.index.isin(X_train)]
	    X_test = dataset[dataset.index.isin(X_test)]

	    # Concat all and save at once
	    X_test["fold"] = i+1
	    folds = pd.concat([folds, X_test], ignore_index=True)
	
	return folds



def sentence_extractor(dataset):
	"""
	Args:
		dataset: pd.DataFrame
	Returns:
		dataset: pd.DataFrame
	"""
	getter = SentenceGetter(dataset)

	sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
	if use_pos:
	    poses = [' '.join([s[1] for s in sent]) for sent in getter.sentences]
	else:
	    poses = None

	dataset.drop_duplicates(subset="sentence_idx", inplace=True, ignore_index=True)
	dataset["sentences"] = sentences
	dataset["poses"] = poses

	return dataset


def main(df, train=False, cv=False):
	"""
	Args:
		df: pd.DataFrame
		train: Bool
		cv: Bool
	Returns:
		dataset: pd.DataFrame
	"""
	if train:
		dname = "nb003-annotation-data"
		if cv:
			dataset = df2dataset(df)
			folds = get_cv(dataset)
			
	        os.makedirs(dname)
	        #pickle.dump(folds, open("folds.pkl", "wb"))
	        folds.to_pickle(f"{dname}/folds.pkl")
	        #shutil.move("folds.pkl", f"{dname}/folds.pkl")
	        dataset = folds.cooy()
	    else:
	    	dataset = pd.read_pickle(f"{dname}/folds.pkl")
	else:
		dataset = df2dataset(df)

    dataset = sentence_extractor(dataset)

    return dataset


    
    





