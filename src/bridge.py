import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import pickle
import gc
import re
from joblib import Parallel, delayed
import spacy
nlp = spacy.load("en_core_web_sm")



class SentenceGetter(object):  
    def __init__(self, df, train=False, use_pos=False):
        self.n_sent = 1
        self.df = df
        self.empty = False
        if train:
            if use_pos:
                agg_func = lambda s: [(w,p,t) for w,p,t in zip(s["word"].values.tolist(),
                                                            s["pos"].values.tolist(),
                                                            s["tag"].values.tolist())]
            else:
                agg_func = lambda s: [(w,t) for w,t in zip(s["word"].values.tolist(),
                                                            s["tag"].values.tolist())]
        else:
            if use_pos:
                agg_func = lambda s: [(w,p) for w,p in zip(s["word"].values.tolist(),
                                                            s["pos"].values.tolist())]
            else:
                agg_func = lambda s: [(w,) for w in s["word"].values.tolist()]

        self.grouped = self.df.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def get_text(filename, train=False):
    if train:
        df = pd.read_json(f'../input/coleridgeinitiative-show-us-the-data/train/{filename}.json')
    else:
        df = pd.read_json(f'../input/coleridgeinitiative-show-us-the-data/test/{filename}.json')

    text = " ".join(list(df['text']))
    return text


def clean_text(txt):
    #return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()








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
            print(f"Is length of text[i:i+max_len] {max_len}?", len(text[i:i+max_len]))
        sentence_hash.extend([f'sentence#{k}']* len(text[i:i+max_len]))
        k+=1
    

    df = pd.DataFrame()  
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



def pos_tagger(df, use_pos=False):
    """
    Args:
        df: pd.DataFrame
        use_pos: Bool
    Returns:
        df: pd.DataFrame
    """
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

    return df


def df2dataset(df, max_len, train=False, use_pos=False, verbose=False):
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

    # Single process
    # dataset = pd.DataFrame()
    # bar = tqdm(total = df.shape[0])
    # for i,row in df.iterrows():
    #     _df = convert_tokens(row,i, max_len, train=train, use_pos=use_pos, verbose=verbose)
    #     dataset = dataset.append(_df,ignore_index=True)
    #     bar.update(1)

    # Parallel process
    dfs = Parallel(n_jobs=-1)(delayed(convert_tokens)(row,
                                                     i, 
                                                     max_len,
                                                     train=train,
                                                     use_pos=use_pos,
                                                     verbose=verbose) for i,row in tqdm(df.iterrows(), desc="Converting tokens..."))
    #df = pd.concat(df, axis=0, ignore_index=True)
    df = pd.DataFrame()
    for _df in tqdm(dfs, desc="Appending..."):
        df = df.append(_df, ignore_index=True)
    

    
    df["sentence_idx"] = df["sentence"] + df["sentence#"]
    #dataset = dataset[["sentence", "sentence_idx", "token", "pos"]].copy()
    #dataset.rename(columns={"token":"word"}, inplace=True)

    return df


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



def sentence_extractor(dataset, train=False, use_pos=False):
    """
    Args:
        dataset: pd.DataFrame
    Returns:
        dataset: pd.DataFrame
    """
    getter = SentenceGetter(dataset, train=train, use_pos=use_pos)

    sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
    if use_pos:
        poses = [' '.join([s[1] for s in sent]) for sent in getter.sentences]
    else:
        poses = None

    dataset.drop_duplicates(subset="sentence_idx", inplace=True, ignore_index=True)
    dataset["sentences"] = sentences
    dataset["poses"] = poses

    return dataset













def main(train=False, max_len=290, gettext=False, use_pos=False, cv=False, debug=False):
    """
    Args:
        train: Bool
        max_len: Int
        gettext: Bool
        use_pos: Bool
        cv: Bool
    Returns:
        dataset: pd.DataFrame
    """
    if train:
        dname = "nb003-annotation-data"
        print("Reading train data...")
        if gettext:
            if debug:
                df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv", nrows=500)
            else:
                df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv")
            print("Starting to get text...")
            df['text'] = df['Id'].apply(lambda x: get_text(x, train=train))
        else:
            if debug:
                df = pd.read_csv("../input/kagglenb006-get-text/df_train.csv", nrows=500)
            else:
                df = pd.read_csv("../input/kagglenb006-get-text/df_train.csv")
        
        #if debug:
        #    df = df.sample(100).copy()

        print("Starting to clean text...")
        df["text"] = df["text"].apply(lambda x: clean_text(x))
        

            
        if cv:
            if use_pos:
                print("Starting to POS tagging...")
                df['length'] = df['text'].apply(lambda x: len(x.split()))
                df = df[df['length'] < 3000]  # remove too long texts (mainly for POS tagging)
                df.reset_index(drop=True, inplace=True)
                df = pos_tagger(df, use_pos=use_pos)
            
            print("Starting to convert df to dataset...")
            df = df2dataset(df, max_len=max_len, train=train, use_pos=use_pos, verbose=False)
            
            print("Starting to get cv...")
            df = get_cv(df)

            os.makedirs(dname, exist_ok=True)
            if use_pos:
                #pickle.dump(folds, open("folds.pkl", "wb"))
                df.to_pickle(f"{dname}/folds_pos_{max_len}.pkl")
                #shutil.move("folds.pkl", f"{dname}/folds.pkl")
            else:
                df.to_pickle(f"{dname}/folds_nopos_{max_len}.pkl")
        else:
            if use_pos:
                df = pd.read_pickle(f"{dname}/folds_pos_{max_len}.pkl")
            else:
                df = pd.read_pickle(f"{dname}/folds_nopos_{max_len}.pkl")
    
    else:
        # Test data
        print("Reading sample_submission.csv...")
        df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/sample_submission.csv", nrows=100)

        
        print("Starting to get text...")
        df['text'] = df['Id'].apply(lambda x: get_text(x, train=train))
        
        print("Starting to clean text...")
        df["text"] = df["text"].apply(lambda x: clean_text(x))
        
        if use_pos:
            print("Starting to POS tagging...")
            df = pos_tagger(df, use_pos=use_pos)
        
        print("Starting to convert df to dataset...")
        df = df2dataset(df)


    print("Starting sentence_extractor...")
    df = sentence_extractor(df, train=train, use_pos=use_pos)
    
    df.to_pickle("dataset.pkl")




if __name__ == '__main__':
    print("Usage example:")
    print()
    print("!python bridge.py {train} {max_len} {gettext} {use_pos} {cv} {debug}")
    print()
    print("Args:")
    
    args = sys.argv[1:]
    print(args)
    
    train = args[0].lower() == 'true'
    max_len = int(args[1])
    gettext = args[2].lower() == 'true'
    use_pos = args[3].lower() == 'true'
    cv = args[4].lower() == 'true'
    debug = args[5].lower() == 'true'
    
    print(f"train: Bool: {train}")
    print(f"max_len: Int: {max_len}")
    print(f"gettext: Bool: {gettext}")
    print(f"use_pos: Bool: {use_pos}")
    print(f"cv: Bool: {cv}")
    print(f"debug: Bool: {debug}")
    
    main(train=train, max_len=max_len, gettext=gettext, use_pos=use_pos, cv=cv, debug=debug)
    print("Output file has been saved at ./dataset.pkl")
