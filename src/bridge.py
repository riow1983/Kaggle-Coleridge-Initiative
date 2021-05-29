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

#from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Process

from functools import reduce
from itertools import chain
import spacy
nlp = spacy.load("en_core_web_sm")



# class SentenceGetter(object):  
#     def __init__(self, df, train=False, use_pos=False):
#         self.n_sent = 1
#         self.df = df
#         self.empty = False
#         if train:
#             if use_pos:
#                 agg_func = lambda s: [(w,p,t) for w,p,t in zip(s["word"].values.tolist(),
#                                                             s["pos"].values.tolist(),
#                                                             s["tag"].values.tolist())]
#             else:
#                 agg_func = lambda s: [(w,t) for w,t in zip(s["word"].values.tolist(),
#                                                             s["tag"].values.tolist())]
#         else:
#             if use_pos:
#                 agg_func = lambda s: [(w,p) for w,p in zip(s["word"].values.tolist(),
#                                                             s["pos"].values.tolist())]
#             else:
#                 agg_func = lambda s: [(w,) for w in s["word"].values.tolist()]

#         self.grouped = self.df.groupby("sentence_idx").apply(agg_func)
#         self.sentences = [s for s in self.grouped]
    
#     def get_next(self):
#         try:
#             s = self.grouped["Sentence: {}".format(self.n_sent)]
#             self.n_sent += 1
#             return s
#         except:
#             return None


def get_text(filename, train=False):
    """
    Args:
        filename: str (publication Id)
    Returns:
        text: str
    """
    if train:
        df = pd.read_json(f'../input/coleridgeinitiative-show-us-the-data/train/{filename}.json')
    else:
        df = pd.read_json(f'../input/coleridgeinitiative-show-us-the-data/test/{filename}.json')

    text = " ".join(list(df['text']))
    return text


def clean_text(txt):
    """
    Args:
        txt: str
    Returns:
        txt: str
    """
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
    #return re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()








def convert_tokens(row, m, max_len, train=False, use_pos=False, verbose=False, tags_vals=None):
    """
    Args:
        row: df row
        m: df row index
        max_len: Int
        train: Bool
        use_pos: Bool
        verbose: Bool
        tags_vals: List[str]
    Returns:
        row: df row
    
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
    
    #### RIOW
    # df = pd.DataFrame()  
    # #df['token'] = list(map(str,text))
    # df['word'] = text
    # if verbose:
    #     print(list(map(str,text)))
    #     print("length of token:", len(list(map(str,text))))
    #     print(pos)
    #     print("length of pos:", len(pos))
    # if use_pos:
    #     df['pos'] = pos
    # else:
    #     df['pos'] = None
    # df['sentence'] = f'sentence{m}'
    # df['sentence#'] = sentence_hash
    # if train:
    #     df['tag'] = tokens

    # return df
    

    row["word"] = text
    if use_pos:
        row["pos"] = pos
    else:
        row["pos"] = None
    row["sentence"] = f"sentence{m}"
    row["sentence#"] = sentence_hash
    if train:
        tag2idx = {t: i for i, t in enumerate(tags_vals)}
        tokens = [tag2idx.get(t) for t in tokens]
        row["tag"] = tokens
    return row

    #### RIOWRIOW



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


def df2dataset(df, max_len, train=False, use_pos=False, verbose=False, tags_vals=None):
    """
    Args:
        df: pd.DataFrame
        max_len: Int
        use_pos: Bool
        make_cv: Bool
        verbose: Bool
        tags_vals: List[str]
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
    #### RIOW
    # dfs = Parallel(n_jobs=-1)(delayed(convert_tokens)(row,
    #                                                   i, 
    #                                                   max_len,
    #                                                   train=train,
    #                                                   use_pos=use_pos,
    #                                                   verbose=verbose) for i,row in tqdm(df.iterrows(), desc="    Converting tokens..."))
    
    with multiprocessing.Pool() as pool:
        process = [pool.apply_async(convert_tokens, (row, i, max_len, train, use_pos, verbose, tags_vals)) for i,row in df.iterrows()]
        rows = [f.get() for f in tqdm(process, desc="    Converting tokens...")]
    
    
    #### RIOWRIOW
    print("    Starting to concatenate...")
    #df = pd.concat(dfs, axis=0, ignore_index=True)
    df = pd.concat(rows, axis=1, ignore_index=True).T
    del rows
    
    # df = pd.DataFrame()
    # for _df in tqdm(dfs, desc="        Appending..."):
    #     df = df.append(_df, ignore_index=True)
    # del dfs
    
    #### RIOW
    #df = reduce(lambda x,y: pd.concat([x,y], axis=0, ignore_index=True), dfs)
    
    # print("Starting to concatenate...")
    # def fast_flatten(input_list):
    #     return list(chain.from_iterable(input_list))
    
    # COLUMN_NAMES = dfs[0].columns
    # df_dict = dict.fromkeys(COLUMN_NAMES, [])
    
    # for col in COLUMN_NAMES:
    #     extracted = (df[col] for df in dfs)
    #     # Flatten and save to df_dict
    #     df_dict[col] = fast_flatten(extracted)
    
    # df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
    # del dfs, df_dict
    #### RIOWRIOW

    #### RIOW
    # reference: https://takazawa.github.io/hobby/pandas_append_fast/
    # counter = 0
    # cols = dfs[0].columns
    # dict_tmp = {}
    # df = pd.DataFrame()
    # with tqdm(total=len(dfs), desc="        Appending to dict...") as pbar:
    #     for _df in dfs:
    #         #for _, row in _df.iteritems():
    #         #for _, row in _df.iterrows():
    #         for row in _df.itertuples():
    #             dict_tmp[counter] = row[1:]
    #             counter += 1
    #         pbar.update(1)
    # del dfs
    # #print("len(dict_tmp):", len(dict_tmp))
    # #print("last element of dict_tmp.values:", list(dict_tmp.values())[-1])
    # print("        From dict to df...")
    # df = df.from_dict(dict_tmp, orient="index", columns=cols)
    # del dict_tmp
    
    print("df.shape after concatenation:", df.shape)
    if df.shape[0]==0:
        raise ValueError("Empty df!")
    #### RIOWRIOW

    #### RIOW
    #df["sentence_idx"] = df["sentence"] + df["sentence#"]
    #### RIOWRIOW
    #dataset = dataset[["sentence", "sentence_idx", "token", "pos"]].copy()
    #dataset.rename(columns={"token":"word"}, inplace=True)

    return df


# def get_cv(dataset, num_splits=5):
#     """
#     Args:
#         dataset: pd.DataFrame
#         num_splits: Int
#     Returns:
#         folds: pd.DataFrame
#     """
#     X = dataset.index.values
#     #y = dataset["tag"].values
#     y = dataset["cleaned_label"].values
#     groups = dataset["sentence"].values

#     group_kfold = GroupKFold(n_splits=num_splits)
#     group_kfold.get_n_splits(X, y, groups)

#     #res = {}
#     folds = pd.DataFrame()
#     for i, (_, test_index) in enumerate(group_kfold.split(X, y, groups)):
#         #X_train, X_test = X[train_index], X[test_index]
#         X_test = X[test_index]
#         #y_train, y_test = y[train_index], y[test_index]
#         #X_train = dataset[dataset.index.isin(X_train)]
#         X_test = dataset[dataset.index.isin(X_test)]

#         # Concat all and save at once
#         X_test["fold"] = i+1
#         folds = pd.concat([folds, X_test], ignore_index=True)
    
#     return folds





# def sentence_extractor(dataset, tags_vals=["o", "o-dataset", "pad"], train=False, use_pos=False):
#     """
#     Args:
#         dataset: pd.DataFrame
#     Returns:
#         dataset: pd.DataFrame
#     """
#     getter = SentenceGetter(dataset, train=train, use_pos=use_pos)

#     sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
    
#     if use_pos:
#         poses = [' '.join([s[1] for s in sent]) for sent in getter.sentences]
#         if train:
#             tags = [' '.join([s[2] for s in sent]) for sent in getter.sentences]
#         else:
#             tags = None
#     else:
#         poses = None
#         if train:
#             tags = [' '.join([s[1] for s in sent]) for sent in getter.sentences]
#         else:
#             tags = None
    
#     if tags:
#         # label encoding
#         # 'pad' never appeared at this point
#         tag2idx = {t: i for i, t in enumerate(tags_vals)}
#         #tags = [[tag2idx.get(t) for t in tag] for tag in tags]
#         tags = [[tag2idx.get(t) for t in tag.split(" ")] for tag in tags]


#     dataset.drop_duplicates(subset="sentence_idx", inplace=True, ignore_index=True)
#     dataset["sentences"] = sentences
#     dataset["poses"] = poses
#     dataset["tags"] = tags

#     return dataset



def sentence_getter(dataset, use_pos=False, train=False):
    """
    Args:
        dataset: pd.DataFrame
        use_pos: bool
        train: bool
    Returns:
        sentences: List[tuple]
    """
    sentences = []
    for _,row in tqdm(dataset.iterrows(), desc="Starting to get sentences..."):
        #id = row["Id"]
        
        hashes = np.array(row["sentence#"])
        num_sentences = len(np.unique(hashes))
        
        words = np.array(row["word"])
        
        if use_pos:
            poses = np.array(row["pos"])
        else:
            poses = None

        if train:
            tags = np.array(row["tag"])
        else:
            tags = None
            ids = row["Id"]
        
        for i in range(num_sentences):
            hash = np.where(hashes==f"sentence#{i}")[0]
            if train:
                if use_pos:
                    sentences.append((words[hash], poses[hash], tags[hash]))
                else:
                    sentences.append((words[hash], poses, tags[hash]))
            else:
                if use_pos:
                    sentences.append((words[hash], poses[hash], tags, ids))
                else:
                    sentences.append((words[hash], poses, tags, ids))
    return sentences






def main(train=False, 
         max_len=290, 
         tags_vals=["o", "o-dataset", "pad"], 
         #gettext=False, 
         use_pos=False, 
         #cv=False, 
         debug=False,
         text_len=30000):
    """
    Args:
        train: Bool
        max_len: Int
        tags_vals: List[str]
        gettext: Bool
        use_pos: Bool
        cv: Bool
        debug: Bool
        text_len: Int
    Returns:
        dataset: pd.DataFrame
    """
    if train:
        #dname = "nb003-annotation-data"
        print("Reading train data (CV folds)...")
        if debug:
            df = pd.read_pickle(f"../input/kagglenb006-get-text/folds_pubcat.pkl")
            #df = df.iloc[0:500, :]
            df = df.sample(500)
        else:
            df = pd.read_pickle(f"../input/kagglenb006-get-text/folds_pubcat.pkl")

        
        # if gettext:
        #     if debug:
        #         df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv", nrows=500)
        #     else:
        #         df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv")
        #     print("Starting to get text...")
        #     df['text'] = df['Id'].apply(lambda x: get_text(x, train=train))
        # else:
        #     if debug:
        #         df = pd.read_csv("../input/kagglenb006-get-text/df_train.csv", nrows=500)
        #     else:
        #         df = pd.read_csv("../input/kagglenb006-get-text/df_train.csv")
        
        #if debug:
        #    df = df.sample(100).copy()
        
        
        print("Starting to clean text...")
        df["text"] = df["text"].apply(lambda x: clean_text(x))
        
        #print("Removing too long texts...")
        #df['length'] = df['text'].apply(lambda x: len(x.split()))
        #df = df[df['length'] < 30_000]
        #df.reset_index(drop=True, inplace=True)
        if text_len > 0:
            print("Shortening too long texts...")
            df["text"] = df["text"].apply(lambda x: " ".join(x.split()[:text_len]))
        
        


        #if cv:
        if use_pos:
            print("Starting to POS tagging...")
            #df['length'] = df['text'].apply(lambda x: len(x.split()))
            #df = df[df['length'] < 3000]  # remove too long texts (mainly for POS tagging)
            #df.reset_index(drop=True, inplace=True)

            #df["text"] = df["text"].apply(lambda x: " ".join(x.split()[:3_000]))
            df = pos_tagger(df, use_pos=use_pos)
        
        print("Starting to convert df to dataset...")
        df = df2dataset(df, max_len=max_len, train=train, use_pos=use_pos, verbose=False, tags_vals=tags_vals)
        df.to_pickle("dataset.pkl")
        print("dataset.pkl has been saved at your current working directory.")
        #sentences = sentence_getter(df, use_pos=use_pos, train=train)
        
        #print("Starting to get cv...")
        #df = get_cv(df)

        # os.makedirs(dname, exist_ok=True)
        # if use_pos:
        #     #pickle.dump(folds, open("folds.pkl", "wb"))
        #     df.to_pickle(f"{dname}/folds_pos_{max_len}.pkl")
        #     #shutil.move("folds.pkl", f"{dname}/folds.pkl")
        # else:
        #     df.to_pickle(f"{dname}/folds_nopos_{max_len}.pkl")
        # else:
        #     if use_pos:
        #         df = pd.read_pickle(f"{dname}/folds_pos_{max_len}.pkl")
        #     else:
        #         df = pd.read_pickle(f"{dname}/folds_nopos_{max_len}.pkl")
    
    else:
        # Test data
        print("Reading sample_submission.csv...")
        df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/sample_submission.csv")

        
        print("Starting to get text...")
        df['text'] = df['Id'].apply(lambda x: get_text(x, train=train))
        
        print("Starting to clean text...")
        df["text"] = df["text"].apply(lambda x: clean_text(x))
        
        if text_len > 0:
            print("Shortening too long texts...")
            df["text"] = df["text"].apply(lambda x: " ".join(x.split()[:text_len]))
        
        if use_pos:
            print("Starting to POS tagging...")
            df = pos_tagger(df, use_pos=use_pos)
        
        print("Starting to convert df to dataset...")
        df = df2dataset(df, max_len=max_len, train=train, use_pos=use_pos, verbose=False, tags_vals=tags_vals)
        sentences = sentence_getter(df, use_pos=use_pos, train=train)
        pickle.dump(sentences, open("sentences.pkl", "wb"))
        print("sentences.pkl has been saved at your current working directory.")

    
    #### RIOW
    #print("Starting sentence_extractor...")
    #df = sentence_extractor(df, tags_vals=tags_vals, train=train, use_pos=use_pos)
    #### RIOWRIOW
    
    #df.to_pickle("dataset.pkl")
    #pickle.dump(tags_vals, open("tags_vals.pkl", "wb"))
    #print("Process ends.")




if __name__ == '__main__':
    print("Usage example:")
    print()
    print("!python bridge.py {train} {max_len} {use_pos} {debug} {text_len} {tags_vals}")
    print()
    print("Args:")
    
    args = sys.argv[1:]
    print(args)
    
    train = args[0].lower() == 'true'
    max_len = int(args[1])
    #gettext = args[2].lower() == 'true'
    use_pos = args[2].lower() == 'true'
    #cv = args[4].lower() == 'true'
    debug = args[3].lower() == 'true'
    text_len = int(args[4])
    tags_vals = args[5:]
    
    print(f"train: Bool: {train}")
    print(f"max_len: Int: {max_len}")
    print(f"tags_vals: List[str]: {tags_vals}")
    #print("....type of tags_vals: ", type(tags_vals))
    #print(f"gettext: Bool: {gettext}")
    print(f"use_pos: Bool: {use_pos}")
    #print(f"cv: Bool: {cv}")
    print(f"debug: Bool: {debug}")
    print(f"text_len: Int: {text_len}")
    
    main(train=train, 
         max_len=max_len, 
         tags_vals=tags_vals, 
         #gettext=gettext, 
         use_pos=use_pos, 
         #cv=cv, 
         debug=debug, 
         text_len=text_len)
    print("Output file has been saved at your current working directory.")
