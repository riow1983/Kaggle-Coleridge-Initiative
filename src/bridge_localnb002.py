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




def ner_zipper(words, tags, segments):
    """
    Args:
        words: np.array[str]
        tags: np.array[int] or None
        segments: np.array[str]
        train: bool
    Returns:
        all_zipped: List[List[List[str]]]

    Examples:
        all_zipped[:3]:
            [[['EU', 'B-ORG'],
              ['rejects', 'O'],
              ['German', 'B-MISC'],
              ['call', 'O'],
              ['to', 'O'],
              ['boycott', 'O'],
              ['British', 'B-MISC'],
              ['lamb', 'O'],
              ['.', 'O']],
             [['Peter', 'B-PER'], 
              ['Blackburn', 'I-PER']],
             [['BRUSSELS', 'B-LOC'], 
              ['1996-08-22', 'O']]]
    """
    # index2tag = {0:"o", 1:"o-dataset", 2:"pad"}
    # f = lambda x: index2tag.get(x)
    # f = np.vectorize(f)
    # tags = f(tags)

    all_zipped = []
    if tags is not None:
        for current_segment in np.unique(segments):
            pos_idx = np.where(segments==current_segment)[0]
            zipped = [[word, tag] for word, tag in zip(words[pos_idx], tags[pos_idx])] # ToDo: for pos
            all_zipped.append(zipped)
    else:
        for current_segment in np.unique(segments):
            pos_idx = np.where(segments==current_segment)[0]
            zipped = [[word] for word in words[pos_idx]] # ToDo: for pos
            all_zipped.append(zipped)

    return all_zipped




def convert_tokens(row, m, max_len, train=False, use_pos=False, verbose=False, tags_vals=None, fold=None, tvt=None):
    """
    Args:
        row: df row
        m: df row index
        max_len: int
        train: bool
        use_pos: bool
        verbose: bool
        tags_vals: List[str]
        fold: int
        tvt: str
    Returns:
        #row: df row
        #(text, pos, tag, sentence_hash, sentence): Tuple[np.array[str], np.array[str], np.array[str], np.array[str], str]
        (all_zipped_per_text, sentence): Tuple(List[List[List[str]]], str)
    
    ex) convert_tokens(row,i, MAX_LEN)
    reference: https://www.kaggle.com/shahules/coleridge-initiative-data-to-ner-format
    """
    #pub_title = row["pub_title"]

    if use_pos:
        text = row["tok"]
        pos = row["pos"]
    else:
        #text = x["text"].split()
        text = row['text'].replace('\uf0b7','').split()
        
    
    if train:
        #### RIOW
        #entity = row['dataset_label'] #str
        entities = row['dataset_label'] #List[str]
        #### RIOWRIOW
        
        ## main
        tokens=[]
        k=0
        
        for entity in entities:
            _tokens=[]
            k = 0            
            for i,x in enumerate(text):

                if k==0:
                    if x==entity.split()[0]:
                        entity_len = len(entity.split())
                        if entity == ' '.join(text[i:i+entity_len]):
                            _tokens.extend([1]*len(entity.split())) # 1 stands for "o-dataset"
                            k = entity_len
                        else:
                            _tokens.append(0) # 0 stands for "o"
                    else:
                        _tokens.append(0) # 0 stands for "o"


                k = max(0,k-1)
            tokens.append(_tokens) # List[List]

        tokens = np.array(tokens) # (number of entities) by (length of text) matrix
        tokens = np.sum(tokens, axis=0) # 1d matrix
        tokens = np.where(tokens>0, "o-dataset", "o") # 1d matrix
        tokens = tokens.tolist() # List[str]
        
                
    k=0
    sentence_hash=[]
    for i in range(0,len(text), max_len):
        if verbose:
            print(f"Is length of text[i:i+max_len] {max_len}?", len(text[i:i+max_len]))
        sentence_hash.extend([f'sentence#{k}']* len(text[i:i+max_len]))
        k+=1
    
    
    #### RIOW
    # row["word"] = text
    # if use_pos:
    #     row["pos"] = pos
    # else:
    #     row["pos"] = None
    # row["sentence"] = f"sentence{m}"
    # row["sentence#"] = sentence_hash

    if not use_pos:
        pos = None
    
    sentence = f"sentence{m}"
    #sentence = pub_title
    """ text: List[str], pos: List[str], sentence_hash: List[str], sentence: str """
    #### RIOWRIOW
    
    #### RIOW
    # if train:
    #     tag2idx = {t: i for i, t in enumerate(tags_vals)}
    #     #tokens = [tag2idx.get(t) for t in tokens]
    #     #row["tag"] = tokens
    #     tag = [tag2idx.get(t) for t in tokens]
    #### RIOWRIOW
    
    #### RIOW
    #return row
    #return (np.array(text), np.array(pos), np.array(tag), np.array(sentence_hash), sentence)
    """ text: List[str], pos: List[str], tag: List[str], sentence_hash: List[str], sentence: str """

    # if train:
    #     all_zipped_per_text = ner_zipper(np.array(text), np.array(tokens), np.array(sentence_hash)) #ToDo: for pos
    # else:
    #     all_zipped_per_text = ner_zipper(np.array(text), None, np.array(sentence_hash)) #ToDo: for pos
    # return (all_zipped_per_text, sentence)
    

    sentence = [sentence]*len(text)
    # ToDo: for pos
    #print("    start writing to a txt file...")
    if train:
        if tvt not in ["train", "valid"]:
            raise ValueError("`tvt` must be 'train' or 'valid'.")
        assert len(text) == len(tokens) == len(sentence_hash) == len(sentence)
        for w,t,h,s in zip(text, tokens, sentence_hash, sentence):
            file = open(f"./list_{tvt}_fold_{fold}.txt", mode="a")
            file.write(w+' '+t+' '+h+' '+s+'\n')
            file.close()
    else:
        assert len(text) == len(sentence_hash) == len(sentence)
        for w,h,s in zip(text, sentence_hash, sentence):
            file = open(f"./list_test.txt", mode="a")
            file.write(w+' '+h+' '+s+'\n')
            file.close()

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


#### RIOW
# def df2dataset(df, max_len, train=False, use_pos=False, verbose=False, tags_vals=None):
#     """
#     Args:
#         df: pd.DataFrame
#         max_len: Int
#         use_pos: Bool
#         make_cv: Bool
#         verbose: Bool
#         tags_vals: List[str]
#     Returns:
#         dataset: pd.DataFrame
#     """


#     with multiprocessing.Pool() as pool:
#         process = [pool.apply_async(convert_tokens, (row, i, max_len, train, use_pos, verbose, tags_vals)) for i,row in df.iterrows()]
#         rows = [f.get() for f in tqdm(process, desc="    Converting tokens...")]
    
    
#     print("    Starting to concatenate...")
#     df = pd.concat(rows, axis=1, ignore_index=True).T
#     del rows
    
    
#     print("df.shape after concatenation:", df.shape)
#     if df.shape[0]==0:
#         raise ValueError("Empty df!")

#     return df





def fold2dict(df1=None, df2=None, max_len=None, train=False, use_pos=False, verbose=False, tags_vals=None, fold=None):
    """
    Args:
        df1: pd.DataFrame
        df2: pd.DataFrame
        max_len: int
        train: bool
        use_pos: bool
        verbose: bool
        tags_vals: List[str]
        fold: int
    """
    if train:
        #### RIOW
        # #dict_per_fold = dict()
        # split_train = []
        # sentence_train = []
        # split_valid = []
        # sentence_valid = []
        
        # # train
        # # with multiprocessing.Pool() as pool:
        # #     process = [pool.apply_async(convert_tokens, (row, i, max_len, train, use_pos, verbose, tags_vals)) for i,row in df1.iterrows()]
        # #     tuples = [f.get() for f in tqdm(process, desc="    Converting train tokens...")]
        # for i,row in tqdm(df1.iterrows(), desc="    Converting train tokens..."):
        #     tpl = convert_tokens(row, i, max_len, train, use_pos, verbose, tags_vals)
        #     """ tpl: (all_zipped_per_text, sentence) """
        #     seq = tpl[0]
        #     split_train.extend(seq)
        #     sentence_train.extend([tpl[1]]*len(seq))
        # #dict_per_fold["train"] = [split_train, sentence_train]
        # out = [split_train, sentence_train]
        # del split_train, sentence_train
        # print("    saving to a pickle file...")
        # pickle.dump(out, open(f"./list_train_fold_{fold}.pkl", "wb"))
        # print(f"list_train_fold_{fold}.pkl has been saved at your current working directory.")
        # del out
        
        if os.path.exists(f"./list_train_fold_{fold}.txt"):
            os.remove(f"./list_train_fold_{fold}.txt")
        for i,row in tqdm(df1.iterrows(), desc="    Converting train tokens...", total=len(df1)):
            convert_tokens(row, i, max_len, train, use_pos, verbose, tags_vals, fold, "train")
        # Parallel(n_jobs=-1)(delayed(convert_tokens)(row, 
        #                                             i, 
        #                                             max_len, 
        #                                             train,
        #                                             use_pos,
        #                                             verbose,
        #                                             tags_vals,
        #                                             fold,
        #                                             "train") for i,row in tqdm(df1.iterrows(), 
        #                                                                        desc="    Converting train tokens...",
        #                                                                        total=len(df1)))
        print(f"list_train_fold_{fold}.txt has been saved at your current working directory.")
        #### RIOWRIOW
        
        #### RIOW
        # # valid
        # # with multiprocessing.Pool() as pool:
        # #     process = [pool.apply_async(convert_tokens, (row, i, max_len, train, use_pos, verbose, tags_vals)) for i,row in df2.iterrows()]
        # #     tuples = [f.get() for f in tqdm(process, desc="    Converting valid tokens...")]
        # for i,row in tqdm(df2.iterrows(), desc="    Converting valid tokens..."):
        #     tpl = convert_tokens(row, i, max_len, train, use_pos, verbose, tags_vals)
        #     """ tpl: (all_zipped_per_text, sentence) """
        #     seq = tpl[0]
        #     split_valid.extend(seq)
        #     sentence_valid.extend([tpl[1]]*len(seq))
        # #dict_per_fold["valid"] = [split_valid, sentence_valid]
        # out = [split_valid, sentence_valid]
        # del split_valid, sentence_valid
        # print("    saving to a pickle file...")
        # pickle.dump(out, open(f"./list_valid_fold_{fold}.pkl", "wb"))
        # print(f"list_valid_fold_{fold}.pkl has been saved at your current working directory.")
        # del out
        
        if os.path.exists(f"./list_valid_fold_{fold}.txt"):
            os.remove(f"./list_valid_fold_{fold}.txt")
        for i,row in tqdm(df2.iterrows(), desc="    Converting valid tokens...", total=len(df2)):
            convert_tokens(row, i, max_len, train, use_pos, verbose, tags_vals, fold, "valid")
        # Parallel(n_jobs=-1)(delayed(convert_tokens)(row, 
        #                                             i, 
        #                                             max_len, 
        #                                             train,
        #                                             use_pos,
        #                                             verbose,
        #                                             tags_vals,
        #                                             fold,
        #                                             "valid") for i,row in tqdm(df2.iterrows(), 
        #                                                                        desc="    Converting valid tokens...",
        #                                                                        total=len(df2)))
        print(f"list_valid_fold_{fold}.txt has been saved at your current working directory.")
        #### RIOWRIOW


        #return dict_per_fold

    else:
        #### RIOW
        # split_test = []
        # sentence_test = []
        
        # # with multiprocessing.Pool() as pool:
        # #     process = [pool.apply_async(convert_tokens, (row, i, max_len, train, use_pos, verbose, tags_vals)) for i,row in df1.iterrows()]
        # #     tuples = [f.get() for f in tqdm(process, desc="    Converting test tokens...")]
        # for i,row in tqdm(df1.iterrows(), desc="    Converting test tokens..."):
        #     tpl = convert_tokens(row, i, max_len, train, use_pos, verbose, tags_vals)
        #     """ tpl: (all_zipped_per_text, sentence) """
        #     seq = tpl[0]
        #     split_test.extend(seq)
        #     sentence_test.extend([tpl[1]]*len(seq))

        # out = [split_test, sentence_test]
        # del split_test, sentence_test
        # print("    saving to a pickle file...")
        # pickle.dump(out, open(f"./list_test.pkl", "wb"))
        # print(f"list_test.pkl has been saved at your current working directory.")
        # del out
        # #return [split_test, sentence_test]
        
        if os.path.exists(f"./list_test.txt"):
            os.remove(f"./list_test.txt")
        for i,row in tqdm(df1.iterrows(), desc="    Converting test tokens...", total=len(df1)):
            convert_tokens(row, i, max_len, train, use_pos, verbose, tags_vals, fold, "test")
        # Parallel(n_jobs=-1)(delayed(convert_tokens)(row, 
        #                                             i, 
        #                                             max_len, 
        #                                             train,
        #                                             use_pos,
        #                                             verbose,
        #                                             tags_vals,
        #                                             fold,
        #                                             "test") for i,row in tqdm(df1.iterrows(), 
        #                                                                       desc="    Converting test tokens...",
        #                                                                       total=len(df1)))
        print(f"list_test.txt has been saved at your current working directory.")
        #### RIOWRIOW



        


#### RIOWRIOW



# def sentence_getter(dataset, use_pos=False, train=False):
#     """
#     Args:
#         dataset: pd.DataFrame
#         use_pos: bool
#         train: bool
#     Returns:
#         sentences: List[tuple]
#     """
#     sentences = []
#     for _,row in tqdm(dataset.iterrows(), desc="Starting to get sentences..."):
        
#         hashes = np.array(row["sentence#"])
#         num_sentences = len(np.unique(hashes))
        
#         words = np.array(row["word"])
        
#         if use_pos:
#             poses = np.array(row["pos"])
#         else:
#             poses = None

#         if train:
#             tags = np.array(row["tag"])
#         else:
#             tags = None
#             ids = row["Id"]
        
#         for i in range(num_sentences):
#             hash = np.where(hashes==f"sentence#{i}")[0]
#             if train:
#                 if use_pos:
#                     sentences.append((words[hash], poses[hash], tags[hash]))
#                 else:
#                     sentences.append((words[hash], poses, tags[hash]))
#             else:
#                 if use_pos:
#                     sentences.append((words[hash], poses[hash], tags, ids))
#                 else:
#                     sentences.append((words[hash], poses, tags, ids))
#     return sentences






def main(fold=None,
         train=False, 
         max_len=290, 
         tags_vals=["o", "o-dataset", "pad"], 
         use_pos=False, 
         debug=False,
         text_len=30000):
    """
    Args:
        df1: pd.DataFrame
        df2: pd.Dataframe
        fold: int or None
        train: bool
        max_len: int
        tags_vals: List[str]
        use_pos: bool
        debug: bool
        text_len: Int
    """
    if train:
        print("Reading train data (CV folds)...")
        if debug:
            df = pd.read_pickle(f"../input/kagglenb006-get-text/folds_pubcat.pkl")
            df = df.sample(50)
        else:
            df = pd.read_pickle(f"../input/kagglenb006-get-text/folds_pubcat.pkl")
        
        
        print("Starting to clean text...")
        df["text"] = df["text"].apply(lambda x: clean_text(x))
        
        if text_len > 0:
            print("Shortening too long texts...")
            df["text"] = df["text"].apply(lambda x: " ".join(x.split()[:text_len]))

        if use_pos:
            print("Starting to POS tagging...")
            df = pos_tagger(df, use_pos=use_pos)
        
        #### RIOW
        # print("Starting to convert df to dataset...")
        # df = df2dataset(df, max_len=max_len, train=train, use_pos=use_pos, verbose=False, tags_vals=tags_vals)
        # df.to_pickle("dataset.pkl")
        # print("dataset.pkl has been saved at your current working directory.")

        # train valid split
        train_df = df[df["fold"]!=fold].reset_index(drop=True)
        valid_df = df[df["fold"]==fold].reset_index(drop=True)
        print(f"train_df.shape: {train_df.shape}")
        print(f"valid_df.shape: {valid_df.shape}")
        print(f"Starting to convert fold {fold} df to list of list...")
        #dict_per_fold = 
        fold2dict(df1=train_df, df2=valid_df, max_len=max_len, train=train, use_pos=use_pos, verbose=False, tags_vals=tags_vals, fold=fold)
        """ 
            [train_list, train_sentence_list]
            [valid_list, valid_sentence_list]
        """
        # if debug:
        #     pickle.dump(dict_per_fold, open(f"./debug_dict_fold_{fold}.pkl", "wb"))
        #     #pickle.dump(dict_per_fold_valid, open("./dict_per_fold_valid.pkl", "wb"))
        #     print(f"debug_dict_fold_{fold}.pkl has been saved at your current working directory.")
        # else:
        #     pickle.dump(dict_per_fold, open(f"./dict_fold_{fold}.pkl", "wb"))
        #     #pickle.dump(dict_per_fold_valid, open("./dict_per_fold_valid.pkl", "wb"))
        #     print(f"dict_fold_{fold}.pkl has been saved at your current working directory.")
        #### RIOWRIOW

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
        

        #### RIOW
        # print("Starting to convert df to dataset...")
        # df = df2dataset(df, max_len=max_len, train=train, use_pos=use_pos, verbose=False, tags_vals=tags_vals)
        # sentences = sentence_getter(df, use_pos=use_pos, train=train)
        # pickle.dump(sentences, open("sentences.pkl", "wb"))
        # print("sentences.pkl has been saved at your current working directory.")
        

        print("Starting to convert test df to list of list...")
        #test_list = 
        fold2dict(df1=df, df2=None, max_len=max_len, train=train, use_pos=use_pos, verbose=False, tags_vals=tags_vals, fold=fold)
        """ 
        [test_list, test_sentence_list]
        """
        # pickle.dump(test_list, open("./test_list.pkl", "wb"))
        # print("test_list has been saved at your current working directory.")

        #### RIOWRIOW




if __name__ == '__main__':
    print("Usage example:")
    print()
    print("!python bridge.py {fold} {train} {max_len} {use_pos} {debug} {text_len} {tags_vals}")
    print()
    print("Args:")
    
    args = sys.argv[1:]
    print(args)
    
    
    train = args[1].lower() == 'true'
    
    if train:
        fold = int(args[0])
    else:
        fold = None
    
    max_len = int(args[2])
    use_pos = args[3].lower() == 'true'
    debug = args[4].lower() == 'true'
    text_len = int(args[5])
    tags_vals = args[6:]

    print(f"fold: int: {fold}")
    print(f"train: bool: {train}")
    print(f"max_len: int: {max_len}")
    print(f"tags_vals: List[str]: {tags_vals}")
    print(f"use_pos: bool: {use_pos}")
    print(f"debug: bool: {debug}")
    print(f"text_len: int: {text_len}")
    
    main(fold=fold,
         train=train, 
         max_len=max_len, 
         tags_vals=tags_vals, 
         use_pos=use_pos, 
         debug=debug, 
         text_len=text_len)
    print("Output file has been saved at your current working directory.")
