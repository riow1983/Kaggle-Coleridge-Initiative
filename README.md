# Kaggle-Coleridge-Initiative

***
## 実験管理テーブル
|commitSHA|comment|Local CV|Public LB|
|----|----|----|----|
|89a6af4ffaee120db89b91d71be0a677823fe480|simple string matching|-|0.522|
|83d655925a96326f2e6d320c52d1b9b8c31d7c4e|text cleaned before label matching|-|0.533|
|c1a856648016fad967882676680dc901081cfcc5|batch_size=4|-|Notebook Timeout|
|e29fe7843bbdb6895cd989b15db7fb653347576e|batch_size=128|-|Notebook Timeout|
|cee078d2341e25dc7406ce1931eff4eff49c126f|batch_size=1|-|Notebook Timeout|
|89c87f757d81cfc608e5fe8b416043adf5ea4b25|max_len=512|-|Notebook Timeout|
|dceb8a5cab5d93f49a53faf32985b5fc08a94578|w/ govt|-|0.328|
|12d1b66eb4cd0fbd79e945ad57b473e28782e73e|simple string matching w/ govt|-|0.243|
|c0570361cac81e789bf70c13dbcf2e3ea5a74bd8|string or list matching w/o govt|-|0.532|
|f1e8a4643993a56a99e867eb5a92103dbe7eb411|acronym more than 2 characters added|-|0.494|
|2ba0af840e1a87ff3c1cbdce20f76221d70f86af|w/ additional govt w/o acronym|-|0.528|
|95f51e10ec0631fdba070b5c2724521f0d8a0aa0|probe_threshold = 1.1 (sanity check)|-|0.000|
|b8a25ab241a6e80d72eacbbf5d9ea83d7c7dba37|probe_threshold = 0.2|-|0.529|
|741771d71cc92ecbf492366b7c11649fb6ab6ab2|probe_threshold = 0.3|-|0.529|
|72d967eedcd3c82a6ee0c31aa128ebdc4287b506|probe_threshold = 0.4|-|0.529|
|399adfadb44e8e771b6b703e4692ea9dc766be64|probe_threshold = 0.5|-|0.000|
|32345fbecbfe2d3647f8fc8defea202057d6543b|max_len=5, epochs=1|f1=0.000|Notebook Exceeded Allowed Compute|


## My Assets
[notebook命名規則]  
- kagglenb001-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- nb001-hoge.ipynb: kagglenb001-hoge.ipynbをlocalにpullしlocalで変更を加えるもの. 番号はkagglenb001-hoge.ipynbと共通.
- localnb001-hoge.ipynb: localで新規作成されたnotebook. 
- l2knb001-hoge.ipynb: localnb001-hoge.ipynbをKaggle platformにpushしたもの. 番号はlocalnb001-hoge.ipynbと共通.

#### Code
作成したnotebook等の説明  
|name|url|input|output|status|comment|
|----|----|----|----|----|----|
|kagglenb001_transformers_test|[URL](https://www.kaggle.com/riow1983/kagglenb001-transformers-test)|-|-|廃止(使用予定なし)|huggingface transformersの簡易メソッド<br>(AutoTokenizer, AutoModelForTokenClassification)<br>を使ったNERタスク練習|
|kagglenb002_NERDA_test|[URL](https://www.kaggle.com/riow1983/kagglenb002-nerda-test)|-|-|廃止(使用予定なし)|NERDAを使ったNERタスク練習|
|kagglenb003_annotation_data|[URL](https://www.kaggle.com/riow1983/kagglenb003-annotation-data)|[NERタスク用trainデータ](https://www.kaggle.com/shahules/ner-coleridge-initiative)|-|Done|NERDAを使ったNERタスク|
|nb003-annotation-data|URL|NERタスク用trainデータ|[5 Fold CV data](https://www.kaggle.com/riow1983/nb003-annotation-data)|廃止(使用予定なし)|NERDAによるNERタスクは放擲. <br>5 Fold CV dataを作成することが目的だったがsrc/bridge.pyとnb009-cvで代替.|
|kagglenb004-transformers-ner-inference|[URL](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)|localnb001によるfine-tuned BERTモデル他|submission.csv|submission error対応中|localnb001によるfine-tuneがうまくいっていないかもしれないがひとまずsubmit挑戦中|
|kagglenb005-pytorch-BERT-for-NER|[URL](https://www.kaggle.com/riow1983/kagglenb005-pytorch-bert-for-ner)|-|fine-tuned BERT model(未作成)|停止中|公開カーネル中高スコア(LB=0.7)を記録している<br>[kaggle notebook (Coleridge: Matching + BERT NER)](https://www.kaggle.com/tungmphung/coleridge-matching-bert-ner)のtrain側. <br>EPOCHS=1でも９時間以上かかりそう. <br>Colabにpullしてnb005-pytorch-bert-for-nerとして訓練する|
|nb005-pytorch-bert-for-ner|URL|kagglenb007-get-text's output files|fine-tuned BERT model <br> [nb005-pytorch-bert-for-ner-512](https://www.kaggle.com/riow1983/nb005-pytorch-bert-for-ner-512) <br> [nb005-pytorch-bert-for-ner](https://www.kaggle.com/riow1983/nb005-pytorch-bert-for-ner)|EPOCHS>5で訓練完了<br>lossが下がらない原因調査中|epochs\>1でもlossが下がらずLB=0.700のまま|
|kagglenb006-get-text|[URL](https://www.kaggle.com/riow1983/kagglenb006-get-text)|riow1983/nb009-cv/folds_pubcat.pkl|folds_pubcat.pkl|Done|JSONファイルからパースしたtextを新規列として加える<br>Colab側で作業する際, Google Driveに置いたJSONファイルをreadする処理に時間がかかるためKaggle上で実施した|
|kagglenb007-get-text|[URL](https://www.kaggle.com/riow1983/kagglenb007-get-text)|-|train/test dataset<br>section構造をそのまま保持|Done|JSONファイルからパースしたtextを新規列として加える<br>Colab側で作業する際, Google Driveに置いたJSONファイルをreadする処理に時間がかかるためKaggle上で実施した|
|kagglenb008-pytorch-bert-for-ner-inference|[URL]()|nb005-pytorch-bert-for-ner|submission.csv|Done|[kaggle notebook (Coleridge: Matching + BERT NER)](https://www.kaggle.com/tungmphung/coleridge-matching-bert-ner)をcopyしたもの<br>|nb005-pytorch-bert-for-nerのinference側|
|localnb001-transformers-ner|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/notebooks/localnb001-transformers-ner.ipynb)|riow1983/kagglenb006-get-text/folds_pubcat.pkl|fine-tuned BERTモデル|作成中|ネット上に落ちていた[Colab notebook](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_ner.ipynb)を本コンペ用に改造したもの. <br>huggingface pre-trainedモデルのfine-tuned後の保存は成功. <br>PytorchXLAによるTPU使用. <br>fine-tuned BERTモデルはkagglenb004-transformers-ner-inferenceの入力になる.|
|l2knb001-transformers-ner|[URL](https://www.kaggle.com/riow1983/l2knb001-transformers-ner)|nb003-annotation-data (5 fold CV data)|fine-tuned BERTモデル|使用予定なし(チームシェア用)|-|
|kagglenb009-cv|[URL](https://www.kaggle.com/riow1983/kagglenb009-cv)|../input/coleridgeinitiative-show-us-the-data/train.csv|-|nb009-cvへ引き継ぎ|[issue #9](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9)に応じたCV作成ノートブック|
|nb009-cv|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/notebooks/nb009-cv.ipynb)|../input/coleridgeinitiative-show-us-the-data/train.csv|riow1983/nb009-cv/folds_pubcat.pkl|作成中|kagglenb009-cvから引き継ぎ|
|src/bridge.py|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/src/bridge.py)|riow1983/kagglenb006-get-text/folds_pubcat.pkl|./dataset.pkl|作成中|CVデータ読み込みからPyTorch Datasetクラスへの受け渡しまでのgapを埋める処理をまとめたもの|
|config/config.yml|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/config/config.yml)|-|-|作成中|nb009, src/bridge.py, localnb001, kagglenb004の各種パラメータを管理|
|kagglenb010-lb-prover|[URL](https://www.kaggle.com/riow1983/kagglenb010-lb-prover)|sample_submission.csv|submission.csv|Done|hidden testの"string matchingできる度合い"などを評価するLB probingを担当<br>ちなみに名称誤りで"prover"では無く"prober"が正しいか|
|kagglenb011-ner-conll|[URL](https://www.kaggle.com/riow1983/kagglenb011-ner-conll)|[CoNLL003 (English-version)](https://www.kaggle.com/alaakhaled/conll003-englishversion)|-|完了|NERの基本に立ち返って実装理解|
|nb011-ner-conll|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/notebooks/nb011-ner-conll.ipynb)|[CoNLL003 (English-version)](https://www.kaggle.com/alaakhaled/conll003-englishversion)|-|作成中|kagglenb011から引き継ぎ<br>BiLSTM NERからBERT NERへ移行予定<br>実装はTensorFlow|








***
## 参考資料
#### Snipets
```Javascript
// Auto click for Colab
function ClickConnect(){
  console.log("Connnect Clicked - Start"); 
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
  console.log("Connnect Clicked - End"); 
};
setInterval(ClickConnect, 60000)
```  
```Python
# PyTorch device
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```  
```Python
# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
```  
```Python
# Sequence padding
seq = np.pad(seq, (0, MAX_LEN-len(seq)), 'constant', constant_values="[PAD]")
```  
```Python
# Kaggle or Colab
import sys
if 'kaggle_web_client' in sys.modules:
    # Do something
elif 'google.colab' in sys.modules:
    # Do something
```  
```Python
# Nested Indexing
import numpy as np
text_lengths = [5, 4, 5, 6, 3, 7, 5, 5]
total_lengths = sum(text_lengths)
accum = np.add.accumulate(text_lengths)
print(accum)
# Output:
# [ 5  9 14 20 23 30 35 40]

for original_index in range(total_lengths):
    print("text_lengths:", text_lengths)
    print("original_index:", original_index)
    sentence_index = len(np.argwhere(accum <= original_index))
    print("sentence_index:", sentence_index)
    index_wrt_sentence = original_index - np.insert(accum, 0, 0)[sentence_index]
    print("index_wrt_sentence:", index_wrt_sentence)
    print("EOF")
    print()

# Output:
# text_lengths: [5, 4, 5, 6, 3, 7, 5, 5]
# original_index: 0
# sentence_index: 0
# index_wrt_sentence: 0
# EOF

# text_lengths: [5, 4, 5, 6, 3, 7, 5, 5]
# original_index: 1
# sentence_index: 0
# index_wrt_sentence: 1
# EOF

# ...
```  
```Python
# tagstring from taglist
import numpy as np
taglist = np.array(["aa", "ada", "dge"])
tagstring = "|".join(taglist).strip("|")
print(tagstring)

# Output:
# aa|ada|dge
```  
```Python
# Move a file to a new path
shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
```  
```Python
import pickle
res = []

# Dump a pickle file
pickle.dump(res, open("./res.pkl", "wb"))

# Load a pickle file
res = pickle.load(open("./res.pkl", "rb"))
```  
```Python
import yaml
with open('./hoge.yml') as file:
    hoge = yaml.load(file, Loader=yaml.FullLoader) # Loader is recommended
```  
```Python
# Pbar for a nested for loop
from tqdm import tqdm
n = 5
m = 300
with tqdm(total=n * m) as pbar:
    for i in range(n):
        for j in range(m):
            # do something, e.g. sleep
            pbar.update(1)
```  
```Python
# Lemmatizer by spaCy
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
doc = nlp('Hoge is hoging a hoge for hoges.')
print (" ".join([token.lemma_ for token in doc]))
```  
```Python
import multiprocessing
from multiprocessing import Process

def myfunc(i,j):
  return i*j

length = 1000

with multiprocessing.Pool() as pool:
    process = [pool.apply_async(myfunc, (i, j)) for j in range(length) for i in range(length)]
    outs = [f.get() for f in process]
```  
```Python
# pandas progress apply
from tqdm.notebook import tqdm
tqdm.pandas()

df = pd.DataFrame({"original": [1, 2, 3]})
def hogefunc(x):
  return x**2

df['applied'] = df['original'].progress_apply(lambda x: hogefunc(x))
```  
```Python
# pd.DataFrame fast appending
# dfs: List[pd.DataFrame]

counter = 0
cols = dfs[0].columns
dict_tmp = {}
df = pd.DataFrame()
with tqdm(total=len(dfs), desc="Appending to dict...") as pbar:
    for _df in dfs:
        for row in _df.itertuples(): # itertuples is approx. 100x faster than iterrows
            dict_tmp[counter] = row[1:]
            counter += 1
        pbar.update(1)

# The above is faster than: 
# df = pd.DataFrame()
# for _df in tqdm(dfs, desc="Appending to df..."):
#     df = df.append(_df, ignore_index=True)

# Of course, concat is the best if memory allows:
# df = pd.concat(dfs, axis=0, ignore_index=True)
```  
```Python
# hoge.txtが/path/toに存在するか否か
import os
if os.path.exists("/path/to/hoge.txt"):
  print("hoge.txt exists!")
else:
  print("hoge.txt does not exist.")
```  
```Python
# plot sentence lengths as a histogram
import matplotlib.pyplot as plt
tmp = [len(sentence) for sentence in sentences]
print(f"max length: {np.max(tmp)} \nmin length: {np.min(tmp)}")
plt.hist(tmp)
plt.xlabel('length of sentence');
```


#### Papers
|name|url|status|comment|
|----|----|----|----|
|Big Bird: Transformers for Longer Sequences|[URL](https://arxiv.org/pdf/2007.14062.pdf)|Reading|Turing completeの意味が分からん|
|Neural Architectures for Named Entity Recognition|[URL](https://arxiv.org/pdf/1603.01360.pdf)|Reading|[arXivTimesで"NER"と検索したら出てきた](https://github.com/arXivTimes/arXivTimes/issues/185)論文.<br>2016年の論文でLSTMベースのNER用モデルの提案.<br>BERT, Transformer系以外のものも見てみようという思い.<br>実装はTheano.|

#### Blogs / Qiita / etc.
|name|url|status|comment|
|----|----|----|----|
|Understanding BigBird's Block Sparse Attention|[URL](https://huggingface.co/blog/big-bird)|Untouched||
|BERT Fine-Tuning Tutorial with PyTorch|[URL](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)|Done|~~これだけ読めばhuggingface BERTのfine-tuningがマスターできてしまいそうな勢い~~<br>BERTはLMではないなど間違った解釈やuncased/casedの違いを説明できないなど|
|Pytorchでモデルの保存と読み込み|[URL](https://tzmi.hatenablog.com/entry/2020/03/05/222813)|Done|GPUで学習してCPUで読み込む場合の説明が参考になる<br>ただしTPUで学習してCPUで読み込む場合の説明はない|
|GPUで使用したoptimizerをsave & load する時の注意|[URL](https://qiita.com/Takayoshi_Makabe/items/00eea382015c9d13911f)|Done|TPUの話はない|
|PyTorch Lightning を使用してノートブック コードを整理する|[URL](https://cloud.google.com/blog/ja/products/ai-machine-learning/increase-your-productivity-using-pytorch-lightning)|Bookmarked|PyTorchの柔軟性とzen性喪失についてすごく共感<br>PyTorch Lightningいずれやりたい|
|PandasのDataFrameのappendの高速化|[URL](https://takazawa.github.io/hobby/pandas_append_fast/)|Done|df.iteritems()とdf.from_dict()を使ったdf.append()の高速化|
|Python joblibの並列処理はuWSGI環境だと動かない。uWSGI上で並列処理するには？|[URL](https://qiita.com/taai/items/15bf6acb5121ae5f5060)|Done|joblib特有のエラーを疑うきっかけとなった記事|
|os.path.join()を活用してパス結合をしてみよう！|[URL](https://www.sejuku.net/blog/64408)|Done|os.path.join()の挙動まとめ|
|Github 上の自分のコードを Kaggle Code Competition で使うのを CI で自動化|[URL](https://qiita.com/cfiken/items/a36b5742e9d26e0b4567)|Done|utility.pyをinference notebookのinputに入れるとerrorになるとか, そんなことはないということの確認で|
|ファイルをBASE64 Encodingし、notebook上で復元する|[URL](https://www.m3tech.blog/entry/2021/01/13/180000)|Done|tility.pyをinference notebookのinputに入れるとerrorになるとか, そんなことはないということの確認で|
|Inference Speed: Batch Size (1,2,4,8,16)|[URL](https://facilecode.com/inference-speed-batch-1-2-4-8/)|Done|一言"Less is Faster"|
|spaCyのCLIで文書のカテゴリ分類を学習する|[URL](https://qiita.com/kyamamoto9120/items/84d62c3b33fb77c03fbe)|Done|全てjsonファイルに格納してspaCy CLIに渡す|
|Removing Stop Words from Strings in Python|[URL](https://stackabuse.com/removing-stop-words-from-strings-in-python)|Done|各種ライブラリによるstopwords除外方法について|
|Deep Learningの学習の様子を可視化する、fastprogressがすごく良さげ|[URL](https://qiita.com/AnchorBlues/items/fd9b9bd00042337ed0e2)|Done|fastprogressを使うとtrain loop中の進捗とloss推移を簡単に可視化できる<br>nb011-ner-conllで動作確認|



#### Documentation / Tutorials / StackOverflow / etc.
|name|url|status|comment|
|----|----|----|----|
|SAVING AND LOADING MODELS|[URL](https://pytorch.org/tutorials/beginner/saving_loading_models.html)|Reading|PyTorch標準方式のモデルsave方法|
|Source code for pytorch_transformers.tokenization_bert|[URL](https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/tokenization_bert.html)|Done|bert-base-cased tokenizerをKaggle上で使用するためS3レポジトリからwget|
|Huggign Face's notebooks|[URL](https://huggingface.co/transformers/notebooks.html)|Bookmarked|-|
|Fine-tuning a model on a token classification task|[URL](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)|Done|huggingfaceによるNERタスクのチュートリアル.<br>ただしfine-tunedモデルの保存に関する実装はない<br>なお標準的なhuggingface方式では保存したいモデルはアカウントを作ってウェブレポジトリにアップロードするらしい<br>Kaggleから使える？|
|Model sharing and uploading|[URL](https://huggingface.co/transformers/model_sharing.html)|Bookmarked|huggingface方式のモデル保存方法について|
|spaCy 101: Everything you need to know|[URL](https://spacy.io/usage/spacy-101)|Bookmarked|spaCyの全体像を把握できる|
|TORCH.LOAD|[URL](https://pytorch.org/docs/stable/generated/torch.load.html)|Done|TPUで訓練したモデルをGPUで読むにはどうすればいいか書いていない|
|SAVING AND LOADING MODELS ACROSS DEVICES IN PYTORCH|[URL](https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html)|Done|TPUで訓練したモデルをCPU側に保存して, GPUでロードする際, <br>`5. Save on CPU, Load on GPU`セクションの通りにやってみたができない|
|Guidelines for assigning num_workers to DataLoader|[URL](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813)|Done|num_workersはマイナス値で設定できないとのこと<br>挙動詳細はまだよく読めてない|
|How to write boolean command line arguments with Python?|[URL](https://stackoverflow.com/questions/41006622/how-to-write-boolean-command-line-arguments-with-python)|Done|sys.argvで渡されたTrue/False (Python Boolean)<br>は"True"/"False"と文字列になってしまうが, <br>これをPython Booleanに戻す方法について|
|The YAML Format|[URL](https://symfony.com/doc/current/components/yaml/yaml_format.html#numbers)|Done|yamlファイル上での指数表記の書き方を参考にした|
|Can I add message to the tqdm progressbar?|[URL](https://stackoverflow.com/questions/37506645/can-i-add-message-to-the-tqdm-progressbar)|Done|tqdmに説明文を記載する方法|
|(spaCy) Word vectors and semantic similarity|[URL](https://spacy.io/usage/linguistic-features#vectors-similarity)|Done|spaCyによるコサイン類似度算出方法.<br>md, lgなどの大きいモデルを使用する必要があるとのこと|
|Generate a graph using Dictionary in Python|[URL](https://www.geeksforgeeks.org/generate-graph-using-dictionary-python/)|Done|Pure Pythonでgraphを作成する方法<br>作成したedgesはnetworkxに渡すことが可能|
|(NetworkX) Examining elements of a graph|[URL](https://networkx.org/documentation/stable/tutorial.html#examining-elements-of-a-graph)|Done|作成したgraphから任意の要素を抽出する方法|
|(NetworkX) networkx.algorithms.components.node_connected_component|[URL](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.node_connected_component.html#networkx.algorithms.components.node_connected_component)|Done|　任意のnodeラベルを渡して接続している要素を全て取り出すメソッド|
|What is the most efficient way to loop through dataframes with pandas?|[URL](https://stackoverflow.com/questions/7837722/what-is-the-most-efficient-way-to-loop-through-dataframes-with-pandas)|Done|itertuplesのススメ|
|DATASETS & DATALOADERS|[URL](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)|Done|Datasetのindexing方法は対象オブジェクト依存|
|What is the trade-off between batch size and number of iterations to train a neural network?|[URL](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu)|Done|train時のbatch_sizeの増減の影響を解説|
|Tutorial: Fine tuning BERT for Sentiment Analysis|[URL](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)|Bookmarked|huggingface + PyTorchでsentiment analysisをやっている例|
|Delete digits in Python (Regex)|[URL](https://stackoverflow.com/questions/817122/delete-digits-in-python-regex)|Done|正規表現で数字を除外する方法|
|(huggingface) Glossary|[URL](https://huggingface.co/transformers/glossary.html)|Bookmarked|idsからdecodeする様子や, position idsなどの各種idsの説明が集約されている|
|(huggingface) Tokenizer|[URL](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html)|Bookmarked|Tokenizerの各種メソッド, encode_plusのargumentsが集約されている|
|(huggingface) Utilities for Tokenizers|[URL](https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus)|Bookmarked|Tokenizerのサブページ(?)<br>encode_plusのargumentsが集約されている|








#### GitHub
|name|url|status|comment|
|----|----|----|----|
|how to save and load fine-tuned model?|[URL](https://github.com/huggingface/transformers/issues/7849)|Done|huggingfaceのpre-trainedモデルを<br>fine-tuningしたものをPyTorch標準方式でsaveする方法|
|Colab crashes due to tcmalloc large allocation|[URL](https://github.com/huggingface/transformers/issues/4668)|Done|不明だったエラーメッセージ`tcmalloc: large alloc`はやはりColab上のメモリーエラーを指すらしい|
|Fast Alternative to pd.concat() for row-wise concatenation|[URL](https://gist.github.com/TariqAHassan/fc77c00efef4897241f49e61ddbede9e)|Done|2018年時点の情報で, 今は必ずしもそうではないらしい|
|Multiprocessing spaCy: Can't find model 'en_model.vectors' in en_core_web_lg|[URL](https://github.com/explosion/spaCy/issues/3552)|Done|spaCyを使った処理を並列化する際は, nlp.load()を対象の関数内に記載する必要がある|
|(huggingface examples) Fine-tuning a model on a token classification task|[URL](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)|Bookmarked|huggingfaceによる各種NLPタスク例題<br>PyTorchの枠組み(Dataset, nn.module)は使われていない|


#### Kaggle Notebooks
|name|url|status|comment|
|----|----|----|----|
|Coleridge - Huggingface Question Answering|[URL](https://www.kaggle.com/jamesmcguigan/coleridge-huggingface-question-answering)|Done|QAのtoy example的なやつ. <br>結局こんな精度じゃ話にならない. <br>また事後学習する方法が分からず終い.|
|HuggingFace Tutorial; Custom PyTorch training|[URL](https://www.kaggle.com/moeinshariatnia/simple-distilbert-fine-tuning-0-84-lb)|Bookmarked|huggingfaceのpre-trainedモデルをfine-tuningするも<br>PyTorch標準のsave方式を採用している<br>らしいところは参考になる|
|HuggingFace Tutorial; Custom PyTorch training (Forked)|[URL](https://www.kaggle.com/riow1983/huggingface-tutorial-custom-pytorch-training)|Done|[HuggingFace Tutorial; Custom PyTorch training](https://www.kaggle.com/moeinshariatnia/simple-distilbert-fine-tuning-0-84-lb)をforkしてBERTの`tokenizer.convert_ids_to_tokens(ids)`を試したもの<br>元がdocument classificationタスクなのでtokenごとのtagが無く, そこは参考にならず|
|Bert PyTorch HuggingFace Starter|[URL](https://www.kaggle.com/theoviel/bert-pytorch-huggingface-starter)|Bookmarked|huggignface PyTorchのとても綺麗なコード.<br>参考になるがfine-tuned modelのsave実装はない.|
|[Training] PyTorch-TPU-8-Cores (Ver.21)|[URL](https://www.kaggle.com/joshi98kishan/foldtraining-pytorch-tpu-8-cores/data?scriptVersionId=48061653)|Bookmarked|offlineでPyTorch-XLAインストールスクリプトが有用|
|EDA & Baseline Model|[URL](https://www.kaggle.com/prashansdixit/coleridge-initiative-eda-baseline-model)|Done|dataset_label, dataset_title, cleaned_labelをsetにして<br>existing_labelsにしている|
|data_preparation_ner|[URL](https://www.kaggle.com/shahules/coleridge-initiative-data-to-ner-format)|Done|[shahules/ner-coleridge-initiative](https://www.kaggle.com/shahules/ner-coleridge-initiative)作成コード|
|TPU|[URL](https://www.kaggle.com/docs/tpu)|Bookmarked|Kaggle Platform上でのTPUの使い方 (Kaggle公式)|
|Bert for Question Answering Baseline: Training|[URL](https://www.kaggle.com/theoviel/bert-for-question-answering-baseline-training)|Reading|BERT Q&Aタスク (train)|
|Bert for Question Answering Baseline: Inference|[URL](https://www.kaggle.com/theoviel/bert-for-question-answering-baseline-inference)|Reading|BERT Q&Atタスク (inference)|
|score 57ish with additional govt datasets|[URL](https://www.kaggle.com/mlconsult/score-57ish-with-additional-govt-datasets/data)|Reading|Best score notebook (as of 11 May)<br>外部データgovt datasetを使用しているがPrivate Datasetになっている|
|The Ultimate PyTorch+TPU Tutorial (Jigsaw XLM-R)|[URL](https://www.kaggle.com/tanlikesmath/the-ultimate-pytorch-tpu-tutorial-jigsaw-xlm-r#Running-PyTorch-model-training-on-8-core-TPUs)|Bookmarked|PyTorch + TPUの参考程度|
|imet-2019-submission|[URL](https://www.kaggle.com/lopuhin/imet-2019-submission)|Bookmarked|utility.py(オレオレPythonファイル)をKaggle Datasetを経由することなく<br>Kaggle Notebookで呼び出すためbs4を使用した恐らく最初の例|


#### Kaggle Datasets
|name|url|status|comment|
|----|----|----|----|
|shahules/ner-coleridge-initiative|[URL](https://www.kaggle.com/shahules/ner-coleridge-initiative)|Bookmarked|NERタスク用のデータセット<br>[ディスカッション](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/230341)に作成コードが紹介されている|
|joshi98kishan/pytorch-xla-setup-script|[URL](https://www.kaggle.com/joshi98kishan/pytorch-xla-setup-script)|Bookmarked|PyTorch-XLAをofflineでインストールするためのスクリプト|
|bigger_govt_dataset_list|[URL](https://www.kaggle.com/mlconsult/bigger-govt-dataset-list)|Adopted|いわゆるgovt<br>世にあるデータセットの蒐集|
|Coleridge additional_gov_datasets_22000popular|[URL](https://www.kaggle.com/chienhsianghung/coleridge-additional-gov-datasets-22000popular)|Adopted|改良版govt<br>詳細は[Discussion](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/241592)で|
|CoNLL003 (English-version)|[URL](https://www.kaggle.com/alaakhaled/conll003-englishversion)|確認中|BERT NERタスクの基礎トレーニングで使用可能か|

#### Kaggle Discussion
|name|url|status|comment|
|----|----|----|----|
|Data preparation for NER|[URL](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/230341)|Done|Dataset作成コードとTrainデータの実際のデータセット[NER Coleridge Initiative](https://www.kaggle.com/shahules/ner-coleridge-initiative)が<br>Kaggle Datasetにアップされている|
|What is "the mention of datasets"|[URL](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/235297)|Done|各データセットに複数のラベル(省略形~完全形まで)が想定されるが, <br>どこまで想定すれば良いか基準が無い点について分かりやすく指摘|
|Local CV is probably meaningless?|[URL](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/235026)|Done|trainのラベルが不完全なのでlocal CVは信頼できないのでは?という指摘<br>CV-LB相関の把握にはそれでも有用か|
|Separate IDs exist for the same pub_title|[URL](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/240384)|Done|同じpublication titleなのに, puclication IDが異なる場合があるとのこと.<br>CVを切るときはtitleを使うべきか|




***
## Diary

#### 2021-04-15  
実験管理方法のスライド作成し共有.
<br>
<br>
<br>

#### 2021-04-16
どうやらQA系では精度がでないらしい. NER系は精度でていそう.  
ひとまず学習済みBERTをNERタスクで事後学習させる方法を確立したい.
<br>
<br>
<br>

#### 2021-04-20
Google Colab ProおよびGoogle Drive strage+185GB課金した.  
課金した理由:  
- Colab無料版ではRAMが足りない局面に多く遭遇しそうだった
- Google Drive無料版のストレージ(15GB)では中間ファイルの吐き出しですぐ満杯になる  

ところでColabのセッション切れに対応する裏技としてChromeのデベロッパーツールのコンソールに  
定期的にconnectボタンをクリックするJavascriptを入力しておくというものがあり試してみた.  
[reference](https://flat-kids.net/2020/07/28/google-colab-%E3%82%BB%E3%83%83%E3%82%B7%E3%83%A7%E3%83%B3%E5%88%87%E3%82%8C%E3%82%92%E9%98%B2%E6%AD%A2%E3%81%99%E3%82%8B/)    
もしくは  
[reference](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82#8c1e)  
<br>
2021-05-02追記) 上記参照コードでは実効性が無かった. 修正版は以下:  
```Javascript
function ClickConnect(){
  console.log("Connnect Clicked - Start"); 
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
  console.log("Connnect Clicked - End"); 
};
setInterval(ClickConnect, 60000)
```  
[reference](https://www.it-swarm-ja.com/ja/python/google-colab%E3%81%8C%E5%88%87%E6%96%AD%E3%81%97%E3%81%AA%E3%81%84%E3%82%88%E3%81%86%E3%81%AB%E3%81%99%E3%82%8B%E3%81%AB%E3%81%AF%E3%81%A9%E3%81%86%E3%81%99%E3%82%8C%E3%81%B0%E3%82%88%E3%81%84%E3%81%A7%E3%81%99%E3%81%8B%EF%BC%9F/810821538/)    
デメリットもありそうだが今のところ大きな問題には遭遇していない. ~セッション切れがあまりない(と言われている)Colab Proでも必要かどうかは微妙.~ Colab Proでも一定時間操作していないとセッションが切れるので多いに必要. なおこれと合わせてPCの自動sleep機能の解除も必要. 参考: [Macノートブックのスリープ/スリープ解除の設定を指定する](https://support.apple.com/ja-jp/guide/mac-help/mchle41a6ccd/mac)  
長時間学習する時などには有効かも.  

それからKaggle APIの使い方についてKaggle Datasetsへのアップロード方法について学びがあった.    
手順としては    
(1) ファイル保存用のフォルダを作成  
(2) フォルダにデータファイルを保存  
(3) フォルダに対してinitする(これでmetaファイルが作成される)  
(4) metaファイルを編集  
(5) フォルダごとアップロード  
詳細はこちら:    
[reference](https://kaeru-nantoka.hatenablog.com/entry/2020/01/17/015551)  
<br>
<br>
<br> 

#### 2021-04-21  
NERの事後学習(fine-tuning)を簡単に実装できるNERDAというPythonライブラリがあったので触り出す.    
実装はできそうだ.  
なお学習データは  
https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/230341  
で作成されたものを利用。seqlen=290になっている根拠は分からず.  
このdf形式のものをNERDAのデータ形式に合わせる.  
```
NERDA_data = {"sentences": [[], [], ..., []], 
              "tags": [[], [], ...,[]]}
```  

ところでV100が引けた. Colab Proにした甲斐があった.  
```
Wed Apr 21 00:48:32 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.67       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   33C    P0    24W / 300W |      2MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
<br>
<br>
<br>

#### 2021-04-22
BERT Uncased / BERT Cased の違いについて  
> In BERT uncased, the text has been lowercased before WordPiece tokenization step while in BERT cased, the text is same as the input text (no changes).

> For example, if the input is "OpenGenus", then it is converted to "opengenus" for BERT uncased while BERT cased takes in "OpenGenus".  
https://iq.opengenus.org/bert-cased-vs-bert-uncased/  

大文字と小文字の区別のことをletter caseというが, これを考慮するのがcased, 考慮しないのがuncasedだと覚えると良い. 
Qiitaに逆の解説が上がっていたが, 間違いだと思う. (コメントしておいた.)  
https://qiita.com/161abcd/items/c73af4fd422f664b3bf6  

今回のコンペの元データは当然大文字と小文字両方出現するし, 固有表現は得てして大文字で始まる場合が多いので,   
BERTなどのモデルもcased一択で良いと思う.
<br>
<br>
<br>

#### 2021-04-23
notebooks/localnb001-transformers-ner.ipynbをColab Proで実行しfine-tuned BERTモデルを  
[localnb001-transformers-ner](https://www.kaggle.com/riow1983/localnb001-transformers-ner)にアップロードした.  
なお, この学習済みモデルでinferenceするためのsample-submission.csvのテーブルデータの加工についての実装はまだできていない.  
そこはフロムスクラッチするよりも公開カーネルを利用できないものかとも思っている.  
と思ったが, そのような公開カーネルは今のところなさそうだったので, 自分で実装することにした.    
それにしてもColab Pro使いやすい. ネットワークが切れても途中から処理がresumeされるので環境要因に対してもrobustな印象. High memory RAMも35GBの強いやつを引くときもあり. これで環境構築の手間やconflictを気にするストレスを大幅に削減できるのはありがたい. 
<br>
<br>
<br>

#### 2021-04-24
huggingfaceのpre-trainedモデルをfine-tuningするところまではできるが, save方式がPyTorch標準方式とhuggingface独自方式とで整理がつかず混乱中.  今のところsaveしたバイナリファイルをKaggle notebookでloadすることに成功していない.  可能であればPyTorch標準方式で一本化したいが.  
ちなみにhuggingface方式は以下のようにsaveしたファイルのデフォルトのファイル名をload前に変更しておく必要があるという糞仕様:  
>```tokenizer = BertTokenizer.from_pretrained("/home/liping/liping/bert/")```
The following files must be located in that folder:
```
vocab.txt - vocabulary file
pytorch_model.bin - the PyTorch-compatible (and converted) model
config.json - json-based model configuration
Please make sure that these files exist and e.g. rename bert-base-cased-pytorch_model.bin to pytorch_model.bin.
```  
https://www.gitmemory.com/issue/huggingface/transformers/1620/545961654
  
以下は[Kaggle Notebook](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)から  
![input file image](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/png/Screenshot%202021-04-25%20at%207.16.04.png?raw=true)  
を読み込もうとした際に遭遇するエラー. 格納されているconfigファイルの名称が`bert_config.json`であるのに対し, `config.json`を要求している.
```
OSError: Can't load config for '../input/localnb001-transformers-ner/bert-base-cased'. Make sure that:

- '../input/localnb001-transformers-ner/bert-base-cased' is a correct model identifier listed on 'https://huggingface.co/models'

- or '../input/localnb001-transformers-ner/bert-base-cased' is the correct path to a directory containing a config.json file
```
なお読み込み方法はhuggingfaceの`.from_pretrained`メソッドを使っている.
```Python
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertForTokenClassification.from_pretrained(f'../input/localnb001-transformers-ner/bert-base-cased')
    
    def forward(self, ids, mask, labels):
        output_1= self.l1(ids, mask, labels = labels)
        return output_1
```
<br>
<br>
<br>

#### 2021-04-25
huggingfaceをPyTorch nn.Moduleで訓練した後どのようにしてモデルをsaveすればいいかについて同じ質問が[huggingfaceのissue](https://github.com/huggingface/transformers/issues/7849)に上がっていた.  
以下のコードで良いらしい.  
```Python
model = MyModel(num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
output_model = './models/model_xlnet_mid.pth'

# save
def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

save(model, optimizer)

# load
checkpoint = torch.load(output_model, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```  
最初にやっていたものがこの方式の一部だった. というのは`optimizer_state_dict`は記載していなかった.    
紆余曲折あったが[riow1983/kagglenb004-transformers-ner-inference](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)でloadからpredictまでエラーに遭遇することなくできた模様. predict結果のサニティチェックはまだできていない. tokenizerのロードについてはhuggingfaceデフォルトのtokenizer(`../input/d/riow1983/localnb001-transformers-ner/bert-base-cased-vocab.txt`)を使用しているが問題ないのか不明.
```Python
# Defining some key variables that will be used later on in the training
CV = 1
MAX_LEN = 200
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained('../input/d/riow1983/localnb001-transformers-ner/bert-base-cased-vocab.txt')
```
なお, inputの一部フォルダパスのparentが`../input/`から`../input/d/riow1983/`に変更されてしまっていてそれに気づくまで時間を消費した. 謎.
<br>
<br>
<br>

#### 2021-04-26
[riow1983/kagglenb004-transformers-ner-inference](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)にて予測結果を確認すると, 全て'o'タグだったため[localnb001](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/notebooks/localnb001-transformers-ner.ipynb)のEPOCHS数を1から5に変更して再挑戦してみる. MAX_LENは200から290に変更した. ([訓練用データセット](https://www.kaggle.com/shahules/ner-coleridge-initiative)の固定長が290だったため.)  
[transformers.BertForTokenClassificationに関する公式ドキュメント](https://huggingface.co/transformers/v3.1.0/model_doc/bert.html#bertfortokenclassification)を見てもわからないが, inference時testデータにlabelsがないことについては`labels=None`と引数を渡してやるだけで良かった.  
> labels (torch.LongTensor of shape (batch_size, sequence_length), optional, defaults to None) – Labels for computing the token classification loss. Indices should be in [0, ..., config.num_labels - 1].  

なお, TPUの場合はbatch sizeを多めに取れるという[記事](https://qiita.com/koshian2/items/fb989cebe0266d1b32fc)があったため試してみたが2倍でもTPUメモリに乗り切らなかった.
<br>
<br>
<br>

#### 2021-04-27
[What is your best score without string matching?](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/232964)に気になる[投稿](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/232964#1277297)があった.  
> I am not doing any training yet. I am using a popular pretrained model and cleaning/filtering the results with basic string operations. These string operations are not informed by the training set labels.  

いまいち英語が理解できていないが, モデルの予測結果をknown labelsとのstring-matchingでcalibrateしてやって初めてLB\>0.7くらいのスコアになるのであって, calibrationをしない場合はLB\~0.2くらいがいいとこということなのだろうか？ データ及びタスクについて理解が浅いことからEDAに立ち返ることから始めたい.  
  
ところで[riow1983/kagglenb004-transformers-ner-inference](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)にて予測結果を確認すると, 全て'o'タグだったため[localnb001](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/notebooks/localnb001-transformers-ner.ipynb)のEPOCHS数を1から5に変更して再挑戦してみる件については, 今度は全て'o-dataset'タグだった.  
validationデータでの予測結果を確認すると, 予測タグには'o'と'o-dataset'両者が出現していたものの, 99%以上が'o'であり,  'o-dataset'タグは10,915,020件中たったの1,549件でしかなかった. また'o-dataset'となっている語句を確認してみると全くdatasetらしいセンテンスが選ばれていないことが判明. 当該結果セルのリンクは[こちら](https://colab.research.google.com/drive/1spO8nZOOgmTiNCNhxMJ2KP0uBYtMtIPK#scrollTo=gpjTHhAo4fm6&line=1&uniqifier=1).  
```
# tmp
    sentence_idx    isTrain pred
0   10883   0   monitor
1   160845  0   four or
2   8318    0   site throughout the
3   153839  0   h, determined by a
4   164381  0   additive predictors that provided the
... ... ... ...
585 101627  0   application. Researchers
586 50516   0   treatment as
587 35504   0   Unveiling pathological
588 48798   0   supported by
589 60614   0   20,
590 rows × 3 columns


# tmp["pred"].sample(10)
433                                        to compensate
573                                          name entity
92     a program designed to reduce the impact of flo...
99      at the point of interest in the new datum (NOAA,
277    species of tellinid bivalves, Macoma spp., at ...
217    Flounder population include an increase in lar...
490                                                of 12
493                                          the scanner
572                                           speed [89]
75     which is a tributary of the St. Lawrence River...
Name: pred, dtype: object
```
おそらくシーケンス長が290では文脈を把握するには不十分であり, より長いものが求められるように思える.  BigBirdのpre-trained modelがhuggingfaceから出ているので一度Colabで挑戦してみたい.  
<br>
<br>
<br>
  
#### 2021-04-28
- nb003-annotation-dataにて, spaCyによるPOS taggingの追加作業を検討
    - [data_preparation_ner](https://www.kaggle.com/shahules/coleridge-initiative-data-to-ner-format)の実装にspaCyによるPOS taggingを挿入する方が早いか
- nb005-pytorch-bert-for-nerにて, EPOCHS\>5で訓練検討
- EDAとして[A shameless journey into NLP from scratch](https://www.kaggle.com/lucabasa/a-shameless-journey-into-nlp-from-scratch)を読み始める
    - spaCyによるPOS taggingの着想を得る (特にpipelineを使った並列バッチ処理は参考になる)
    - spaCy公式: https://spacy.io/usage/linguistic-features
- チームシェアのためlocalnb001-transformers-nerをkaggle kernels push (l2knb001-transformers-ner)
<br>
<br>
<br>

#### 2021-04-29
情報整理を兼ねてREADME.mdに`My Assets`セクションを追加し, 自分が作成したnotebooks, datasets, modelsのメタ情報を記載. 今後新規作成の都度こまめに追記していく.  
従来から記載していたものはレファレンスの意味合いが強かったので`参考資料`セクション配下に置いた.  
<br>
Focusedは常に1つに絞る. 以下のようにする.  
[Focused]  
nb005-pytorch-bert-for-nerにて, EPOCHS>5で訓練検討  [issue #2](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/2)
<br>
[Secondary]  
nb003-annotation-dataにて, spaCyによるPOS taggingの追加作業を検討  [issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)
<br>
<br>
<br>

#### 2021-04-30
[issue #2](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/2)にてnb005をepochs=5で訓練するもColab Proのセッションが途中で切れて学習がresumeできない状況.  
<br>
なおnb005の入力データ作成用にkagglenb007を作成した. これはkagglenb006とほぼ同じ処理内容だがtextのsection構造を保持している点が異なる. これのoutputファイルをretrieveするため`kaggle kernels output`コマンドを実行したがエラーになっていた. これはKaggle APIのバージョンが古いことが起因していた. Kaggle APIのバージョン更新方法はやや工夫が必要で`!pip install --upgrade --force-reinstall --no-deps kaggle`としなければならなかった. https://qiita.com/RIRIh/items/6c8495a190e3c978a48f  
<br>
<br>
<br>

#### 2021-05-02
[issue #2](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/2)にてnb005をepochs=5で訓練するもColab Proのセッションが途中で切れて学習がresumeできない状況について, idle timeoutが仕込んでいたJavascriptでも防げていなかったことが原因と分かった. 修正版のJavascriptで解決した. なお本学習にはTesla V100で10時間程度要する見込み.  
<br>
nb005のinference notebook (Kaggle notebook)として[オリジナル](https://www.kaggle.com/tungmphung/coleridge-matching-bert-ner)をコピーしてkagglenb008とした.
<br>
<br>
<br>

#### 2021-05-03
[issue #2](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/2)にて, P100でおよそ10時間要したがnb005のepochs=5訓練完了. ただしepochs\>1でlossがepochs<=1よりも上昇していたため, 学習には失敗している可能性あり.  
とはいえひとまずfine-tunedモデルを[nb005](https://www.kaggle.com/riow1983/nb005-pytorch-bert-for-ner)としてKaggle dataset化し, [kagglenb008](https://www.kaggle.com/riow1983/kagglenb008-pytorch-bert-for-ner-inference)のinputにしてsubmission.csvを作成しsubmitしたが, LB=0.700とオリジナル(epochs=1)の結果と変わらず. やはり学習の失敗が原因か. nb005を中心に原因検証していきたい.  
<br>
[Tips: huggingfaceモデルのtraining途中再開方法]  
Colabのセッションが途中で切れるなどしてtrainingが中断しても, 中間ファイルcheckpoint-{num_checkpoint}を作成していればそこからtrainingを再開できる.    
> Resuming training
>You can resume training from a previous checkpoint like this:
>
>Pass --output_dir previous_output_dir without --overwrite_output_dir to resume training from the latest checkpoint in output_dir (what you would use if the training was interrupted, for instance).
>
>Pass --model_name_or_path path_to_a_specific_checkpoint to resume training from that checkpoint folder.  

[huggingface公式](https://huggingface.co/transformers/examples.html)    
[how to continue training from a checkpoint with Trainer?](https://github.com/huggingface/transformers/issues/7198)     
<br>
具体例:
```Python
def train(resume_training=False, num_checkpoint=None):
    if resume_training:
        os.environ["MODEL_PATH"] = f"./{output_folder}/checkpoint-{num_checkpoint}"
        !python ../input/kaggle-ner-utils/kaggle_run_ner.py \
        --model_name_or_path $MODEL_PATH \
        --train_file './train_ner.json' \
        --validation_file './train_ner.json' \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --save_steps 15000 \
        --output_dir $OUTPUT_DIR \
        --report_to 'none' \
        --seed 123 \
        --do_train 
    else:
        !python ../input/kaggle-ner-utils/kaggle_run_ner.py \
        --model_name_or_path 'bert-base-cased' \
        --train_file './train_ner.json' \
        --validation_file './train_ner.json' \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --save_steps 15000 \
        --output_dir $OUTPUT_DIR \
        --report_to 'none' \
        --seed 123 \
        --do_train 
```
<br>
<br>
<br>

#### 2021-05-04
nb005のMAX_LENをデフォルトの64のままepochs=5でsubmitしたところLB=0.700だった.  
<br>
huggingfaceでCV trainingする方法について少し調べたところ, 専用パイプライン的なものは無さそうだった.  
[Do transformers need Cross-Validation](https://discuss.huggingface.co/t/do-transformers-need-cross-validation/4074)  
[K fold cross validation](https://discuss.huggingface.co/t/k-fold-cross-validation/5765)  
[Splits and slicing](https://huggingface.co/docs/datasets/splits.html)  
<br>
<br>
<br>

#### 2021-05-05
[testデータのannotationが見直されたらしく](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/236508)自チームへの影響で言うと, これまでLB=0.700だったものがLB=0.533になった. これに伴いPublic LB順位もshakeしている.　別スレッド[Test data are NOT fully labeled (!!)](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/233170)も要確認か.  
<br>
nb005のMAX_LENを512まで延伸してepochs=5のsubmitをしてみたがLB=0.532だった. MAX_LEN=64でepochs=5はLB=0.533(前日までは0.700)だったので精度低下である.  
<br>
<br>
<br>

#### 2021-05-06
nb005は行き詰まったため, nb003再開.  
<br>
[Focused]  
nb003-annotation-dataにて, spaCyによるPOS taggingの追加作業を検討  [issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)  
<br>
[Secondary]  
nb005-pytorch-bert-for-nerにて, EPOCHS>5で訓練検討  [issue #2](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/2)  
<br>
ところがspaCyによるPOS taggingのfor loopがColab Proの制限時間(24時間)以内に終わらない. 問題の箇所は以下:
```Python
pos = []
for doc in nlp.pipe(df_train['text'].values, batch_size=50, n_process=-1):
    if doc.is_parsed:
        pos.append([n.pos_ for n in doc])
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        pos.append(None)
```
<br>
<br>
<br>

#### 2021-05-07
nb003-annotation-dataにて, spaCyによるPOS tagging引き続き取組中.  
<br>
<br>
<br>

#### 2021-05-08
nb003-annotation-dataにて, spaCyによるPOS tagging引き続き取組中.  
<br>
Discussion [Request to standardize the labels](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/236972)などannotationが一意に定まらない問題に触れたものを読んでいた. このままいくとコンペの質や信頼性はかなり低くなりそうだ.
<br>
<br>
<br>

#### 2021-05-09
nb003-annotation-dataにて, spaCyによるPOS taggingが進捗した, というか妥協してシーケンスlength\>=3000はdropしたら数分で終わった. なおこのやり方は[Bert for Question Answering Baseline: Training](https://www.kaggle.com/theoviel/bert-for-question-answering-baseline-training)からヒントを得た.  
これを入力とするlocalnb001-transformers-nerにて, posをsecond sentenceとするfine-tuningを開始.  
<br>
<br>
<br>

#### 2021-05-10
localnb001-transformers-nerにて, posをsecond sentenceとするfine-tuningが完了し, それを入力とするkagglenb004を実行したが, fine-tunedモデルの読み込みの際, num_labelsが合致しない(fine-tunedモデルのnum_labelsは2, 初期化モデルのnum_labelsは3)ためエラーとなっている.  
これは[PAD]トークンを'pad'というtagにして{'o', 'o-dataset', 'pad'}の3つのtag (label)を予測するBERTモデルを作成しているためnum_labels=3が正しいのだが, なぜかfine-tunedモデルのnum_labelが2のままになっているため保存方法に誤りはなかったかなど調査中.   
エラー全文:  
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-28-c0b53fe32178> in <module>
----> 1 model = BERTClass()
      2 #model.load_state_dict(torch.load(f'../input/localnb001-transformers-ner/bert-base-cased-ner-cv{CV}.pt',
      3 #                                 map_location=dev))
      4 #output_model = f"../input/d/riow1983/localnb001-transformers-ner/bert-base-cased-ner-cv{CV}.pth"
      5 #output_model = f"../input/localnb001-transformers-ner/bert-base-cased-ner-cv{CV}.pth"

<ipython-input-25-905fd1052f1d> in __init__(self)
      9         #self.l1 = transformers.BertForTokenClassification.from_pretrained(f'../input/localnb001-transformers-ner/bert-base-cased-ner-cv{CV}.bin')
     10         #self.l1 = transformers.BertForTokenClassification.from_pretrained('../input/d/riow1983/localnb001-transformers-ner', num_labels=num_labels)
---> 11         self.l1 = transformers.BertForTokenClassification.from_pretrained('../input/localnb001-transformers-ner', num_labels=num_labels)
     12         #self.l1 = transformers.BertForTokenClassification#.from_pretrained('./')
     13 

/opt/conda/lib/python3.7/site-packages/transformers/modeling_utils.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
   1181             if len(error_msgs) > 0:
   1182                 error_msg = "\n\t".join(error_msgs)
-> 1183                 raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
   1184         # make sure token embedding weights are still tied if needed
   1185         model.tie_weights()

RuntimeError: Error(s) in loading state_dict for BertForTokenClassification:
  size mismatch for classifier.weight: copying a param with shape torch.Size([2, 768]) from checkpoint, the shape in current model is torch.Size([3, 768]).
  size mismatch for classifier.bias: copying a param with shape torch.Size([2]) from checkpoint, the shape in current model is torch.Size([3]).
```  
<br>
<br>
<br>

#### 2021-05-11
Kaggle dataset [localnb001](https://www.kaggle.com/riow1983/localnb001-transformers-ner)更新. s3に置いてあるhuggingface pre-trained BERT modelのconfigファイルbert_config.jsonをconfig.jsonに改名(Google Drive上で手動改名)し, Kaggleにupload. というのもこれをinputとするkagglenb004でpre-trained BERTを呼び出した際, num_labels=3としているにも関わらず, output featuresが3にならないという不具合があったため, config.jsonがおかしいと疑ったため. 結果改善した模様.  
<br>
kagglenb004の実装完了し, submission.csvも出力可能となったためsubmitしようとしたが, なんとaccerelatorにTPUを指定したnotebookからはinternet disabledでもsubmitが許可されていないということが判明.  
![input file image](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/png/Screenshot%202021-05-12%20at%208.12.58.png?raw=true)  
<br>
こうなっている理由として今のところ見つけた説明としては"TPUは裏で(インターネット経由で)
GCSにデータを取りに行くから"ということらしい. https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/199750  
このtopic authorが作成したnotebook [[Training] PyTorch-TPU-8-Cores](https://www.kaggle.com/joshi98kishan/foldtraining-pytorch-tpu-8-cores/comments?scriptVersionId=48061653)を参考にして作成したのがkaggle004だっただけに釈然としないところがある. これを受けて他コンペではあるが[Code Competition規約](https://www.kaggle.com/c/hubmap-kidney-segmentation/overview/code-requirements)にTPU使用に関する注意が加筆されていた.  
> TPUs will not be available for making submissions to this competition. You are still welcome to use them for training models. A walk-through for how to train on TPUs and run inference/submit on GPUs, see our TPU Docs.  

本コンペのDiscussionに投稿しようかとも思ったが, TPUの仕様上GCSとonlineで繋ぐことになるということであればインターネット使用制限に抵触するためsubmitできないのは納得できる. 詳しい説明がなされている[notebook](https://www.kaggle.com/docs/tpu)があったのでこちらをまずは通読したい.  
<br>
と同時にkagglenb004をGPUでも動くように変更する作業に着手開始.  
<br>
閑話休題.  
今更ながらnp.whereの使い方:  
```Python
a = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
np.where(a<=2, a, "-")
```
array(['1', '2', '-', '-', '-', '-', '-', '2', '1'], dtype='<U21')
```Python
cond = (a>=3)&(a<5)
print(cond)
out = " ".join(list(np.where(cond, a, "|")))
print(out)
```  
[False False  True  True False  True  True False False]    
'| | 3 4 | 4 3 | |'  
cond.shapeとa.shapeが一致していることが必要.  
<br>
<br>
<br>

#### 2021-05-12
kagglenb004をGPUでも動くように変更する作業難航中. localnb001で訓練したfine-tunedモデルをcpu側に移して保存したものをkagglenb004でloadする方法は今のところうまくいかない. この方式は[こちらのnotebook](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step2#MODEL)で採用されているように見えるが細部が違うのかもしれない. 遭遇しているエラーは以下の通り:  
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-29-eb037b7c0e98> in <module>
      7     checkpoint = torch.load(output_model, map_location='tpu')
      8 else:
----> 9     checkpoint = torch.load(output_model)
     10 model.load_state_dict(checkpoint['model_state_dict'])
     11 

/opt/conda/lib/python3.7/site-packages/torch/serialization.py in load(f, map_location, pickle_module, **pickle_load_args)
    593                     return torch.jit.load(opened_file)
    594                 return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
--> 595         return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
    596 
    597 

/opt/conda/lib/python3.7/site-packages/torch/serialization.py in _legacy_load(f, map_location, pickle_module, **pickle_load_args)
    772     unpickler = pickle_module.Unpickler(f, **pickle_load_args)
    773     unpickler.persistent_load = persistent_load
--> 774     result = unpickler.load()
    775 
    776     deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

/opt/conda/lib/python3.7/site-packages/torch/_utils.py in _rebuild_xla_tensor(data, dtype, device, requires_grad)
    175 
    176 def _rebuild_xla_tensor(data, dtype, device, requires_grad):
--> 177     tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    178     tensor.requires_grad = requires_grad
    179     return tensor

RuntimeError: Could not run 'aten::empty_strided' with arguments from the 'XLA' backend. 'aten::empty_strided' is only available for these backends: [CPU, CUDA, BackendSelect, Named, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, Tracer, Autocast, Batched, VmapMode].

CPU: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/build/aten/src/ATen/CPUType.cpp:2127 [kernel]
CUDA: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/build/aten/src/ATen/CUDAType.cpp:2983 [kernel]
BackendSelect: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/build/aten/src/ATen/BackendSelectRegister.cpp:761 [kernel]
Named: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
AutogradOther: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
AutogradCPU: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
AutogradCUDA: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
AutogradXLA: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
AutogradPrivateUse1: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
AutogradPrivateUse2: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
AutogradPrivateUse3: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/VariableType_0.cpp:7974 [autograd kernel]
Tracer: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/torch/csrc/autograd/generated/TraceType_0.cpp:9341 [kernel]
Autocast: fallthrough registered at /opt/conda/conda-bld/pytorch_1603729047590/work/aten/src/ATen/autocast_mode.cpp:254 [backend fallback]
Batched: registered at /opt/conda/conda-bld/pytorch_1603729047590/work/aten/src/ATen/BatchingRegistrations.cpp:511 [backend fallback]
VmapMode: fallthrough registered at /opt/conda/conda-bld/pytorch_1603729047590/work/aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
```  
<br>
<br>
<br>

#### 2021-05-13
kagglenb004をGPUでも動くように変更する作業成功か. localnb001(TPU)にてfine-tunedモデルを保存する際, `torch.save()`ではなく`xm.save()`をすることで, kagglenb004からGPUでloadできるようになった.
```Python
folder = "localnb001-transformers-ner"
!mkdir {folder}
#PATH = f"bert-base-cased-ner-cv{cv}.pth"
PATH = f"bert-base-cased-ner-pad-cv{cv}.pth"

def save(model, optimizer, folder, path, as_tpu=False):
    # save
    if as_tpu:
        #torch.save({
        xm.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "./"+folder+"/"+path)
    else:
        #torch.save({
        xm.save({
            'model_state_dict': model.to("cpu").state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "./"+folder+"/"+path)

save(model, optimizer, folder, PATH, as_tpu=False)
```  
これは以下のissues:  
- [Loading a model checkpoint that is trained on TPU using a GPU #2303](https://github.com/PyTorchLightning/pytorch-lightning/issues/2303)
- [Use xm.save to save model on TPU #3044](https://github.com/PyTorchLightning/pytorch-lightning/pull/3044)  

を参考にした.  
<br>
<br>
<br>

#### 2021-05-14
kagglenb004はsubmitできた. しかし２回のsubmitの内, 1つは"Kaggle Error", 1つはNotebook Timeout"だった.  
<br>
PyTorch(-XLA)によるtrain loop, inferenceのtemplateコードの作成を以下notebookを参考に開始.  
- [RANZCR / ResNet200D / 3-stage training / step1](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step1)  
- [RANZCR / ResNet200D / 3-stage training / step2](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step2)
- [RANZCR / ResNet200D / 3-stage training / sub](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-sub)  
<br>
<br>
<br>

#### 2021-05-15
kagglenb004がsubmit errorになった主な原因はspaCyによるPOS taggingに時間がかかっていることだと思われ, 当該箇所を除外したもので再提出.  
引き続きPyTorch(-XLA)によるtrain loop, inferenceのtemplateコードを作成中.   
<br>
<br>
<br>

#### 2021-05-16 ~ 2021-05-17
kagglenb004のsubmission errorが解消されず対応中.  
Dataset作成の前処理でデータセット全体をメモリに読み込む処理があり`Notebook Exceeded Allowed Compute`となっていた点はPyTorch Datasetでbatchごとに処理する方式に変更することで解消した模様.  
これはモデルへの直接的な入力は, sentencesからindexを指定したもの(sentencesのsubset)になるが, sentencesはdfから作成されており, dfからindexを指定したもの(dfのsubset)に対して加工処理が必要になる, というもの. このindexのnest構造についてPyTorchのDatasetで実装する方法として以下の例が大変参考になった:  
```Python
class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc = transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class
```  
[source](https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f)


ただし今度は`Notebook Timeout`となった.  
<br>
<br>
<br>

#### 2021-05-18
kagglenb004が`Notebook Timeout`になる件について, batch sizeを16から32にしてsubmitするも, やはり`Notebook Timeout`.  
<br>
効率化のための施作:  
- パラメータなどをまとめた`config/config.yml`作成開始  
- train.csv (or sample_submission.csv)を読み込んだ時点からPyTorch Datasetに入力するまでの処理を記載した`src/bridge.py`を作成開始  
<br>
<br>
<br>

#### 2021-05-19
kagglenb004が`Notebook Timeout`になる件について, batch sizeを32から48にしてsubmitするも, やはり`Notebook Timeout`.  
PyTorch DataLoaderのnum_workersが0になっていたので, これを8にして, なおかつsentence_idxをlabel encodingするといった不要な処理を除外して再度submitするも, これも`Notebook Timeout`.  
<br>
train.csv (or sample_submission.csv)を読み込んだ時点からPyTorch Datasetに入力するまでの処理を記載した`src/bridge.py`完成. 実行&結果確認へ.  
<br>
<br>
<br>

#### 2021-05-20
`src/bridge.py`の処理待ち中にディスカッションを読む.  
<br>
チーム定例会にて以下課題を認識:  
- kagglenb004が`Notebook Timeout`になる件については, 1 iterationごとの処理量を削減する方向 (batch size減らす, max_len減らす, など)も検討すべき
- textをmax_lenごとに分割する場合, 分割前後でoverlapを20単語程度設けないとdataset言及途中で分割されることになる
- localnb001系統はtagを`{"o", "o-dataset", "pad"}`にしていたが, BIO方式`{"o":O", "o-dataset":B", "o-dataset":I", "pad":O"}`にした方が良いか
- CVを切る際, publicationのドメインカテゴリをグループにしたGroup KFoldが望ましい:
  - ただしpublicationのドメインカテゴリを把握するのは容易ではない
  - したがって妥協策として, "publication ID (or title)をグループとしたlabelによる層別化"をするStratified Group Kfoldを実施すると良いかもしれない
- dataset labelは大文字の場合も小文字の場合もあるので, casedモデルよりも大文字小文字の違いを区別しないuncasedモデルの方が望ましい可能性あり
- labelの付け方など問題山積なコンペなので, 機械学習モデルを学習させて競うというよりは, string matchingを始めとする有象無象のハッキング手法を競うコンペなのかもしれない
<br>
<br>
<br>

#### 2021-05-21
`src/bridge.py`の下記処理がメモリエラーなのか処理が完結できない.  
```Python
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
```  
エラーメッセージ:   
```
Starting to convert df to dataset...
Converting tokens...: 13489it [01:31, 11.26it/s]/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/process_executor.py:691: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  "timeout or by a memory leak.", UserWarning
Converting tokens...: 19661it [02:42, 121.18it/s]
Appending...:  32% 6321/19661 [1:23:00<5:38:32,  1.52s/it]tcmalloc: large alloc 1359011840 bytes == 0x55b4e3a5a000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  36% 7039/19661 [1:42:11<6:00:38,  1.71s/it]tcmalloc: large alloc 1528913920 bytes == 0x55b548d72000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  40% 7902/19661 [2:08:13<6:16:55,  1.92s/it]tcmalloc: large alloc 1720107008 bytes == 0x55b4f9258000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  44% 8590/19661 [2:31:44<6:37:04,  2.15s/it]tcmalloc: large alloc 1934966784 bytes == 0x55b57946a000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  51% 10083/19661 [3:28:47<6:28:36,  2.43s/it]tcmalloc: large alloc 2176851968 bytes == 0x55b51463e000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  56% 11054/19661 [4:12:55<9:04:05,  3.79s/it]tcmalloc: large alloc 2448891904 bytes == 0x55b492a6a000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  63% 12309/19661 [5:42:47<9:32:49,  4.67s/it]tcmalloc: large alloc 2755297280 bytes == 0x55b536db8000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  66% 12980/19661 [6:37:33<10:05:42,  5.44s/it]tcmalloc: large alloc 3100139520 bytes == 0x55b492a6a000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  68% 13332/19661 [7:13:08<11:16:12,  6.41s/it]tcmalloc: large alloc 3487178752 bytes == 0x55b5627a8000 @  0x7f2560428001 0x7f255d94754f 0x7f255d997b58 0x7f255d997d18 0x7f255da3f010 0x7f255da3f73c 0x7f255da3f85d 0x55b22dcb2f68 0x7f255d984ef7 0x55b22dcb0c47 0x55b22dcb0a50 0x55b22dd24453 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dcb230a 0x55b22dd2060e 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f4ae 0x55b22dcb23ea 0x55b22dd2132a 0x55b22dd1f7ad 0x55b22dcb23ea
Appending...:  68% 13380/19661 [7:18:20<11:35:30,  6.64s/it]/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/backend/resource_tracker.py:320: UserWarning: resource_tracker: There appear to be 6 leaked semlock objects to clean up at shutdown
  (len(rtype_registry), rtype))
/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/backend/resource_tracker.py:320: UserWarning: resource_tracker: There appear to be 1 leaked folder objects to clean up at shutdown
  (len(rtype_registry), rtype))
^C
```  
<br>
CVを切る際, publicationのドメインカテゴリをグループにしたGroup KFoldが望ましいについて  
チームメイトと検討の結果以下の方法で行うこととした [issue #9](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9):  
- 対象は無加工のtrain.csv  
- cleaned_labelをカテゴライズしたものをgroupとしてGroup Kfoldを行う
  - その際, 教師ラベルをどの変数(カラム)にするかは未定 (適当でいい?)  
<br>
<br>
<br>

#### 2021-05-22
`src/bridge.py`のメモリエラーに対応すべく, 以下の通り変更を加えた(functools.reduce導入)が同じ結果に.  
```Python
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
    df = Parallel(n_jobs=-1)(delayed(convert_tokens)(row,
                                                     i, 
                                                     max_len,
                                                     train=train,
                                                     use_pos=use_pos,
                                                     verbose=verbose) for i,row in tqdm(df.iterrows(), desc="Converting tokens..."))
    #df = pd.concat(df, axis=0, ignore_index=True)
    #df = pd.DataFrame()
    #for _df in tqdm(dfs, desc="Appending..."):
    #    df = df.append(_df, ignore_index=True)
    df = reduce(lambda x,y: pd.concat([x,y], axis=0, ignore_index=True), df)
    

    
    df["sentence_idx"] = df["sentence"] + df["sentence#"]
    #dataset = dataset[["sentence", "sentence_idx", "token", "pos"]].copy()
    #dataset.rename(columns={"token":"word"}, inplace=True)

    return df
```
```
Starting to convert df to dataset...
Converting tokens...: 14844it [01:50, 14.90it/s]/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/process_executor.py:691: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  "timeout or by a memory leak.", UserWarning
Converting tokens...: 19661it [02:40, 122.47it/s]
tcmalloc: large alloc 1359011840 bytes == 0x5575691ca000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 1528913920 bytes == 0x5575c4380000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 1720107008 bytes == 0x5575691ca000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 1934966784 bytes == 0x5575dc6e6000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 2176851968 bytes == 0x5575691ca000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 2448891904 bytes == 0x55776a488000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 2755297280 bytes == 0x5575691ca000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 3100139520 bytes == 0x55776a488000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
tcmalloc: large alloc 3487178752 bytes == 0x5575691ca000 @  0x7f62f186a001 0x7f62eed8954f 0x7f62eedd9b58 0x7f62eedd9d18 0x7f62eee81010 0x7f62eee8173c 0x7f62eee8185d 0x5572fec79f68 0x7f62eedc6ef7 0x5572fec77c47 0x5572fec77a50 0x5572feceb453 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece67ad 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fec7930a 0x5572fece760e 0x5572fece64ae 0x5572fec793ea 0x5572fece832a 0x5572fece64ae 0x5572febb8eb1 0x5572fedea5df 0x5572fec77c47 0x5572fec77a50
/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/backend/resource_tracker.py:320: UserWarning: resource_tracker: There appear to be 6 leaked semlock objects to clean up at shutdown
  (len(rtype_registry), rtype))
/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/backend/resource_tracker.py:320: UserWarning: resource_tracker: There appear to be 1 leaked folder objects to clean up at shutdown
  (len(rtype_registry), rtype))
^C
```  
<br>
<br>
<br>

#### 2021-05-23
`src/bridge.py`のメモリエラーに対応作業継続.  
<br>
<br>
<br>

#### 2021-05-24
`src/bridge.py`のメモリエラーについて, train.csvのtext長が3000以下の論文をdropすると正常かつ迅速に処理が完了することを確認.  
ただしその場合`19661 rows`が`3000 rows`強まで落ち込んでいることに気づく. これは流石に減らしすぎだ.  
train.csvのtext長の分布を見ると30000以下が大部分となっている.  
![input file image](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/png/20210524.png?raw=true)  
![input file image](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/png/20210524(2).png?raw=true)  
またdropするのではなく, text長をtruncateすればdropする論文は0になる. その方式でひとまずやってみることにした.  
またconcatenationはpd.concatではなくappend方式を採用した.  
しかしやはり以下のエラーで処理が中断されてしまった.  
```
Starting to convert df to dataset...
    Converting tokens...: 19661it [02:21, 139.19it/s]
    Starting to concatenate...
        Appending...:  32% 6351/19661 [1:25:45<5:41:28,  1.54s/it]tcmalloc: large alloc 1359241216 bytes == 0x560551c4c000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  36% 7047/19661 [1:44:38<6:02:46,  1.73s/it]tcmalloc: large alloc 1528823808 bytes == 0x560551c4c000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  40% 7915/19661 [2:11:03<6:20:45,  1.94s/it]tcmalloc: large alloc 1720090624 bytes == 0x560551c4c000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  44% 8605/19661 [2:34:52<6:43:34,  2.19s/it]tcmalloc: large alloc 1935032320 bytes == 0x560551c4c000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  51% 10106/19661 [3:32:59<6:31:20,  2.46s/it]tcmalloc: large alloc 2176942080 bytes == 0x56062bb34000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  56% 11075/19661 [4:18:23<9:17:12,  3.89s/it]tcmalloc: large alloc 2448932864 bytes == 0x5607df248000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  63% 12328/19661 [5:49:06<9:36:45,  4.72s/it]tcmalloc: large alloc 2755100672 bytes == 0x5606cfe28000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  66% 13034/19661 [6:49:12<10:36:24,  5.76s/it]tcmalloc: large alloc 3100155904 bytes == 0x5607df248000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  69% 13657/19661 [7:53:09<10:57:11,  6.57s/it]tcmalloc: large alloc 3486982144 bytes == 0x5606fb83a000 @  0x7f449de60001 0x7f449b37f54f 0x7f449b3cfb58 0x7f449b3cfd18 0x7f449b477010 0x7f449b47773c 0x7f449b47785d 0x560370374f68 0x7f449b3bcef7 0x560370372c47 0x560370372a50 0x5603703e6453 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x56037037430a 0x5603703e260e 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e14ae 0x5603703743ea 0x5603703e332a 0x5603703e17ad 0x5603703743ea
        Appending...:  70% 13797/19661 [8:08:33<10:51:40,  6.67s/it]/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/backend/resource_tracker.py:320: UserWarning: resource_tracker: There appear to be 6 leaked semlock objects to clean up at shutdown
  (len(rtype_registry), rtype))
/usr/local/lib/python3.7/dist-packages/joblib/externals/loky/backend/resource_tracker.py:320: UserWarning: resource_tracker: There appear to be 1 leaked folder objects to clean up at shutdown
  (len(rtype_registry), rtype))
^C
```  
joblib特有の問題かもしれない.  
<br>
[issue #9](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9)について  
[kagglenb009-cv](https://www.kaggle.com/riow1983/kagglenb009-cv)作成開始. 130 labelsについてlabels by labelsのペアワイズ・コサイン類似度を計算しようと思ったがやはり時間がかかるので他の方法を模索中.  
<br>
<br>
<br>

#### 2021-05-25
`src/bridge.py`のメモリエラーに対応作業継続. joblibを廃止しmultiprocessingへ変更したところAppend処理は100%達成できたもののその直後に^C (中断)を喰らう. エラー内容が表示されておらず理由不明だが恐らくメモリエラー.  
```
Starting to convert df to dataset...
    Converting tokens...: 19661it [00:04, 4357.25it/s]
    Starting to concatenate...
        Appending...:  32% 6351/19661 [1:24:58<5:38:53,  1.53s/it]tcmalloc: large alloc 1359241216 bytes == 0x556faf58e000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  36% 7047/19661 [1:43:40<6:02:58,  1.73s/it]tcmalloc: large alloc 1528823808 bytes == 0x556faf58e000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  40% 7915/19661 [2:10:20<6:22:04,  1.95s/it]tcmalloc: large alloc 1720090624 bytes == 0x5570b7c8c000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  44% 8605/19661 [2:34:13<6:42:43,  2.19s/it]tcmalloc: large alloc 1935032320 bytes == 0x5570c4994000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  51% 10106/19661 [3:32:03<6:26:54,  2.43s/it]tcmalloc: large alloc 2176942080 bytes == 0x557154c1a000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  56% 11075/19661 [4:13:48<6:41:23,  2.80s/it]tcmalloc: large alloc 2448932864 bytes == 0x557175278000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  63% 12328/19661 [5:15:37<6:25:33,  3.15s/it]tcmalloc: large alloc 2755100672 bytes == 0x5570f5766000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  66% 13034/19661 [5:55:01<7:52:02,  4.27s/it]tcmalloc: large alloc 3100155904 bytes == 0x5571c2a16000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  69% 13657/19661 [6:34:46<6:32:33,  3.92s/it]tcmalloc: large alloc 3486982144 bytes == 0x557121178000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  75% 14768/19661 [8:05:27<8:27:37,  6.22s/it]tcmalloc: large alloc 3923599360 bytes == 0x55728fcec000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  81% 15925/19661 [10:17:13<7:50:12,  7.55s/it]tcmalloc: large alloc 4414021632 bytes == 0x5571584d6000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...:  91% 17887/19661 [14:25:41<2:51:24,  5.80s/it]tcmalloc: large alloc 4964737024 bytes == 0x557051472000 @  0x7fb3418e7001 0x7fb33ee0654f 0x7fb33ee56b58 0x7fb33ee56d18 0x7fb33eefe010 0x7fb33eefe73c 0x7fb33eefe85d 0x556de8426f68 0x7fb33ee43ef7 0x556de8424c47 0x556de8424a50 0x556de8498453 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de842630a 0x556de849460e 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84934ae 0x556de84263ea 0x556de849532a 0x556de84937ad 0x556de84263ea
        Appending...: 100% 19661/19661 [17:24:39<00:00,  3.19s/it]
^C
```  
<br>
[issue #9](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9)について  
[kagglenb009-cv](https://www.kaggle.com/riow1983/kagglenb009-cv)をlocal(Colab)にpullしたnb009-cvで作業継続. 130 x 130のペアワイズコサイン類似度を求めることにしたが, その前段で目視確認による人手マッピングを噛ませて精度向上を図る.  
<br>
<br>
<br>

#### 2021-05-26
[issue #9](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9)について  
`nb009-cv: CV作成 -> kagglenb006: get_text処理 -> src/bridge.py on localnb001: Dataset作成`
の処理フローが正常に動作することを確認. この流れは今後別のコンペに参加する際も流用できる.  
<br>
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
`src/bridge.py`のメモリエラーをpandasのappend処理の効率化で対応しようとしていたが悉く失敗. 根本的な見直しとして:
- 文字列たるtextを配列化し, 要素単語ごとにdataframe１行を与える処理(縦持ち変換)がメモリ効率性が最悪  
- 縦持ち変換をしていた主な理由は, CV作成のため教師ラベル列=`tag`列を作成することだった  
- しかし新採用のnb009-cv方式では教師ラベル列=`cleaned_label`列なので`tag`列不要となった ([詳細](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9#issuecomment-848335514))  

以上のことから縦持ち変換処理を廃止したところ, 処理は早期に終了した. これによりinference時の`Notebook Timeout`も回避できるのではないかと思う.  
<br>
<br>
<br>

#### 2021-05-27
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
`src/bridge.py`, `config/config.yml`もKaggle Dataset [localnb001-transformers-ner](https://www.kaggle.com/riow1983/localnb001-transformers-ner)にuploadしてinference notebook [kagglenb004](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)で実行. submission.csv作成のためのマイナー処理記載.  
<br>
<br>
<br>

#### 2021-05-28
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
inference notebook [kagglenb004](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)にてsubmit失敗(`submission scoring error`)  
状況確認中.  
<br>
<br>
<br>

#### 2021-05-29 ~ 2021-05-30
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
inference notebook [kagglenb004](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)にてsubmit失敗(`Notebook Timeout`)  
状況確認中. 
<br>
<br>
<br>

#### 2021-05-31
実験管理テーブルを本READMEのtopに記載することとした. なおsubmitしたら返す刀でgit pushする習慣にする. その際LBスコアの結果が出てないことが大半なのでgit commit messageにはLBスコアは空白とし, kaggle submit messageをコピペしておくこととし(e.g., `git commit -m "[Submit] {kaggle submit message} LB= {issue #}"`), LBスコアは後で得られ次第README上の実験管理テーブルに記載するようにする. こうすることで連続実験(serial experiments)にも耐え得るはず.    
<br>
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
引き続き`Notebook Timeout`を回避すべくハイパラ調整中.  
<br>
[issue #9](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/9)について  
cosine similarity計算前の目視確認による辞書作成完了, CVデータをチーム内リリース.  
<br>
<br>
<br>

#### 2021-06-01
優先順位見直し. [string matching a.k.a. LB proving #10](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/10)を第一優先事項に.  
<br>
<br>
<br>

#### 2021-06-02
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
チームメイトによる指摘でBERT tokenizerが一単語を複数トークンに分割する仕様になっていることを認識. huggingface + PyTorchのハイブリッド方式でモデルを作成する場合, どのようにこの点を考慮すれば良いか検討する必要が出てきた.  
<br>
[issue #10](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/10)について  
govtやacronym, matching方式の工夫(string in string / string in List)など実験したがgovtを使うと精度が落ちるようだった.  
<br>
チームメイトによるsubmit後のNotebook Runtimeの長短に関する事象報告とLB provingに関する考察:  
- (正しいinference notebook) string matchingでlabelが取得**できなかった**場合に限って, NERモデルによる予測実行  
- (誤ったinference notebook) string matchingでlabelを取得**できた**場合に限って, NERモデルによる予測実行  

(誤)によるNotebook Runtimeが, (正)によるものよりも大幅に短かった. これはhidden test dataにおいてstring matchingでlabelを取得できたケースが少なく, NERモデルによる予測実行の機会が少なかったことによるものと推測される. つまりstring matchingによる精度改善はhidden test dataにおいてはそれほど期待できないという結論になる.  
これはBERTなどによる言語モデルによる予測を全うにやっていかなければコンペに勝てないということを意味するため, やはり[issue #10](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/10)よりも[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)を優先しようと思う.  
<br>
<br>
<br>

#### 2021-06-03
優先順位見直し.  
[Focused]  
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)  
<br>
[Secondary]  
[issue #10](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/10)  
<br>
<br>
<br>

#### 2021-06-04
[issue #10](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/10)について  
testデータの行数に対し, string matchする行をカウントしていき, カウント数/testデータ行数(=p)が閾値以下であればpredを全てnull string("")に置き換える処理を書いてsubmitしたところ, 0.4 \<= p \< 0.5であることが判明した.  
![input file image](https://github.com/riow1983/Kaggle-Coleridge-Initiative/blob/main/png/20210604.png?raw=true)  
<br>
<br>
<br>

#### 2021-06-05
[issue #7](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/7)について  
max_len=5にしてBERT tokenizerのword-piece tokenizerの動きを確認. huggingface公式ドキュメントにピンポイントの説明をまだ見つけられていないが, まずtokenizer.decoderはspecial tokens ([CLS], [SEP])およびword-piece tokenのsub (##で始まるもの)を考慮して元の文に戻す(復元する)ことができる,  
https://huggingface.co/transformers/glossary.html#token-type-ids 
<br>
つまり一見ids-tagの対応関係が崩れているように見えるが, huggingface tokenizerの機能としてこの対応関係は保持されている.    
これに関してMediumの記事([Fine Tuning BERT for NER on CoNLL 2003 dataset with TF 2.0](https://medium.com/analytics-vidhya/fine-tuning-bert-for-ner-on-conll-2003-dataset-with-tf-2-2-0-2f242ca2ce06))に以下の図が参考になった:  
```
input_ids - [101, 7270, 22961, 1528, 1840, 1106, 21423, 1418, 2495, 12913, 119, 102, 0, 0, 0, 0]
input_mask - [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
segment_ids - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label_ids - [10, 6, 1, 2, 1, 1, 1, 2, 1, 1, 11, 0, 0, 0, 0, 0]
label_mask - [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False]
valid_ids - [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1] # 0 for '##mb'
```  
valid_idsがword-piece tokenのmain-subの関係情報を記述している. ここでいうvalid_idsと同等のものがhuggingface tokenizerにも内蔵されていると思われる. (でなければtokenizer.decoderは機能し得ない.) 

<br>
その上で依然として問題になるのが, tokenizer.encode_plus()でmax_lenでtruncateされることで(このtruncateは後続のBERTモデルに入力する際にlen(ids)とlen(tags)の長さがmax_lenに揃っていないとエラーになるため必須), special tokensとword-piece tokenのsubのトークン数分だけ復元した文が尻切れトンボになる点だ.  
以下2例を示す:  

```
# 例1
id: 5df96c17-d198-4116-b7cd-55bf8904b3c8 ---- index: 9 ---- len(ids): 5 ---- len(label): 5
<<< original_tokens: of the probability distribution with >>>
<<< decoded_tokens:  of the probability >>>
<<< overflowing_tokens:                      ['with', 'distribution'] >>>

recovered_token <----> original_token
-------------------------------------
of <----> of
the <----> the
probability <----> probability



# 例2
id: 5df96c17-d198-4116-b7cd-55bf8904b3c8 ---- index: 10 ---- len(ids): 5 ---- len(label): 5
<<< original_tokens: various thresholds for extremeness commonly >>>
<<< decoded_tokens:  various thresholds >>>
<<< overflowing_tokens:                      ['commonly', '##ness', 'extreme', 'for'] >>>

recovered_token <----> original_token
-------------------------------------
various <----> various
threshold <----> thresholds
##s <----> for
```  
この問題は, 入力のsentenceを尻切れトンボ分を見越して一定長のoverlapを設けることで解消するのが最も単純な方法だと思う.  
<br>
<br>
<br>

#### 2021-06-06
優先順位見直し.  
[Focused]  
[CONLL Corpora (2003) でNERモデル構築 (huggingface + PyTorch 利用) #11](https://github.com/riow1983/Kaggle-Coleridge-Initiative/issues/11)  
<br>
<br>
<br>

#### 2021-06-07










