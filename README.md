# Kaggle-Coleridge-Initiative


#### notebook命名規則
- kagglenb001-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- nb001-hoge.ipynb: kagglenb001-hoge.ipynbをlocalにpullしlocalで変更を加えるもの. 番号はkagglenb001-hoge.ipynbと共通.
- localnb001-hoge.ipynb: localで新規作成されたnotebook. 

#### Papers
|name|url|status|comment|
|----|----|----|----|
|Big Bird: Transformers for Longer Sequences|https://arxiv.org/pdf/2007.14062.pdf|Reading|Turing completeの意味が分からん|

#### Blogs
|name|url|status|comment|
|----|----|----|----|
|Understanding BigBird's Block Sparse Attention|https://huggingface.co/blog/big-bird|Untouched||

#### Documentation / Tutorials
|name|url|status|comment|
|----|----|----|----|
|SAVING AND LOADING MODELS|https://pytorch.org/tutorials/beginner/saving_loading_models.html|Reading|PyTorch標準方式のモデルsave方法|
|Source code for pytorch_transformers.tokenization_bert|https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/tokenization_bert.html|Done|bert-base-cased tokenizerをKaggle上で使用するためS3レポジトリからwget|
|Huggign Face's notebooks|https://huggingface.co/transformers/notebooks.html|Bookmarked|-|
|Fine-tuning a model on a token classification task|https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb|Done|huggingfaceによるNERタスクのチュートリアル.<br>ただしfine-tunedモデルの保存に関する実装はない<br>なお標準的なhuggingface方式では保存したいモデルはアカウントを作ってウェブレポジトリにアップロードするらしい<br>Kaggleから使える？|
|Model sharing and uploading|https://huggingface.co/transformers/model_sharing.html|Bookmarked|huggingface方式のモデル保存方法について|

#### Kaggle Notebooks
|name|url|status|comment|
|----|----|----|----|
|Coleridge - Huggingface Question Answering|https://www.kaggle.com/jamesmcguigan/coleridge-huggingface-question-answering|Done|QAのtoy example的なやつ. <br>結局こんな精度じゃ話にならない. <br>また事後学習する方法が分からず終い.|
|HuggingFace Tutorial | Custom PyTorch training|https://www.kaggle.com/moeinshariatnia/simple-distilbert-fine-tuning-0-84-lb|Bookmarked|huggingfaceのpre-trainedモデルをfine-tuningするも<br>PyTorch標準のsave方式を採用している<br>らしいところは参考になる|
|Bert PyTorch HuggingFace Starter|https://www.kaggle.com/theoviel/bert-pytorch-huggingface-starter|Bookmarked|huggignface PyTorchのとても綺麗なコード.<br>参考になるがfine-tuned modelのsave実装はない.|


#### Kaggle Datasets
|name|url|status|comment|
|----|----|----|----|
|nb003-annotation-data|https://www.kaggle.com/riow1983/nb003-annotation-data |Done|CVデータ|


#### 2021-04-15  
実験管理方法のスライド作成し共有.


#### 2021-04-16
どうやらQA系では精度がでないらしい. NER系は精度でていそう.  
ひとまず学習済みBERTをNERタスクで事後学習させる方法を確立したい.

#### 2021-04-20
Google Colab ProおよびGoogle Drive strage+185GB課金した.  
課金した理由:  
- Colab無料版ではRAMが足りない局面に多く遭遇しそうだった
- Google Drive無料版のストレージ(15GB)では中間ファイルの吐き出しですぐ満杯になる  

ところでColabのセッション切れに対応する裏技としてChromeのデベロッパーツールのコンソールに  
定期的にconnectボタンをクリックするJavascriptを入力しておくというものがあり試してみた.
```Javascript
function ClickConnect(){
  console.log("60sごとに再接続");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,1000*60);
```
https://flat-kids.net/2020/07/28/google-colab-%E3%82%BB%E3%83%83%E3%82%B7%E3%83%A7%E3%83%B3%E5%88%87%E3%82%8C%E3%82%92%E9%98%B2%E6%AD%A2%E3%81%99%E3%82%8B/  
デメリットもありそうだが今のところ大きな問題には遭遇していない. セッション切れがあまりない(と言われている)Colab Proでも必要かどうかは微妙. 
長時間学習する時などには有効かも.  

それからKaggle APIの使い方についてKaggle Datasetsへのアップロード方法について学びがあった.    
手順としては    
(1) ファイル保存用のフォルダを作成  
(2) フォルダにデータファイルを保存  
(3) フォルダに対してinitする(これでmetaファイルが作成される)  
(4) metaファイルを編集  
(5) フォルダごとアップロード  
詳細はこちら:    
https://kaeru-nantoka.hatenablog.com/entry/2020/01/17/015551  




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


#### 2021-04-23
notebooks/localnb001-transformers-ner.ipynbをColab Proで実行しfine-tuned BERTモデルを  
[localnb001-transformers-ner](https://www.kaggle.com/riow1983/localnb001-transformers-ner)にアップロードした.  
なお, この学習済みモデルでinferenceするためのsample-submission.csvのテーブルデータの加工についての実装はまだできていない.  
そこはフロムスクラッチするよりも公開カーネルを利用できないものかとも思っている.  
と思ったが, そのような公開カーネルは今のところなさそうだったので, 自分で実装することにした. 
<br> 
それにしてもColab Pro使いやすい. ネットワークが切れても途中から処理がresumeされるので環境要因に対してもrobustな印象. High memory RAMも35GBの強いやつを引くときもあり. これで環境構築の手間やconflictを気にするストレスを大幅に削減できるのはありがたい. 


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





