# Kaggle-Coleridge-Initiative

***
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
|kagglenb001_transformers_test|[URL](https://www.kaggle.com/riow1983/kagglenb001-transformers-test)|-|-|使用予定なし|huggingface transformersの簡易メソッド<br>(AutoTokenizer, AutoModelForTokenClassification)<br>を使ったNERタスク練習|
|kagglenb002_NERDA_test|[URL](https://www.kaggle.com/riow1983/kagglenb002-nerda-test)|-|-|使用予定なし|NERDAを使ったNERタスク練習|
|kagglenb003_annotation_data|[URL](https://www.kaggle.com/riow1983/kagglenb003-annotation-data)|[NERタスク用trainデータ](https://www.kaggle.com/shahules/ner-coleridge-initiative)|-|Done|NERDAを使ったNERタスク|
|nb003-annotation-data|URL|NERタスク用trainデータ|[5 Fold CV data](https://www.kaggle.com/riow1983/nb003-annotation-data)|spaCyによるPOS tagging追加作業中|NERDAによるNERタスクは放擲. <br>5 Fold CV dataを作成することが目的.|
|kagglenb004-transformers-ner-inference|[URL](https://www.kaggle.com/riow1983/kagglenb004-transformers-ner-inference)|localnb001によるfine-tuned BERTモデル他|submission.csv(未作成)|保留中|localnb001によるfine-tuneがうまくいっていないためsubmitは保留中|
|kagglenb005-pytorch-BERT-for-NER|[URL](https://www.kaggle.com/riow1983/kagglenb005-pytorch-bert-for-ner)|-|fine-tuned BERT model(未作成)|停止中|公開カーネル中高スコア(LB=0.7)を記録している<br>[kaggle notebook (Coleridge: Matching + BERT NER)](https://www.kaggle.com/tungmphung/coleridge-matching-bert-ner)のtrain側. <br>EPOCHS=1でも９時間以上かかりそう. <br>Colabにpullしてnb005-pytorch-bert-for-nerとして訓練する|
|nb005-pytorch-bert-for-ner|URL|kagglenb007-get-text's output files|fine-tuned BERT model <br> [nb005-pytorch-bert-for-ner-512](https://www.kaggle.com/riow1983/nb005-pytorch-bert-for-ner-512) <br> [nb005-pytorch-bert-for-ner](https://www.kaggle.com/riow1983/nb005-pytorch-bert-for-ner)|EPOCHS>5で訓練完了<br>lossが下がらない原因調査中|epochs\>1でもlossが下がらずLB=0.700<|
|kagglenb006-get-text|[URL](https://www.kaggle.com/riow1983/kagglenb006-get-text)|-|JSONファイルからパースしたtextを新規列として保持する<br>tran/test dataset|Done|Colab側で作業する際, Google Driveに置いたJSONファイルをreadする処理に時間がかかるためKaggle上で実施した|
|kagglenb007-get-text|[URL](https://www.kaggle.com/riow1983/kagglenb007-get-text)|-|JSONファイルからパースしたtextを新規列として保持する<br>tran/test dataset<br>section構造をそのまま保持|Done|Colab側で作業する際, Google Driveに置いたJSONファイルをreadする処理に時間がかかるためKaggle上で実施した|
|kagglenb008-pytorch-bert-for-ner-inference|[URL]()|nb005-pytorch-bert-for-ner|submission.csv|作成中|[kaggle notebook (Coleridge: Matching + BERT NER)](https://www.kaggle.com/tungmphung/coleridge-matching-bert-ner)をcopyしたもの<br>|nb005-pytorch-bert-for-nerのinference側|
|localnb001-transformers-ner|URL|[nb003-annotation-data (5 fold CV data)](https://www.kaggle.com/riow1983/nb003-annotation-data)|fine-tuned BERTモデル|POS taggingを入力に加えて精度向上するか試してみる|ネット上に落ちていたColab notebookを本コンペ用に改造したもの. <br>huggingface pre-trainedモデルのfine-tuned後の保存は成功. <br>PytorchXLAによるTPU使用. <br>fine-tuned BERTモデルはkagglenb004-transformers-ner-inferenceの入力になる.|
|l2knb001-transformers-ner|[URL](https://www.kaggle.com/riow1983/l2knb001-transformers-ner)|nb003-annotation-data (5 fold CV data)|fine-tuned BERTモデル|使用予定なし(チームシェア用)|-|









***
## 参考資料
#### Papers
|name|url|status|comment|
|----|----|----|----|
|Big Bird: Transformers for Longer Sequences|[URL](https://arxiv.org/pdf/2007.14062.pdf)|Reading|Turing completeの意味が分からん|
|Neural Architectures for Named Entity Recognition|[URL](https://arxiv.org/pdf/1603.01360.pdf)|Reading|[arXivTimesで"NER"と検索したら出てきた](https://github.com/arXivTimes/arXivTimes/issues/185)論文.<br>2016年の論文でLSTMベースのNER用モデルの提案.<br>BERT, Transformer系以外のものも見てみようという思い.<br>実装はTheano.|

#### Blogs
|name|url|status|comment|
|----|----|----|----|
|Understanding BigBird's Block Sparse Attention|[URL](https://huggingface.co/blog/big-bird)|Untouched||

#### Documentation / Tutorials
|name|url|status|comment|
|----|----|----|----|
|SAVING AND LOADING MODELS|[URL](https://pytorch.org/tutorials/beginner/saving_loading_models.html)|Reading|PyTorch標準方式のモデルsave方法|
|Source code for pytorch_transformers.tokenization_bert|[URL](https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/tokenization_bert.html)|Done|bert-base-cased tokenizerをKaggle上で使用するためS3レポジトリからwget|
|Huggign Face's notebooks|[URL](https://huggingface.co/transformers/notebooks.html)|Bookmarked|-|
|Fine-tuning a model on a token classification task|[URL](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)|Done|huggingfaceによるNERタスクのチュートリアル.<br>ただしfine-tunedモデルの保存に関する実装はない<br>なお標準的なhuggingface方式では保存したいモデルはアカウントを作ってウェブレポジトリにアップロードするらしい<br>Kaggleから使える？|
|Model sharing and uploading|[URL](https://huggingface.co/transformers/model_sharing.html)|Bookmarked|huggingface方式のモデル保存方法について|

#### GitHub
|name|url|status|comment|
|----|----|----|----|
|how to save and load fine-tuned model?|[URL](https://github.com/huggingface/transformers/issues/7849)|Done|huggingfaceのpre-trainedモデルを<br>fine-tuningしたものをPyTorch標準方式でsaveする方法|

#### Kaggle Notebooks
|name|url|status|comment|
|----|----|----|----|
|Coleridge - Huggingface Question Answering|[URL](https://www.kaggle.com/jamesmcguigan/coleridge-huggingface-question-answering)|Done|QAのtoy example的なやつ. <br>結局こんな精度じゃ話にならない. <br>また事後学習する方法が分からず終い.|
|HuggingFace Tutorial; Custom PyTorch training|[URL](https://www.kaggle.com/moeinshariatnia/simple-distilbert-fine-tuning-0-84-lb)|Bookmarked|huggingfaceのpre-trainedモデルをfine-tuningするも<br>PyTorch標準のsave方式を採用している<br>らしいところは参考になる|
|Bert PyTorch HuggingFace Starter|[URL](https://www.kaggle.com/theoviel/bert-pytorch-huggingface-starter)|Bookmarked|huggignface PyTorchのとても綺麗なコード.<br>参考になるがfine-tuned modelのsave実装はない.|
|[Training] PyTorch-TPU-8-Cores (Ver.21)|[URL](https://www.kaggle.com/joshi98kishan/foldtraining-pytorch-tpu-8-cores/data?scriptVersionId=48061653)|Bookmarked|offlineでPyTorch-XLAインストールスクリプトが有用|
|EDA & Baseline Model|[URL](https://www.kaggle.com/prashansdixit/coleridge-initiative-eda-baseline-model)|Done|dataset_label, dataset_title, cleaned_labelをsetにして<br>existing_labelsにしている|
|data_preparation_ner|[URL](https://www.kaggle.com/shahules/coleridge-initiative-data-to-ner-format)|Done|[shahules/ner-coleridge-initiative](https://www.kaggle.com/shahules/ner-coleridge-initiative)作成コード|


#### Kaggle Datasets
|name|url|status|comment|
|----|----|----|----|
|shahules/ner-coleridge-initiative|[URL](https://www.kaggle.com/shahules/ner-coleridge-initiative)|Bookmarked|NERタスク用のデータセット<br>[ディスカッション](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/230341)に作成コードが紹介されている|
|joshi98kishan/pytorch-xla-setup-script|[URL](https://www.kaggle.com/joshi98kishan/pytorch-xla-setup-script)|Bookmarked|PyTorch-XLAをofflineでインストールするためのスクリプト|

#### Kaggle Discussion
|name|url|status|comment|
|----|----|----|----|
|Data preparation for NER|[URL](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/230341)|Done|Dataset作成コードとTrainデータの実際のデータセット[NER Coleridge Initiative](https://www.kaggle.com/shahules/ner-coleridge-initiative)が<br>Kaggle Datasetにアップされている|


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


