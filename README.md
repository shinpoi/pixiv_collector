Pixiv Collector
=================
#### [中文版点这里 (Chinese version)](#cn)
***
<p id='jp'></p>
## 目次:

1. [説明 & デモ](#jp_1)
2. [使い方](#jp_2)
  * [環境構築](#jp_2_1)
  * [Chainer-デバッグ](#jp_2_2)
  * [実行](#jp_2_3)
3. [自分のネットをトレーニングしよう！](#jp_3)
4. [プロジェクトの仕組み](#jp_4)
  * [ウェブクローラ](#jp_4_1)
  * [畳み込みニューラルネットワーク](#jp_4_2)
5. [その他](#jp_5)

***
<p id='jp_1'> </p>
## 1. 説明 & デモ
このプロジェクトは[Pixiv](http://www.pixiv.net/)の画像作品をスキャン、識別、二項分類、まだ保存するのプログラムです.

プログラムは主に２つの部分で構成されています：  
ひとつはウェブクローラ、Pixiv上で各作品をスキャンと保存するためのプログラムです.  
もうひとつは畳み込みニューラルネットワーク（Convolutional Neural Networks, 以下CNNと略します）に用いる画像分類器、[Chainer](https://github.com/pfnet/chainer)で書きました.

* 分類ルール：
  * グルプ１： 完成度の高い絵（例えば、綺麗な背景を持つ、彩り豊か、線も綺麗など）
  * グルプ２： 漫画，スケッチ，線画、落書き，ちびチャラ，ラブライブ顔，巨乳，大きな尻，足（しかない）など
  * 個人趣味でこの分類ルールを作りました、自分の分類器を作りたい場、[第３節](#jp_3)を参考にしてください.

——————————————————————————————————————————————————————
<p id='jp_2'> </p>
## 2. 使い方

<p id='jp_2_1'> </p>
### 2.1. 環境構築  

**システム**: 開発とテスト共にLinuxです、論理上Windowsにも使えますか*（クローラ部分は win10x64 でも使えると実証しました）*...Linuxを推奨します   
**ハードウェア**: 実行だけなら、月500円の格安VPS（メモリ1G、共有CPU、グラボ無し）でも問題ないです.　自分の分類器を作りたい場、[第３節](#jp_3)を参考にしてください.

#### Liunx上の環境構築：
(*Ubuntu* の場は *Debian* を参考にしてください，コマンドは同じです)

————————————————————————————————————————  
**2.1.1 Python3：**  
Debian: `apt-get install python3 python3-dev python3-all`  
CentOS: [コンパイルインストール](https://docs.python.org/3/using/unix.html?highlight=install)が必要です， Python3.6.0を例にすれば:    
```bash
wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz
tar xf Python-3.6.0.tar.xz
cd Python-3.6.0*
./configure
make
make install

# パッケージをダウンロード　-> 解凍 -> 解凍したファイルに入る -> 環境設定 & チェック -> コンパイル　-> インストール
```

————————————————————————————————————————  
**2.1.2 OpenCV：**  
Debian: [ホムページ](http://docs.opencv.org/3.2.0/d7/d9f/tutorial_linux_install.html)を参考にしてコンパイルインストール  
CentOS: `yum install opencv*`

————————————————————————————————————————  
**（推奨）2.1.3 virtualenvをインストールする**（クリーンな実行環境を維持するため）：  
```bash
pip3 install virtualenv
```

**virtualenv環境を作る：**  
```bash
virtualenv -p python3 --no-site-package PixivCollector  
```

**virtualenv環境に入る：**
```bash
source PixivCollector/bin/active
```

————————————————————————————————————————  
**2.1.4 必要なPythonパッケージをインストールする：**  
```bash
pip install numpy opencv-python chainer requests lxml beautifulsoup4
```

————————————————————————————————————————  
<p id='jp_2_2'> </p>
### 2.2. Chainer Debug
もうひとつちょっと厄介なことがあります...  
このプログラムは`chainer`の`Links.inceptionBN()`関数を使っています，でもChainerいまのヴァージョンにこの関数はbug([issue#1662](https://github.com/pfnet/chainer/pull/1662))を含めています.  
github上の最新ヴァージョンも修復しましたか，pipで公開しているヴァージョンはまだ修復していない，だから自力でchainerのソースコードを編集する必要があります.
```bash
gedit PixivCollector/lib/python3.x(Python3のヴァージョンによって違います)/site-packages/chainer/links/connection/inceptionbn.py
```

```python
-   def __call__(self, x):
-        test = not self.train

+   def __call__(self, x, test=False):
```

————————————————————————————————————————  
<p id='jp_2_3'> </p>  
### 2.3.1 実行
いよいよ実行部分になりました！  

#### コードをダウンロードしよう：  
`git clone https://github.com/shinpoi/pixiv_collector.git`

#### そして：  
1. `setting.py`を編集してください： `PIXIV_ID="----"` と `PIXIV_PW="----"` の`----`部分を自分のPixivアカウントとパスワードで入れ替わってください.  
2. コマンド: `python crawler.py` で実行.

そうしますと、プログラムは昨日の（0:00~12:00で使うなら，一昨日の）日間ランキング前２００枚を分類して、ディレクトリ`./pixiv`に保存します.

————————————————————————————————————————  
### 2.3.2 パラメータ付きの使い方
`crawler.py` ショットとロング二種類の書き方があります、効果は同じです.  
日付を例にして：  
ショット： `-d 20170312`  
ロング： `--date=20170312`  
P.S. 文字列に引用符はいらない.

#### パラメータ:  
* モード選択: `-m --mode` //'rank'が'artist'が必要、スキャンの目標をランキングがユーザーがを設定する. デフォルト値は'rank'.


* `'rank'`モード:  
  1. 時間選択: `-d --date` // ８桁の数値を必要、例えば、２０１７年３月１２日のランキングなら、‘20170312’と入力（引用符いらない），デフォルト値は昨日/一昨日.  
  2. ページ選択: `-p --page` //　1〜１０の整数が必要、１ページの画像idは５０枚， デフォルト値は４.  
  3. ランキング選択: `-c --class` // 'daily'、'weekly'、'monthly' のいずれは必要、日間、週間、月間ランキングを指定する、デフォルト値は'daily'.  


* `'artist'`モード:
  1. uid指定: `-u --uid` // ユーザーのuidが必要、'artist'モードなら必ず入力してください、入力しない場エラーが出ます.
  2. モード指定#2: `-c --class` // 'works'が'bookmarks' が必要、ユーザーの作品をスキャンするか、ブックマークをスキャンするかを選択する、デフォルト値は'works'.

#### クローラだけを使う（分類器がいらないの場）：

* パラメータ`no-classify` を付けてください, そうすると分類器がoffになります.

例えば，私（uid=1941321）のブックマークにある全ての画像を保存したいなら：  
コマンド: `python crawler.py -m artist -u 1941321 --class=bookmarks no-classify`

——————————————————————————————————————————————————————
<p id='jp_3'> </p>
## 3. 自分のネットをトレーニングしよう！
必要環境：メモリが8G以上のNvidiaグラフィックカード（目安: GTX1070+）  
（いきなり酷い要求？！  

#### 環境構築：  
[CUDA](https://developer.nvidia.com/cuda-downloads)をインストールしてください（ヴァージョンは7.5と8のどっちでも問題ないです）   
[cuDNN](https://developer.nvidia.com/cudnn)をインストールしてください  
cuDNNインストールしたら，chainerを入れなおす:  `(PixivCollector)$ pip install chainer`

————————————————————————————————————————  
#### トレーニング手順:

#### 3.1 トレーニングデータを用意する：  
画像データの置く場所は`./dataset`中の`train_positive`, `train_negative`, `test_positive`, `test_positive`四つのファイル.  
`train*`の中はトレーニングデータ，positiveとnegative各２００枚以上、まだ５０の整数倍の枚数の画像を勧めします。   
`test*`の中はテストデータ，**トレーニングに影響がない**，分類器の性能を評価するだけ使います，**トレーニングデータと違うデータを**、少しだけ入れいいです（各５０枚を勧め）  

簡単な人物と非人物の識別から，抽象的な'好き'と'嫌い'，どんどん試してください！（でも実際的に、特徴が具体的な方が分類効果がいいです  
画像の色は分類効果に影響しません，全てのデータはモノクロ画像として解析していますから，理由は[4.2節](#cn_4_2)で説明します。  

#### 3.2 トレーニングデータの前処理：  
`python init_data.py`

#### 3.3 元ネットのバックアップ：  
`cpu_model.npz`をバックアップしてください.

#### 3.4 トレーニング：  
`python train_model.py`  
トレーニング終わったら、新しい`cpu_model.npz`ファイルが生成されます.

——————————————————————————————————————————————————————
<p id='jp_4'> </p>
## 4. プロジェクトの仕組み（開発者向け）
構造は下記のとうりとなります：  

*pixiv_collector*  
├── **setting.py** ---- 各ファイル共通のパラメータ，まだ、アカウントとパスワードを設定する場所  
├── **init_dataset.py**  ---- サンプルの前処理プログラム（サンプルを133x133のグレースケール画像として保存する）  
├── **train_model.py** ---- CNNをトレーニングするプログラム  
├── **cpu_model.npz**  ---- トレーニング完成したCNNのパラメータ  
├── **predictor.py**  ---- 画像分類のプログラム  
├── **crawler.py**  ---- pixivクローラ  
├── **demo_creator.py**  ---- デモサイトを生成するプログラム  
├── **src**    
　　　├── **\__init__.py**  ---- 空っぽ，pythonを使うため必要なファイル   
　　　├── **model.py**  ---- ニューラルネットワーク構造を定義するファイル   
　　　├── **index.html** ----- デモサイトホムページのテンプレート  
　　　└── **template.html**   ----- デモサイト実演ページのテンプレート


————————————————————————————————————————  
<p id='jp_4_1'> </p>
### 4.1 クローラ
`setting.py`  
実装した、でも使っていない機能：
* 評価（pixivでとある作品を評価する）`score()`  
（このプロジェクト最初の目的は、プログラムに自分はどんなの作品が'好き'を教えて、そしてクローラをpixiv上自由に駆け回させて、自分が好きな作品を集める、ついてに満点を付けることです. でも'好き'という感覚はやはり曖昧過ぎて、分類の効果はあんまりいいではなっかた. プログラムを修正しつつ、今の形になりました.　まぁ、今後努力の目標はプログラムを私の'好き'と'嫌い'の作品が分かるように改良することです~）

————————————————————————————————————————  
<p id='jp_4_2'> </p>
### 4.2 畳み込みニューラルネットワーク
使うモデルはGoogle Inception v2の簡素化ヴァージョン. モデル定義は`model.py`にあります.  
>**参考文献:**   
>https://arxiv.org/abs/1502.03167  
>https://arxiv.org/abs/1512.00567  

* **どうして簡素化？**  
元々Inception v2の完全ヴァージョンを使うつもりなんですか………グラボメモリ不足のエラーが出てきました.  

* **どうしてモノクロ画像？**  
メモリ不足の問題を解決するため、いろいろ試行錯誤をしました.  
そして、画像のサイズをある程度縮って、モノクロ画像と変わるとしでも、分類器の性能は殆ど変りませんの結果が得られました.  
様々な調整の後、モデルは今の形になりました.

——————————————————————————————————————————————————————
<p id='jp_5'> </p>
## 5. 最後
実用にはまだまだです.
正解率高い（グルプ１と分類される作品は確か皆完成率高い）が、再現率が低い（完成率高い作品はグルプ２の中にもたくさんあります）  

今の目標は正解率を維持するまま再現率を上がることです.
遠い目標は当然、最初の目的　——　プログラムに自分はどんなの作品が'好き'を教えることです.

自分の訓練用データ、ネットパラメータの更新など、ここでおけます、ネットの構造が変化しない限り、`cpu_model.npz`をGithubで更新しないようにしています.

以上です.

***
***
Pixiv Collector
=================
#### [日本語はこちらへ (Japanese version)](#jp)
***
<p id='cn'></p>
### 目录:

1. [说明 & demo](#cn_1)
2. [使用](#cn_2)
  * [环境构筑](#cn_2_1)
  * [chainer-debug](#cn_2_2)
  * [运行](#cn_2_3)
3. [训练自己的网络](#cn_3)
4. [文件结构](#cn_4)
  * [爬虫](#cn_4_1)
  * [神经网络](#cn_4_2)
5. [结尾 & 计划](#cn_5)

***

<p id='cn_1'> </p>
### 1. 说明 & demo
一个按指定规则扫描pixiv上的图片，并按一定规律自动分类的程序.  
* 规律：
  * 正类： 完成度比较高的图（比如丰富的背景，流畅的线条，有上色什么的）
  * 负类： 漫画，线稿，草稿，Q版画，ll脸，巨乳，巨臀，只有腿的图……等等
  * 目前按个人喜好分为这两类，想定制自己的分类规则的话，参考[第三节](#cn_3).

程序由两部分组成：一部分是pixiv爬虫，用来扫描并下载排行榜信息和图片；另一部分是一个基于卷积神经网络（CNN）的分类器，由chainer实现。

————————————————————————————————————————————————————
<p id='cn_2'> </p>
### 2. 使用

——————————————————————————————————————————
<p id='cn_2_1'> </p>
#### 2.1. 环境配置
开发和测试的环境为皆Liunx。Windows上理论可以使用，有兴趣可以自行研究。。。  
（爬虫部分在 *win10x64* 下测试可以正常使用）

Liunx下的环境配置：  
(*Ubuntu* 参照 *Debian*，命令相同)

**Python3：**  
Debian: `apt install python3 python3-dev python3-all`  
CentOS: 需要[编译安装](https://docs.python.org/3/using/unix.html?highlight=install)， 大致步骤:    
```bash
wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz
tar xf Python-3.6.0.tar.xz
cd Python-3.6.0*
./configure
make
make install
```

**OpenCV：**  
Debian: 参照官网[编译安装](http://docs.opencv.org/3.2.0/d7/d9f/tutorial_linux_install.html)  
CentOS: `yum install opencv*`

**（推荐）安装virtualenv**（用于保持环境纯净）：  
```bash
pip3 install virtualenv
```

**进入virtualenv环境：**  
```bash
virtualenv -p python3 --no-site-package PixivCollector  
source PixivCollector/bin/active
```

**安装相关的Python包：**  
```bash
pip install numpy opencv-python chainer requests lxml beautifulsoup4
```

——————————————————————————————————————————
<p id='cn_2_2'> </p>
#### 2.2. Chainer Debug
还有个麻烦的地方...  
这个程序使用到了`chainer`的`Links.inceptionBN()`函数，而这个函数在当前版本有bug([issue#1662](https://github.com/pfnet/chainer/pull/1662)).  
虽然github上的最新版已经修正了，但是pip的发行版还没有跟上，所以我们需要手动修改一下chainer的代码.
```bash
gedit PixivCollector/lib/python3.x(视你的Python版本而定)/site-packages/chainer/links/connection/inceptionbn.py
```

```python
-   def __call__(self, x):
-        test = not self.train

+   def __call__(self, x, test=False):
```

——————————————————————————————————————————
<p id='cn_2_3'> </p>  
#### 2.3. 运行
激动人性的步骤！  

下载本程序：  
`git clone https://github.com/shinpoi/pixiv_collector.git`

然后：  
1. 打开`setting.py`，设置 `PIXIV_ID=` 和 `PIXIV_PW=` 为自己的pixiv帐号和密码.  
2. 执行 `python crawler.py`

程序会扫描并下载昨天（如果你在0:00~12:00使用的话，扫描前天）的日间排行榜前200张图片，分类保存在`./pixiv`目录下.

——————————————————————————————————————————
<p id='cn_2_4'> </p>
#### 2.4. 进阶使用
`crawler.py` 接受长短参数 —— 两种参数写法不同但效果相同。  
以日期为例：  
短参数： `-d 20170312`  
长参数： `--date=20170312`  
字符串不用加引号.

参数：  
* 指定模式: `-m --mode` // 接受'rank'或者'artist'，判断扫描排行榜还是扫描用户. 默认为'rank'.


* rank模式下:  
  1. 指定时间: `-d --date` // 接受一个时间字符串作为参数，比如: ‘20170312’，默认为昨天.  
  2. 指定页数: `-p --page` //接受一个1～10的整数作为参数，1页50张图，默认为四页.  
  3. 指定榜单: `-c --class` // 接受 'daily', 'weekly', 'monthly' 中的一个，默认为'daily'.  


* artist模式下
  1. 指定uid: `-u --uid` // 接受用户的数字id，不输入会报错
  2. 指定模式: `-c --class` // 接受 'works', 'bookmarks' 中的一个，默认为'works'.

仅仅作为爬虫使用：

* 参数`no-classify` (不需要值，不加`--`), 关闭分类器.

举例，比如我想下载我自己 (uid=1941321) 所有收藏的图片的话，可以输入：  
输入: `python crawler.py -m artist -u 1941321 --class=bookmarks no-classify`

————————————————————————————————————————————————————
<p id='cn_3'> </p>
### 3. 训练自己的网络
要求：一张显存8G在以上的Nvidia显卡（GTX1070+）.

环境配置：  
安装 [CUDA](https://developer.nvidia.com/cuda-downloads)（7.5和8都行）   
安装 [cuDNN](https://developer.nvidia.com/cudnn)  
装好CuDNN后，重装 chainer:  `(PixivCollector)$ pip install chainer`

准备训练数据：
将数据分别放入`dataset`文件夹下的train_positive, train_negative, test_positive, test_positive里.  
`train*`里为正负训练数据，推荐各200张以上，并为50的整数倍。   
`test*`为测试数据，**不影响训练结果**，放少量测试就行，**不能和train里的重复**（推荐正负各50张）  

这个网络是个简单的图像二分类器，要针对哪两个属性分类请自由发挥，从最实际的分辨人物和非人物，到抽象的比如喜欢和不喜欢，都可以试试……当然差异越具体分类效果越好。  
另外颜色不会影响分类，样本都被预处理为灰度图了，原因在4.2节说明。  

初始化训练样本：  
`python init_data.py`

备份原网络：  
给`cpu_model.npz`做个备份

训练：  
`python train_model.py`  
训练完成后新的`cpu_model.npz`会覆盖目录下的原文件.

————————————————————————————————————————————————————
<p id='cn_4'> </p>
### 4. 文件结构（开发者向）
总体结构如图：  

*pixiv_collector*  
├── **setting.py** ---- 一些各模块通用的设定，也是设置帐号密码的地方  
├── **init_dataset.py**  ---- 初始化样本（把图片压缩为133x133的单色图， 原因见4.2节）  
├── **predictor.py**  ---- 分类器，用训练完成的`cpu_model.npz`进行分类  
├── **train_model.py** ---- 训练神经网络的主程序  
├── **cpu_model.npz**  ---- 训练好的网络参数文件  
├── **crawler.py**  ---- pixiv爬虫  
├── **demo_creator.py**  ---- 生成demo网页的程序  
├── **src**    
　　　├── **\__init__.py**  ---- 空文件，python自身需要  
　　　├── **model.py**  ---- 神经网络结构的定义文件  
　　　├── **index.html** ----- demo网页主页的模板  
　　　└── **template.html**   ----- demo网页展示页面的模板


<p id='cn_4_1'> </p>
#### 4.1 爬虫
`setting.py`  
实装了但没使用的功能：
* 打分  
（这个项目最初的目的是分类出我喜欢的图，然后让爬虫在pixiv上随机扫描，并收集我可能喜欢的图，然后给正类评分。但喜欢这个感觉实在是太玄学了，分类效果一直不太理想，所以渐渐做成了现在这样。当然，努力的目标还是让计算机理解我‘喜欢’什么样的图。）

<p id='cn_4_2'> </p>
#### 4.2 神经网络
使用的网络模型为Google Inception v2的简化型. 具体可以查看`model.py`，对CNN有了解的话应该很容易看懂.  
>**参考这两篇论文:**   
>https://arxiv.org/abs/1502.03167  
>https://arxiv.org/abs/1512.00567  

**为什么简化？**  
原本我用的是完整版的v2网络…………然后显存爆了  


调试的过程中试着降低样本的尺寸和保存为单色，发现分类效果几乎没变化（甚至还略有提高 <-- 实验次数不多，也可能是偶然）
于是最终选用了这个尺寸。

————————————————————————————————————————————————————
<p id='cn_5'> </p>
### 5. 结尾 & 计划
第一次做神经网络在图像识别方面的应用，也算是自己第一个大点的程序方面的项目……不成熟的地方有很多，样本也太多时间没好好搜集……  
以后会慢慢完善的（大概（相信不会坑（真的！  
花了一个多星期没日没夜训练网络 & 写程序，成型的时候还是挺有成就感（虽然完全不实用wwwww  

附: 每次更新网络参数都上传git的话太浪费了，新训练的网络我会放在自己的网站上，有兴趣可以下下来试试.  
（暂没开放，下次改版时附上链接）
