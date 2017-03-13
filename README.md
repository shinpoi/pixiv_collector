Pixiv Collector
=================

***
<span id='cn'></span>
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

[demo](http://demo.aoi-lucrio.org)  
（展示了每日排行榜前200张图的分类结果。 展示用，一个月后自行删除）

<p id='cn_2'> </p>
### 2. 使用

<span id='cn_2_1'> </span>
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

<span id='cn_2_2'> </span>
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

<span id='cn_2_3'> </span>  
#### 2.3. 运行
激动人性的步骤！  

下载本程序：  
`git clone https://github.com/shinpoi/pixiv_collector.git`

然后：  
1. 打开`setting.py`，设置 `PIXIV_ID=` 和 `PIXIV_PW=` 为自己的pixiv帐号和密码.  
2. 执行 `python crawler.py`

程序会扫描并下载昨天（如果你在0:00~12:00使用的话，扫描前天）的日间排行榜前200张图片，分类保存在`./pixiv`目录下.

<span id='cn_2_4'> </span>
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

<span id='cn_3'> </span>
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

<span id='cn_4'> </span>
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


<span id='cn_4_1'> </span>
#### 4.1 爬虫
`setting.py`  
实装了但没使用的功能：
* 打分  
（这个项目最初的目的是分类出我喜欢的图，然后让爬虫在pixiv上随机扫描，并收集我可能喜欢的图，然后给正类评分。但喜欢这个感觉实在是太玄学了，分类效果一直不太理想，所以渐渐做成了现在这样。当然，努力的目标还是让计算机理解我‘喜欢’什么样的图。）

<span id='cn_4_2'> </span>
#### 4.2 神经网络
使用的网络模型为Google Inception v2的简化型. 具体可以查看`model.py`，对CNN有了解的话应该很容易看懂.  
>**参考这两篇论文:**   
>https://arxiv.org/abs/1502.03167  
>https://arxiv.org/abs/1512.00567  

**为什么简化？**  
原本我用的是完整版的v2网络…………然后显存爆了  


调试的过程中试着降低样本的尺寸和保存为单色，发现分类效果几乎没变化（甚至还略有提高 <-- 实验次数不多，也可能是偶然）
于是最终选用了这个尺寸。

<span id='cn_5'> </span>
#### 5. 结尾 & 计划
第一次做神经网络在图像识别方面的应用，也算是自己第一个大点的程序方面的项目……不成熟的地方有很多，样本也太多时间没好好搜集……  
以后会慢慢完善的（大概（相信不会坑（真的！  
花了一个多星期没日没夜训练网络 & 写程序，成型的时候还是挺有成就感（虽然完全不实用wwwww  

附: 每次更新网络参数都上传git的话太浪费了，新训练的网络我会放在自己的网站上，有兴趣可以下下来试试.  
（暂没开放，下次改版时附上链接）
