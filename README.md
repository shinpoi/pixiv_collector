Pixiv Collector
=================
***
## 1. 説明 & デモ  


**機能**: 毎日とあるサイトの日間ランキングに乗ってる画像作品をサーバーへダウンロードして，分類する．そして分類結果を展示するためのwebページを生成する．  
\* **対象サイトのクローラ防止対策により, クローラーもう正常動作できません**

プログラムは主に２つの部分で構成されています：  
ひとつはウェブクローラ、作品をスキャンと保存するためのプログラムです.  
もうひとつは[CNN](https://ja.wikipedia.org/wiki/%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)（Convolutional Neural Networks）に用いる画像分類器、[Chainer](https://github.com/pfnet/chainer)で書きました.

* 分類ルール：
  * グルプ１： 完成度の高い絵（綺麗な背景を持つ、線が綺麗など）
  * グルプ２： 漫画，スケッチ，線画、落書き，ちびチャラなど  

## 2. 使い方
### 2.1. 環境

**システム**: Linux or Mac (Windos未検証)  
```bash
# package
pip install -r pip_req.txt

# model
wget -O dataset/cpu_model.npz http://pc.aoi-lucario.org/cpu_model.npz --no-check-certificate
```

### 2.2 実行
```bash
export PYTHONPATH=`pwd`/src

# 分類だけ
python src/main_classifier.py ${your_images_folder}

# クローラ --> 分類 --> 展示html作成
# !! pixivのクローラ対策より，今使用不可になってます !!
python src/main.py ${your_images_folder}
```

## 3. モデルトレーニング

#### 環境：  
[CUDA](https://developer.nvidia.com/cuda-downloads)
[cuDNN](https://developer.nvidia.com/cudnn)
cuDNNインストール完成後，chainerとcupyを入れなおす

#### CNNのトレーニング：  
`nn_model/trainer.py`を参照  
トレーニング終わったら、新しい`cpu_model.npz`ファイルが生成されます