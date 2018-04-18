# Generative Adversarial Network

Wasserstein generative adversarial network implementation with Tensorflow.

<img src="https://github.com/WojciechMormul/gan/blob/master/imgs/houses.bmp" width="800">
<img src="https://github.com/WojciechMormul/gan/blob/master/imgs/net.png" width="300">
<img src="https://github.com/WojciechMormul/gan/blob/master/imgs/gen2.png" width="800">
<img src="https://github.com/WojciechMormul/gan/blob/master/imgs/gen_conv_bn.png" width="400">
<img src="https://github.com/WojciechMormul/gan/blob/master/imgs/dis2.png" width="800">
<img src="https://github.com/WojciechMormul/gan/blob/master/imgs/loss.png" width="400">

## Usage
Prepare folder with images of fixed size.
```
git clone https://github.com/WojciechMormul/gan.git
cd gan
python make_record.py --images-path=./data --record-path=./train.record
python gan_train.py
python gan_eval.py
```


