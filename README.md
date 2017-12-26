# Generative Adversarial Network

Wasserstein generative adversarial network implementation with Tensorflow.

<img src="https://s14.postimg.org/7vfj7jj5d/houses.png" width="800">
<img src="https://s14.postimg.org/gxzf0we41/net.png" width="300">
<img src="https://s14.postimg.org/t14qo49n5/batch.png" width="400">
<img src="https://s14.postimg.org/j3tpvc4ox/gen.png" width="800">
<img src="https://s14.postimg.org/gnwhuwz1t/gen2.png" width="800">
<img src="https://s14.postimg.org/i2y2k2fox/gen_conv_bn.png" width="400">
<img src="https://s14.postimg.org/85n1r25j5/dis.png" width="800">
<img src="https://s14.postimg.org/x06jklvcx/dis2.png" width="800">
<img src="https://s14.postimg.org/5pl8ctpwx/loss.png" width="400">
<img src="https://s14.postimg.org/bdrj3rrpd/optimizer.png" width="400">

## Usage
Prepare folder with images of fixed size.
```
git clone https://github.com/WojciechMormul/gan.git
cd gan
python make_record.py --images-path=./data --record-path=./train.record
python gan_train.py
python gan_eval.py
```


