# CG-GAN(CVPR2022)

This is the official PyTorch implementation of the CVPR 2022 paper: "Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator".

![](img/pipeline.png)

## Requirements

(Welcome to develop CG-GAN together.)

We recommend you to use [Anaconda](https://www.anaconda.com/) to manage your libraries.

- [Python](https://www.python.org/) 3.6* 
- [PyTorch](https://pytorch.org/) 1.0.* 
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/#)
- [LMDB](https://pypi.org/project/lmdb/)
- [matplotlib](https://pypi.org/project/matplotlib/)

## Data Preparation
Please convert your own dataset to **LMDB** format by using the tool ``lmdb_maker.py`` (run in **Python 2.7**, you can upgrade it). 

Both the char(text) label, the radical list and the corresponding writer ID are required for every text image. 

Please prepare the **TTF font** and **corpus** for the rendering of printed style images.

For Chinese font generation task, we recommend you to use [思源宋体](https://github.com/adobe-fonts/source-han-serif.git) as the source font, download it and put it into **data/font** folder. You can download the target fonts from [方正字库](https://www.foundertype.com/index.php/FindFont/index) for making your own dataset.

For handwritten word synthesis task, please down the [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) dataset and then convert it to LMDB format. You can also download the training and testing datasets prepared by us. 

- [BaiduCloud (training datasets and testing datasets in **LMDB** format)](https://pan.baidu.com/s/1t8PhGtTvS6xUX2iCRj3w3w), password: 1exa
- [Google Drive (training datasets and testing datasets in **LMDB** format)](https://drive.google.com/drive/folders/1uFuVwEaC6IXBscCqaDFkreks8_H8V48j?usp=sharing)

## Training

### Chinese font generation task 

Modify the **dataRoot** , **ttfRoot** and **corpusRoot** in `scripts/train_character.sh`as your settings.

```bash
  --dataroot data/path to dataset \
  --ttfRoot data/font \
  --corpusRoot data/path to corpus \
```

Train your model, run

```bash
	 sh scripts/train_character.sh
```

### Handwritten word synthesis task 

Modify the **dataRoot** , **ttfRoot** and **corpusRoot** in `scripts/train_handwritten.sh`as your settings.

```bash
  --dataroot data_iam/train_IAM \
  --ttfRoot data_iam/fonts_iam \
  --corpusRoot data_iam/seen_char.txt \
```

Train your model, run

```bash
 sh scripts/train_handwritten.sh
```

## Testing

### Chinese font generation task 

test your model, run

```bash
 sh scripts/test_character.sh
```

### Handwritten word synthesis task 

test your model, run

```bash
 sh scripts/test_handwritten.sh
```

## Citation
If our paper helps your research, please cite it in your publication(s):
```
@article{cluo2019moran,
  author    = {Yuxin Kong, Canjie Luo, Weihong Ma, Qiyuan Zhu, Shenggao Zhu, Nicholas Yuan, Lianwen Jin},
  title     = {Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator},
  year      = {2022},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  publisher = {IEEE}
}
```
