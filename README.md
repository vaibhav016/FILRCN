<h1 align="center">
<p>Feature integration in acoustic models using Low Rank Spectro-Temporal Decomposition in Convolutional Nets :microphone:</p>
    <img src="https://github.com/vaibhav016/FILRCN/blob/main/LRCNN.png" width="500" height="300" />

<p align="center">
<a href="https://github.com/vaibhav016/TensorFlowASR/blob/main/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/TensorSpeech/TensorFlowASR?logo=apache&logoColor=green">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.4.1-orange?logo=tensorflow">
<a href="https://pypi.org/project/TensorFlowASR/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/TensorFlowASR?color=%234285F4&label=release&logo=pypi&logoColor=%234285F4">
</a>
</p>
</h1>

## What's New?

- (05/15/2021) Trained vanilla Contextnet over Librispeech Dataset [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191)
- (05/16/2021) Generated Loss Landscapes for the trained models, see [demo_loss](./examples/contextnet/contextnet_visualisation/)
- (05/20/2020) Trained a Low Rank Decomposition based Deep Net with wave input [Low rank decomposition model](http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf)
- (06/7/2020)  Generated Integrated Gradients for trained models [Keras Integrated Gradients documentation](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)


## Table of Contents

<!-- TOC -->

- [What's New?](#whats-new)
- [Table of Contents](#table-of-contents)
- [Publications](#publications)
- [Installation](#installation)
  - [Installing from source](#installing-from-source)
  - [Running in a container](#running-in-a-container)
- [Setup training and testing](#setup-training-and-testing)
- [Features Extraction](#features-extraction)
- [Augmentations](#augmentations)
- [Training & Testing Tutorial](#training--testing-tutorial)
- [Loss landscape visualisation and gradient attribution](#loss-landscape-visualisation-and-gradient-attribution)
  - [Gradient_Visualisation_Notebook](#gradient-visualisation-notebook)
  - [Loss Lists Notebook](#loss-lists-notebook)
- [English Dataset](#english)
- [References & Credits](#references--credits)
- [Contact](#contact)

<!-- /TOC -->

### Publications

- **ContextNet** (Reference: [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191))
  See [examples/contextnet](./examples/contextnet)
- **Raw Wwaveform Based CNN Through LOW-RANK Spectro-Temporal Decoupling ** (Reference: [http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf](http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf))
  See [tensorflow_asr/models/encoders](./tensorflow_asr/models/encoders)
    

### Installing from source

```bash
git clone https://github.com/vaibhav016/TensorFlowASR.git
cd TensorFlowASR
python setup.py build
python setup.py install
```
### Running in a container

```bash
docker-compose up -d
```

## Setup training and testing

- For _mixed precision training_, use flag `--mxp` when running python scripts from [examples](./examples)

- For _enabling XLA_, run `TF_XLA_FLAGS=--tf_xla_auto_jit=2 python3 $path_to_py_script`)

- For _hiding warnings_, run `export TF_CPP_MIN_LOG_LEVEL=2` before running any examples

## Features Extraction

See [features_extraction](./tensorflow_asr/featurizers/README.md)

## Augmentations

See [augmentations](./tensorflow_asr/augmentations/README.md)

## Training & Testing Tutorial

1. Define config YAML file, see the `config.yml` files in the [example folder](./examples/contextnet) for reference (you can copy and modify values such as parameters, paths, etc.. to match your local machine configuration)
2. Download your corpus (a.k.a datasets) and run `download_links.sh`[scripts folder](./scripts) to download files  For more detail, see [datasets](./tensorflow_asr/datasets/README.md). **Note:** Make sure your data contain only characters in your language, for example, english has `a` to `z` and `'`. **Do not use `cache` if your dataset size is not fit in the RAM**.
3. [Optional] Generate TFRecords to use `tf.data.TFRecordDataset` for better performance by using the script [create_tfrecords.py](./scripts/create_tfrecords.py)
4. Create vocabulary file (characters or subwords/wordpieces) by defining `language.characters`, using the scripts [generate_vocab_subwords.py](./scripts/generate_vocab_subwords.py) or [generate_vocab_sentencepiece.py](./scripts/generate_vocab_sentencepiece.py). There're predefined ones in [vocabularies](./vocabularies)
5. [Optional] Generate metadata file for your dataset by using script [generate_metadata.py](./scripts/generate_metadata.py). This metadata file contains maximum lengths calculated with your `config.yml` and total number of elements in each dataset, for static shape training and precalculated steps per epoch.
6. run `create_transcripts_from_data.sh` from [scrpts folder](./scripts) to generate .tsv files(the format in which the input is given is .tsv)
6. For training, see `train.py` files in the [example folder](./examples) to see the options
7. For testing, see `test.py` files in the [example folder](./examples) to see the options. 


## Loss landscape visualisation and gradient attribution

For visualisations, we have two kinds of scripts.
`cd examples/contextnet/contextnet_visualisation`
1) for loss landscapes, cd into context_visualisation/loss_landscape_visualisation.
    1) run generate_lists.py(This generates the loss and accuracy lists)
    2) now run plot_loss.py (From those lists, images are drawn both 2d and 3d)
    3) now run video_create.py(It sews all the images into a single video)
2) For gradient visualisation, 
    1) run integrated_grad_vis.py, which will generate the integrated gradients for all the trained models
    2) then run plot_gradients.py
    3) Finally run video_create.py
    
    
    
### Gradient_Visualisation_Notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/vaibhav016/FILRCN/blob/main/contextnet/contextnet_visualisation/gradient_visualisation/gradient_visualisation.ipynb)
### Loss Lists Notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/vaibhav016/FILRCN/blob/main/contextnet/contextnet_visualisation/loss_landscape_visualisation/generate_loss_landscape.ipynb)

### For loss landscape, go to [drive](https://drive.google.com/file/d/1rYCHvoJGesCQZhpyuLAjNDNfQkbyE7nR/view?usp=sharing)
### For gradient attribution, go to [drive](https://drive.google.com/file/d/1Smw05OEhrptbitom-lOUh7E9LjjAb-cu/view?usp=sharing)

<h4 align="centre"> 
    <p align="centre" > gradients attribution</p>
    <img src="https://github.com/vaibhav016/FILRCN/blob/main/gradient_gf.gif" width="500" height="300" />
</h4>

<h4 align="centre"> 
    <p align="centre">  loss landscape </p> 
    <img src="https://github.com/vaibhav016/FILRCN/blob/main/loss.gif" width="500" height="300" />
</h4>


### English Dataset

|   **Name**   |                             **Source**                             | **Hours** |
| :----------: | :----------------------------------------------------------------: | :-------: |
| LibriSpeech  |              [LibriSpeech](http://www.openslr.org/12)              |   970h    |

## References & Credits

1. [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR)
2. [Loss landscape visualisation](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-3/blob/master/Loss%20LandScape/1.1.%20Relu%20no%20normalization%20.ipynb)
3. [Keras Integrated Gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)

## Contact

Vaibhav Singh __(vaibhav.singh@nyu.edu)__

Dr Vinayak Abrol __(abrol@iiitd.ac.in)__
