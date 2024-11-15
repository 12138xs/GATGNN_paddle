# GATGNN: Global Attention Graph Neural Network

This software package implements our GATGNN for improved inorganic materials' property prediction. This is the `paddlepaddle` implementation repository.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Usage for Custom Property & Custom Dataset](#usage-for-custom-property--custom-dataset)
- [Official PyTorch Version](#official-pytorch-version)

## Introduction

The package provides three major functions:

- Train a GATGNN model for any of the seven properties referenced.
- Evaluate the performance of a trained GATGNN model on these properties.
- Predict the property of a given material using its CIF file.

The following paper describes the details of our framework:
[Global Attention Based Graph Convolutional Neural Networks for Improved Materials Property Prediction](https://arxiv.org/pdf/2003.13379.pdf)

![GATGNN](docs/front-pic.png)

## Installation

### Required Packages

```bash
conda create -n paddle39 python=3.9
conda activate paddle39

pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
pip install pybind11 pyparsing pymatgen

git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
cd GammaGL
python setup.py install
```

### GammaGL Modification

Modify the following code in GammaGL:

Original Implementation:

```python
def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids) + 1
    if use_ext:
        return paddle_ext.unsorted_segment_sum(x, segment_ids, num_segments)
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    output = pd.incubate.segment_sum(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]], dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output
```

New Implementation:

```python
def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids) + 1
    if use_ext:
        return paddle_ext.unsorted_segment_sum(x, segment_ids, num_segments)
    
    idx_ = pd.argsort(segment_ids)
    segment_ids = pd.gather(segment_ids, idx_)
    if len(x.shape) > 2:
        res = []
        for i in range(x.shape[0]):
            x_i = pd.gather(x[i], idx_)
            output_i = pd.incubate.segment_sum(x_i, segment_ids)
            if output_i.shape[0] == num_segments:
                res.append(output_i)
            else:
                init_output = pd.zeros(shape=[num_segments, x_i.shape[1]], dtype=output_i.dtype)
                idx = pd.arange(output_i.shape[0])
                final_output = _scatter(init_output, idx, output_i)
                res.append(final_output)
        return pd.stack(res)
    else:
        x = pd.gather(x, idx_)
        output = pd.incubate.segment_sum(x, segment_ids)
        if output.shape[0] == num_segments:
            return output
        else:
            init_output = pd.zeros(shape=[num_segments, x.shape[1]], dtype=output.dtype)
            idx = pd.arange(output.shape[0])
            final_output = _scatter(init_output, idx, output)
            return final_output
```

## Dataset

1. Download the compressed file of our dataset using [this link](https://widgets.figshare.com/articles/12522524/embed?show_title=1).
2. Unzip its content (a directory named 'DATA').
3. Move the DATA directory into your GATGNN directory, so the path `GATGNN/DATA` now exists.

## Usage

### Training a New Model

Once all the aforementioned requirements are satisfied, you can train a new GATGNN by running `train.py` in the terminal along with the appropriate flags. At a minimum, use `--property` to specify the property and `--data_src` to identify the dataset (CGCNN or MEGNET). The details can be found in `runtrain.sh`.

- Example 1: Train a model on the bulk-modulus property using the CGCNN dataset.
  ```bash
  python train.py --property bulk-modulus --data_src CGCNN
  ```
- Example 2: Train a model on the shear-modulus property using the MEGNET dataset.
  ```bash
  python train.py --property shear-modulus --data_src MEGNET
  ```
- Example 3: Train a model with 5 layers on the bulk-modulus property using the CGCNN dataset and the global attention technique of fixed cluster unpooling (GI M-2).
  ```bash
  python train.py --property bulk-modulus --data_src CGCNN --num_layers 5 --global_attention cluster --cluster_option fixed
  ```

The trained model will be automatically saved under the TRAINED directory. *Pay attention to the flags used as they will be needed again to evaluate the model.*

### Evaluating the Performance of a Trained Model

After training a GATGNN, evaluate its performance using `evaluate.py` in the terminal exactly as `train.py`. *It is IMPORTANT to run `evaluate.py` with the exact same flags used during the training.* The details can be found in `runtest.sh`.

- Example 1: Evaluate the performance of a model trained on the bulk-modulus property using the CGCNN dataset.
  ```bash
  python evaluate.py --property bulk-modulus --data_src CGCNN
  ```
- Example 2: Evaluate the performance of a model trained on the shear-modulus property using the MEGNET dataset.
  ```bash
  python evaluate.py --property shear-modulus --data_src MEGNET
  ```
- Example 3: Evaluate the performance of a model trained with 5 layers on the bulk-modulus property using the CGCNN dataset and the global attention technique of fixed cluster unpooling (GI M-2).
  ```bash
  python evaluate.py --property bulk-modulus --data_src CGCNN --num_layers 5 --global_attention cluster --cluster_option fixed
  ```

### Predicting the Property of a Single Inorganic Material Using Its CIF File

Using a trained model, you can predict the property of a single inorganic material using its CIF file. Follow these steps:

1. Place your CIF file inside the directory `DATA/prediction-directory/`.
2. Run `predict.py` similarly to `evaluate.py`, with the addition of the `--to_predict` flag to specify the name of the CIF file.

- Example 1: Predict the bulk-modulus property of a material named mp-1 using the CGCNN graph constructing specifications.
  ```bash
  python predict.py --property bulk-modulus --data_src CGCNN --to_predict mp-1
  ```
- Example 2: Predict the shear-modulus property of a material named mp-1 using the MEGNET graph constructing specifications.
  ```bash
  python predict.py --property shear-modulus --data_src MEGNET --to_predict mp-1
  ```

## Usage for Custom Property & Custom Dataset

Once you've downloaded and unzipped the dataset, follow these steps:

1. Place all of your CIF files in the directory `DATA/CIF-DATA_NEW`.
2. Format your CSV property dataset to have only two columns (ID, value). Your file should look like any of our CSV files located in the directory `DATA/properties-reference/`.
3. Once your CSV property dataset is correctly formatted, rename your file as `newproperty.csv` and place it in the `DATA/properties_reference/` directory.

With these steps complete, you are ready to use our GATGNN on your own dataset. To either [train](#usage), [evaluate](#usage), or even [predict](#usage) your own property, refer to the instructions listed in the [Usage](#usage) section. Use `new-property`, `NEW`, and any ratio (like 0.75) as values for the `--property` flag, `--data_src` flag, and `--train_size` flag. Examples are provided below:

- Example 1: Train a new GATGNN on your property.
  ```bash
  python train.py --property new-property --data_src NEW --train_size 0.8
  ```
- Example 2: Evaluate the performance of a model trained on your property.
  ```bash
  python evaluate.py --property new-property --data_src NEW --train_size 0.8
  ```
- Example 3: Predict the value of your property for a material named mp-1.
  ```bash
  python predict.py --property new-property --data_src NEW --to_predict mp-1
  ```

## Official PyTorch Version

For the official PyTorch version of GATGNN, visit the [GATGNN repository](https://github.com/superlouis/GATGNN).

[Machine Learning and Evolution Laboratory](http://mleg.cse.sc.edu)  
Department of Computer Science and Engineering


How to cite:
```bibtex
Louis, Steph-Yves, Yong Zhao, Alireza Nasiri, Xiran Wang, Yuqi Song, Fei Liu, and Jianjun Hu*. "Graph convolutional neural networks with global attention for improved materials property prediction." Physical Chemistry Chemical Physics 22, no. 32 (2020): 18141-18148.
```