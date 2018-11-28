# Image recognition

## Table of contents

- [Installation](#installation)
- [Instructions](#instructions)
- [Project motivation](#project-motivation)
- [File descriptions](#file-descriptions)
- [Results](#results)
- [Creator](#creator)
- [Thanks](#thanks)


## Installation

In order to be able to execute your own python statements it should be noted that scripts are only tested on **anaconda distribution 4.5.11** in combination with **python 3.6.6**. The scripts require additional python libraries.

Run the following commands in anaconda prompt to be able to run the scripts that are provided in this git repository.
- `conda install scikit-learn`
- `conda install pandas`
- `conda install numpy`
- `conda install pytorch`
- `conda install tochvision`
- `conda install argparse`
- `conda install tqdm `

Two quick start options are available:
- [Download the latest release.](https://github.com/FrankTub/DeepLearning/zipball/master/)
- Clone the repo: `git clone https://github.com/FrankTub/DeepLearning.git`

### Instructions:
Note that you will need to download the [training data](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) yourself and place this somewhere on your machine where the python script can access it.

Train your model
```text
Example of use:
   - python train.py data_directory              ==> where data_directory is directory where the train data is located.
   - python train.py data_dir -s save_directory  ==> -s allows you to dictate the directory where the checkpoint will be saved.
   - python train.py data_dir -a "vgg13"         ==> -a dictates the architecture of the pretrained model.
   - python train.py data_dir -g                 ==> -g enables GPU training.
   - python train.py data_dir -L 0.001           ==> -L sets the learning rate.
   - python train.py data_dir -H 512             ==> -H sets the number of hidden_layers in the classifier.
   - python train.py data_dir -E 4               ==> -E sets the number of iteration over the training set.
```
Make predictions
```text
Example of use:
   - python train.py input checkpoint            ==> where input is filepath of image and checkpoint is filepath where the model is saved.
   - python train.py input checkpoint -t 4       ==> where 4 represents the number of flowers returned by this script.
   - python train.py input checkpoint -c bla.json==> where bla.json represents the file that translates the number to a flower name.
   - python train.py input checkpoint -g         ==> -g enables predictions on GPU.
```
## Project motivation
For the first term of the nanodegree [become a data scientist](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://eu.udacity.com/) I got involved in this project. I was particular interested in trying out deep learning python libraries and see what results could be achieved.  

## File descriptions

Within the download you'll find the following directories and files.

```text
DeepLearning/
├── README.md
├── Image Classifier Project.ipynb # Notebook to try out deep learning python libraries
├── predict.py # Make prediction for an image
├── utility.py # Helper functions
├── train.py # Python script to train model on data directory
├── model.py # Model relation functions
└── cat_to_name.json # map index to flower name
```

## Results
By training the classifier of a [pretrained neural network](https://pytorch.org/docs/stable/torchvision/models.html) I was able to make predictions on images of more than a 100 flowers with an accuracy of 85%.

## Creator

**Frank Tubbing**

- <https://github.com/FrankTub>


## Thanks

<a href="https://eu.udacity.com/">
  <img src="https://eu.udacity.com/assets/iridium/images/core/header/udacity-wordmark.svg" alt="Udacity Logo" width="490" height="106">
</a>

Thanks to [Udacity](https://eu.udacity.com/) for setting up the projects where we can learn cool stuff!
