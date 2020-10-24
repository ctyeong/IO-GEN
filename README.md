# Inner-Outlier Generator 

*Inner Outlier Generator (IO-GEN)* is designed to tackle 
One-class Classification (OC) problems by generating synthetic data 
that has a useful geographic property on the feature space of 
Deep Support Vector Data Description (DSVDD). 
This repo provides the official Tensorflow implementations of IO-GEN, which was first proposed in the paper: 

**"Identification of Abnormal States in Videos of Ants Undergoing Social Phase Change", under review for [IAAI-21](https://aaai.org/Conferences/AAAI-21/iaai-21-call/), (arXiv: https://arxiv.org/abs/2009.08626)**

<img src=Imgs/scenario.jpg width="85%">

Although theoretically, IO-GEN is applicable to any type of 
OC problems, here we focus on the exemplar scenario discussed in the above paper, where the classifier is trained only with observational samples from *stable* colony but has to distinguish *unstable* samples.
Instructions below start with a quick introduction to the pipeline of involved networks during training and test, followed by technical manuals to reproduce similar results to the paper. Ant motional data are also available at https://github.com/ctyeong/OpticalFlows_HsAnts.

# Contents 

1. [Model Pipeline](https://github.com/ctyeong/IO-GEN#model-pipeline)
2. [Installation](https://github.com/ctyeong/IO-GEN#installation)

# Model Pipeline
To better understand the code, we first review the pipeline of network model and data flows during training and test. 

## Training Steps

1. Deep Convolutional Autoencoder (DCAE) is trained. 
2. Encoder part of DCAE is fine-tuned as DSVDD.
3. IO-GEN is trained in an adversarial manner involving the feature space of DSVDD, which is frozen this time. 
   
   <img src=Imgs/pipeline1.jpg width="65%">
4. Classifier is trained on top of frozen (IO-GEN, DSVDD).

   <img src=Imgs/pipeline2.jpg width="65%">

## Structure for Test

(DSVDD, CLS) are the only components used after training to classify input optical flows as either stable or unstable. 

<!-- # SW/HW/Data Requirements
All codes here have been run with the following Python packages without issues:
- Python v3.6
- Tensorflow v2.1.0
- PIL v7.2.0
- Pandas v1.0.5

on a single NVIDIA TITAN XP graphics card (12GB) with the installation of NVIDIA Driver v440.100 (CUDA v10.2).
Moreover, users are encouraged to utilize optical flow data of ants at the following repo: https://github.com/ctyeong/OpticalFlows_HsAnts. -->

# Installation 

1. Clone the repository 
   ```
   git clone https://github.com/ctyeong/IO-GEN.git
   ```

2. Install the required Python packages
    ```
    pip install -r requirements.txt
    ```
    - Python 3.6 is assumed to be installed already

3. Download ant motional data
   ```
   cd IO-GEN
   git clone https://github.com/ctyeong/OpticalFlows_HsAnts.git
   mv split1 split2 split3 Stable Unstable ../ && cd ..
   rm -rf OpticalFlows_HsAnts
   ```
   


The following instruction assumes the input optical flows and at least one of the suggested splits have been downloaded from ["OpticalFlows_HsAnts"](https://github.com/ctyeong/OpticalFlows_HsAnts). That is, under the current directory, there are three folders such as:

`./Stable` 

`./Unstable` 

`./split1` 

For custom data, if the data hierarchy and a similar split format are given, the code here will require only few modifications.

## Training 
*train.py* is a *python* script to train *DCAE, DSVDD,* and *IO-GEN (followed by Classifier)* in order. Each model is evaluated per epoch, and it is saved into a specified directory as the best performance has been achieved during epochs. (The numbers of epochs are set to 750, 160, 20K, and 40, respectively as done in the paper above.)

Here is an example to run *train.py* with *split1* in the current directory.

`$ python train.py -s ./split1`

*'-s'* argument must be provided although all other arguments can be ignored just to run with default values.

The number of x,y optical flow pairs per input can be specified by *'-m'*: 

`$ python train.py -s ./split1 -m 1`

This prepares the data and models to only use one optical flow observation per sample (default=2).

The directories to store trained models and Tensorboard logs can also be specified:

`$python train.py -s ./split1 -d ./saved_models -t ./tb_logs`

The default directories are *./saved_models* and *./tb_logs*, respectively. Best models are named with *DCAE.h5, DSVDD.h5, IO-GEN.h5,* and *Classifier.h5* under the directory.

Lastly, *'-v'* can be set either *1* or *0* to control the level of explanation during training (default=1). 

## Test

*test.py* can be used to test either *DCAE, DSVDD,* or *IO-GEN* saved during training. As in training, a specific split must be given by *'-s'* argument, and additionally, the model name is required with *'-n'*. For example: 

`$ python test.py -s ./split1 -d ./saved_models -n IO-GEN -m 4 -v 0`

This command will use *split1* to run the saved *IO-GEN* model located under *./saved_models* folder, which was trained with 4 optical flow pairs per input. Also, less explanations will be shown by *'-v=0'*.

For each run, the Area Under the Curve (AUC) score of the Receiver Operating Characteristics (ROC) is given for two types of tests. For instance: 

`D+1: .700`

`D+2: .725`

`D+3 - D+6: .687`

`D+7 - D+10: .655`

`D+11 - D+14: .579`

`D+15 - D+18: .551`

`All: .672`

The first six rows indicate AUC scores of the model at different time frames in the unstable state while the last row the performance when all data are considered. This is the evaluation protocol used for the paper above.

# Contact

If there is any question or suggestion, please do not hesitate to shoot an email to *tchoi4@asu.edu*. Thanks!



