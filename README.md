# IO-GEN

# Introduction

Codes used for 
"Identification of Abnormal States in Videos of Ants Undergoing Social Phase Change",
Taeyeong Choi, Benjamin Pyenson, Juergen Liebig, Theodore P. Pavlic, 
*Submitted to [IAAI-21](https://aaai.org/Conferences/AAAI-21/iaai-21-call/)*. 

The following models from the paper can be built, trained, and tested for One-class Classification problems: 
- Deep Convolutional Autoencoder (DCAE)
- Deep Support Vector Data Description (DSVDD)
- IO-GEN (with Classifier) 

# Requirements
We have tested the code with the following Python packages without issues:
- Python v3.6.9
- Tensorflow v2.1.0
- PIL v7.2.0
- Pandas v1.0.5

on a NVIDIA TITAN XP graphics card (12GB) with the installation of NVIDIA Driver v440.100 (CUDA v10.2)

# Usage 

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



