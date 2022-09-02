# Introduction to Deep Learning (IN2346)
# Technical University Munich - SS 2022

## 1. Python Setup

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.10, so this is what we are going to install here.

If you are using Windows, you will have to google or check out the forums for setup help from other students. There are plenty of instructions for Anaconda for Windows using a graphical user interface though.

To avoid issues with different versions of Python and Python packages we recommend that you always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*. For simplicity, we are going to focus on Anaconda.

### Anaconda setup
Download and install miniconda (minimal setup with less start up libraries) or conda (full install but larger file size) from [here](https://www.anaconda.com/products/distribution#Downloads). Create an environment using the terminal command:

`conda create --name i2dl python=3.10`

Next activate the environment using the command:

`conda activate i2dl`

Continue with installation of requirements and starting jupyter notebook using:

`pip install -r requirements.txt` 

`jupyter notebook`

Jupyter notebooks use the python version of the current active environment so make sure to always activate the `i2dl` environment before working on notebooks for this class.

## 2. Exercise Download

The exercises will be uploaded to the [course website](https://dvl.in.tum.de/teaching/i2dl-ws20/) as well as our forum. You can download the exercises directly from there. Each time we start a new exercise you will have to unzip the exercise and copy it into the current directory as we are utilizing some shared folders.
### The directory layout for the exercises

    i2dl_exercises
    ├── datasets      # The datasets will be stored here
    ├── exercise_1                 
    ├── exercise_2                     
    ├── exercise_3                    
    ├── exercise_4
    ├── exercise_5
    ├── exercise_6
    ├── exercise_7                              
    ├── exercise_8
    ├── exercise_9
    ├── exercise_10
    ├── exercise_11
    ├── exercise_12                    
    ├── output         # Where you will find zipped exercises for uploading
    └── README.md
    └── requirements.txt


## 3. Dataset Download

Datasets will generally be downloaded automatically by exercise notebooks and stored in a common datasets directory shared among all exercises. A sample directory structure for cifar10 dataset is shown below:-

    i2dl_exercises
        ├── datasets                   # The datasets required for all exercises will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files 

## 4. Exercise Submission
Your trained models will be automatically evaluated on a test set on our server. To this end, login or register for an account at:

https://i2dl.dvl.in.tum.de/

Note that only students who have registered for this class in TUM Online can register for an account. By using your matriculation number we send an login data to your associated email address (probably the tum provided account if you didn't change it).

After you have worked through an exercise, execute the notebook cells that saves and zips the exercise. The output can be found in the global `i2dl_exercises/output` folder.

You can login to the above website and upload your zip submission for the current exercise. Your submission will be evaluated by our system. 

You will receive an email notification with the results upon completion of the evaluation. To make the exercises more fun, you will be able to see a leaderboard of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leader board always represents the best score for the respective exercise.

## 5. Acknowledgments

We want to thank the **Stanford Vision Lab**, **PyTorch** and **PyTorch Lightning** for allowing us to build these exercises on material they had previously developed. We also thank the **TU Munich Visual Computing and Artificial Intelligence Group** for helping create course content.
