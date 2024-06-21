# Signal vs Background discrimination @SABRE

This repository contains the code used to train and evaluate the signal vs background discrimination model for the SABRE experiment. 

## Model

Currently the model is based on a 1 dimensional convolutional neural network (CNN), but many other approaches can be included easily just by adding a new files in the folder `models/`.

## Data

The data format is a folder containing individual txt files with the numbers representing the waveforms.

Two folders are needed: one for the signal and one for the background.

The data is then divided into training, validation and test sets for significant training and evaluation.

## Settings

To change the specific setting for the run, multiple settings files in json format can be created in the folder `settings/`.

Right now, the settings include only the path for the data and the output folder for the results, but more can be added easily.

## Running the code

To run the code, just execute the script with:

```source run.sh```

where the settings file is hardcoded in the script, or with:

```source run.sh -j <settings_file>```

or

```source run.sh --json_file <settings_file>```

where the `<settings_file>` is the json file with the settings.
