# Actor Critic methods applied to a modified version of the Catch environment
A collection of actor critic RL methods that are applied to the Catch env. with PyTorch.

This project implements two different Actor-Critic algorithms with bootstrapping for the Catch environment. The first algorithm is the Advantage Actor-Critic (A2C) with bootstrapping, and the second is a standard Actor-Critic (AC) with bootstrapping. The catch enviornment is built in the codes.

## Installation
To run this project, you'll need Python 3.7 or higher, as well as the following packages:

gym

numpy

matplotlib

torch

## Usage
To run the A2C algorithm, use the following command:

`python A2C_bs.py`

To run the AC algorithm, use the following command:

`python AC_bs.py`

## Tuning
Both scripts will train the corresponding algorithm. You can also change the hyperparameters and other settings in the codes in a class called `Hyperparameters`.

## Results
After training, the scripts will generate plots showing the average reward per episode over time.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
