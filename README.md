# Deep Learning Final Project 
## A. Barbieri - A. Pederzani

The aim of this project is to replicate (at least partially) the model and the results presented in this [paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/49182f81e6a13cf5eaa496d51fea6406-Paper.pdf), approaching the style transfer task by means 
of autoencoders trained for image reconstruction and the WCT transformation presented, pertrubing the latent representation of the input image.

## Folder Content

This repository contains the following files/directories:
- `images`: folder containing two images needed for the `visualize.ipynb` notebook.
- `train`: contains all the files used to train and test the models.
- `model.py`: file containing the model implemented, together with additional functions nedded to implement the WCT transformation.
- `visualize.ipynb`: a toy example showing the models in action.
- `parameters`: a folder that should contain the parameters of the model.

## Parameters
Due to the size of the files containing the trained parameters (~134 MB in total) we can't load them directly here on GitHub. However, following this [link](https://units-my.sharepoint.com/:f:/g/personal/s280811_ds_units_it/EvOXWJCiPUNDraVgZXqt1bIBb9HDxPnJguzaJ7-0oRSY0A?e=Uoqkhr) you should be able to download them. 
