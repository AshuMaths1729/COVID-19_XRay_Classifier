# COVID-19 XRay Classifier
As part of the SAMHAR-COVID-19 Hackathon

## The project aims to predict if a person is Normal or having normal Pneumonia or is suffering from COVID-19 Pneumonia.

___
Combined datasets from two sources:

1. Joseph Paul Cohen and Paul Morrison and Lan Dao
	COVID-19 image data collection, arXiv:2003.11597, 2020
	https://github.com/ieee8023/covid-chestxray-dataset	
	https://arxiv.org/abs/2003.11597

2. http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
	https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

-> Modified data from source 1. Put COVID-19 and COVID-19, ARDS into one group - COVID-19. Put No Finding tagged images under Normal.
Put all other images other than these tags, under Pneumonia.

-> Put source 2 images which are under Pneumonia and Normal, under respective categories. Chose very fewer images from source 2, to enable balanced dataset.

-> Will be better if we get more dataset regarding COVID-19 cases, because we do have data regarding Pneumonia and Normal, but nor so much regarding COVID-19.

-> Trained a custom CNN classifier to achieve the task of classfication on three classes of images.

-> The Trained model can be found at: https://drive.google.com/file/d/1nwXP0_YMcC-_3QA5wf7lj2unNU-yzMLi/view?usp=sharing

## To-do:
Deploy the trained classifier on to an Android App, for easy functionality.

