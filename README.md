# MovieLens

This project was conducted as part of the HarvardX Data Science Capstone Module and was inspired by the Netflix challenge. 
The primary goal is to develop a movie recommendation system that leverages the extensive Movielens dataset 
(https://grouplens.org/datasets/movielens/10m/)  of 10 millionentries provided by the Department of Computer Science and 
Engineering at the University of Minnesota.

The Movielens database comprises movies, complete with title and genre information, alongside movie ratings that include 
user and timestamp details. The overarching objective involves constructing a predictive model for unknown ratings. 
The model's accuracy is assessed using the Root Mean Square Error (RMSE) metric.

The development of the model is done in three main steps.

## 1. Data Load, Analyis and Preparation 
The Movielens datasets are loaded from the files supplied in the course, initiating a sequence of 
straightforward analyses aimed at comprehending the dataset. To facilitate the implementation of 
advanced models, supplementary variables used as predictors are computed and presented visually. 
At last, the data is partinioned into model and validation datasets. The model dataset is further
partinioned into a train and test dataset and a small dataset is created to avoid long run times 
in some of the later steps.

## 2. Model Development
Multiple prediction models are constructed and tested using subsets of the model data.
The progression involves enhancing the approach step by step:
 
 - Initially, a simple linear model is established, employing movie and user as the sole predictors.
 - The simple linear model undergoes refinement through the incorporation of regularization.
 - A more intricate regularized linear model is crafted by introducing two additional predictors: 
   the number of ratings for a movie and the duration since the movie was first rated.
 - A general linear model is devised for the residuals of the preceding model.
 - An ensemble of models is created for the residuals of the aforementioned model.
 
 
## 3. Model Validation
Model validation was executed using a distinct subset of the data, exclusively reserved 
for this purpose.

The project consists of the following files:

- Movielens.pdf report of the project and results
- Movielens.R R script
- Movielens.Rmd R markdown, which was created from the R script and generates the pdf
  
