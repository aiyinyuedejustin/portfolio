This is a project for https://www.kaggle.com/competitions/optiver-realized-volatility-prediction

Download the data and simply run main.ipynb, feature engineering and model training are all in the same file.

Brief Summary:

1. Based on the actual quotation and trading data of stocks, extract a series of feature combinations F1 of trading spread, quotation spread, trading volume, etc.; use the clustering algorithm to cluster 113 classes of stocks into 7 classes and calculate the above feature combination F2 of these 7 classes; use the KNN algorithm to find the most similar data points in time sequence and calculate the feature combination F3 in the most similar time; combine F1, F2, F3 to get the final feature vector F.

2. Based on the above features, build machine learning LightGBM and neural network models, use RMSPE as the loss function. Use sklearn, lightgbm, tensorflow and other modules to build models for training, to achieve the effect of predicting the actual volatility when given trading data.

