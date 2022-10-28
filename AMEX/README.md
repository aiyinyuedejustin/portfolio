This is a project for https://www.kaggle.com/competitions/amex-default-prediction

Steps to run:

1：download original data to input/amex-default-prediction；

download processed data to input/amex-data-integer-dtypes-parquet-format
（https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format）

2：run fe.ipynb，it takes some time 

3：run lgb.ipynb，train lgb

4：run xgb.ipynb，train xgb(on GPU)

5：run fe_v2.ipynb, which is a further feature engineering process

6：run lgb_v2.ipynb，train lgb_v2

7：run lgb_v3.ipynb，train lgb_v3

8：run combine.ipynb，get combined results


It might need 64 GB Memory to run


Brief Summary:

The algorithm is developed for customer credit default prediction based on a variety of customer credit card usage records provided by American Express: including time-series behavioral data and anonymous customer profile information. The task is a classification problem based on highly unbalanced data, and the average of Normalized Gini Coefficient and Default Rate Captured at 4% are used to evaluate the accuracy of the model.

Based on a large number of user credit card evaluation records, extract features and add features to historical time series data, including but not limited to:
a) add 'mean', 'std', 'min', 'max' to payment, balance, and spend
b) add lagged features by cacluate difference between "last" and "first"
c) delete B_29 feature

The different combinations of the features were trained with LightGBM, XGBoost and CatBoost models. Through correlation analysis, three LightGBM and one XGBoost are  retained. Finally, the linear weighted average method is used to combine the four files.

