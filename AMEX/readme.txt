Steps to run:
1：download original data to input/amex-default-prediction；
download processed data to input/amex-data-integer-dtypes-parquet-format
（https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format）
2：run code/fe.ipynb，it takes some time 
3：run code/lgb.ipynb，train lgb
4：run code/xgb.ipynb，train xgb(on GPU)
5：run code/fe_v2.ipynb
6：run  code/lgb_v2.ipynb，train lgb_v2
7：run code/lgb_v3.ipynb，train lgb_v3
8：run code/combine.ipynb，get combined results


It might need 64 GB RAM to run