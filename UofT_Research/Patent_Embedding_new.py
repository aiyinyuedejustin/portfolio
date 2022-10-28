#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 3 11:23:46 2022

@author: justin
"""
import pandas as pd

import numpy as np

import os

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModelForMaskedLM


os.chdir(r'/Users/justin/Dropbox/RA work/AI/patent')


#%% AI PATENT ABSTRACT

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

model = AutoModelForMaskedLM.from_pretrained("anferico/bert-for-patents")

df = pd.read_csv('./AI_identifier_new.csv')

df_old = pd.read_csv('./AI_identifier.csv')

df = pd.concat([df,df_old,df_old]).drop_duplicates(keep=False)


df_abstract = pd.read_csv('./abstract.csv',low_memory=False)

df = df.merge(df_abstract,how='inner')

df = df[df['abstract'].notnull()].reset_index(drop=True)

sentence_vecs = np.empty(shape=(768,len(df)))

sentence_vecs = np.row_stack((df['Publication number'],sentence_vecs))


for i in range(len(df)):
        
    sentence_vecs[1:,i]= model.encode(df['abstract'][i])
    
    print(i)
    
sentence_vecs = sentence_vecs.T

sentence_vecs_old = np.load('./AI_patent_embedding.npy',allow_pickle=True)

sentence_vecs = np.concatenate((sentence_vecs_old,sentence_vecs),axis=0)


np.save('./AI_patent_embedding.npy',sentence_vecs)     
    


#%% APP DESCRIPTION

os.chdir(r'/Users/justin/Desktop/SensorTower/Data_Cleaned/datasets')

df = pd.read_csv('final_app_info.csv')

df = df[df['description'].notnull()].reset_index(drop=True)

sentence_vecs = np.empty(shape=(768,len(df)))

sentence_vecs = np.row_stack((df['app_id'],sentence_vecs))


for i in range(len(df)):
    
    sentence_vecs[:,i]= model.encode(df['description'][i])
    
    print(i)
    
sentence_vecs = sentence_vecs.T
    
np.save(r'/Users/justin/Dropbox/RA work/AI/patent/description_embedding.npy',sentence_vecs)


#%% SIMILARITY


os.chdir(r'/Users/justin/Dropbox/RA work/AI/patent')

description_embedding = np.load('./description_embedding.npy',allow_pickle=True)

description_embedding = description_embedding[:,1:].astype('float32')

AI_patent_embedding = np.load('./AI_patent_embedding.npy',allow_pickle=True)     

AI_patent_embedding = AI_patent_embedding[:,1:].astype('float32')

similarity = cosine_similarity(description_embedding,AI_patent_embedding)


np.save('./app_AI_similarity.npy',similarity)   


#%% SIMILARITY


data = np.load('./app_AI_similarity.npy',allow_pickle=True)

app_id = np.load('./description_embedding.npy',allow_pickle=True)

app_id = app_id[:,0]

average_similarity = data.sum(axis=1)

data[data<0] = 0

average_similarity0 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum0 = data.sum(axis=1)


data = np.load('./app_AI_similarity.npy',allow_pickle=True)

data[data<0.1] = 0

average_similarity1 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum1 = data.sum(axis=1)


data = np.load('./app_AI_similarity.npy',allow_pickle=True)

data[data<0.2] = 0

average_similarity2 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum2 = data.sum(axis=1)


data = np.load('./app_AI_similarity.npy',allow_pickle=True)

data[data<0.3] = 0

average_similarity3 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum3 = data.sum(axis=1)


data = np.load('./app_AI_similarity.npy',allow_pickle=True)

data[data<0.4] = 0

average_similarity4 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum4 = data.sum(axis=1)


data = np.load('./app_AI_similarity.npy',allow_pickle=True)

data[data<0.5] = 0

average_similarity5 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum5 = data.sum(axis=1)



average_similarity = np.column_stack((app_id,average_similarity,average_similarity0,average_similarity1,average_similarity2,average_similarity3,average_similarity4,average_similarity5,
                                      average_similarity_dum0,average_similarity_dum1,average_similarity_dum2,average_similarity_dum3,average_similarity_dum4,average_similarity_dum5))

df = pd.DataFrame(average_similarity,columns=['app_id','AI Exposure','AI Exposure (0)','AI Exposure (0.1)','AI Exposure (0.2)','AI Exposure (0.3)','AI Exposure (0.4)','AI Exposure (0.5)'
                                              ,'AI Exposure dummy (0)','AI Exposure dummy (0.1)','AI Exposure dummy (0.2)','AI Exposure dummy (0.3)','AI Exposure dummy (0.4)','AI Exposure dummy (0.5)'])

df.to_csv('../App_AI_Exposure_threshold.csv',index=False)


#%%WITHIN FIRM SIMILARITY

similarity = np.load('./app_AI_similarity.npy',allow_pickle=True)


df = pd.read_excel('../firm/top2000_BVD.xlsx')

df_firm_list = df.dropna(subset='BvD ID number').drop_duplicates(subset='BvD ID number')

df_firm = pd.read_csv('/Users/justin/Desktop/SensorTower/Data_Cleaned/datasets/Firm_ranks.csv')

df = df.merge(df_firm,on='unified_publisher_id',how='inner').dropna(subset='BvD ID number')

df = df[['unified_publisher_id', 'BvD ID number']]

df_info = pd.read_csv('../app/final_app_info.csv')

df = df.merge(df_info,on='unified_publisher_id',how='inner')

df = df[['app_id','BvD ID number']].reset_index(drop=True)

df_patent = pd.read_csv('../AI_patent_firm.csv')


description_embedding = np.load('./description_embedding.npy',allow_pickle=True)

description_embedding = description_embedding[:,0]


AI_patent_embedding = np.load('./AI_patent_embedding.npy',allow_pickle=True)     

AI_patent_embedding = AI_patent_embedding[:,0].T


df_final = pd.DataFrame()



for firm in df_firm_list['BvD ID number']:
    
    app = df[df['BvD ID number']==firm].reset_index(drop=True)
    
    patent = df_patent[df_patent['Publisher BvD ID number']==firm].reset_index(drop=True)
    
    patent_id = list()
    
    for j in range(len(patent)):
                    
        patent_id.extend(np.where(AI_patent_embedding == patent.loc[j,'Publication number'])[0])
        
    if len(patent_id)>0:
        
        patent_id.sort()
        
        app_id = list()
        
        for j in range(len(app)):
            
            app_id.extend(np.where(description_embedding == app.loc[j,'app_id'])[0])
            
        app = description_embedding[app_id]
        
        similarity_firm = similarity[app_id,:]
        
        similarity_firm = similarity_firm[:,patent_id]
        
        
        data = similarity_firm.copy()
        
        average_similarity = data.sum(axis=1)
    
        data[data<0] = 0
    
        average_similarity0 = data.sum(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum0 = data.sum(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.1] = 0
    
        average_similarity1 = data.sum(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum1 = data.sum(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.2] = 0
    
        average_similarity2 = data.sum(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum2 = data.sum(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.3] = 0
    
        average_similarity3 = data.sum(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum3 = data.sum(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.4] = 0
    
        average_similarity4 = data.sum(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum4 = data.sum(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.5] = 0
    
        average_similarity5 = data.sum(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum5 = data.sum(axis=1)
        
        
        average_similarity = np.column_stack((app,average_similarity,average_similarity0,average_similarity1,average_similarity2,average_similarity3,average_similarity4,average_similarity5,
                                          average_similarity_dum0,average_similarity_dum1,average_similarity_dum2,average_similarity_dum3,average_similarity_dum4,average_similarity_dum5))
    
        df_firm = pd.DataFrame(average_similarity,columns=['app_id','AI Exposure','AI Exposure (0)','AI Exposure (0.1)','AI Exposure (0.2)','AI Exposure (0.3)','AI Exposure (0.4)','AI Exposure (0.5)'
                                                  ,'AI Exposure dummy (0)','AI Exposure dummy (0.1)','AI Exposure dummy (0.2)','AI Exposure dummy (0.3)','AI Exposure dummy (0.4)','AI Exposure dummy (0.5)'])
                    
        df_final = df_final.append(df_firm)
    

df = df.merge(df_final,on='app_id',how='outer')

df = df.fillna(0)
                
df.to_csv('../App_Within_AI_Exposure_threshold.csv',index=False)




"""

This section is for app historical descriptions to estimate the similarity

"""

#%% APP DESCRIPTION

os.chdir(r'/Users/justin/Dropbox/RA work/AI/app')

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


df = pd.read_csv('app_description_list.csv')

df = df[df['description'].notnull()].reset_index(drop=True)

sentence_vecs = np.empty(shape=(768,len(df)))

sentence_vecs = np.row_stack((df['year'],sentence_vecs))

sentence_vecs = np.row_stack((df['app_id'],sentence_vecs))



for i in range(len(df)):
    
    sentence_vecs[2:,i]= model.encode(df['description'][i])
    
    print(i)
    
sentence_vecs = sentence_vecs.T
    
np.save(r'/Users/justin/Dropbox/RA work/AI/patent/description_embedding_historical.npy',sentence_vecs)



#%% SIMILARITY


os.chdir(r'/Users/justin/Dropbox/RA work/AI/patent')

description_embedding = np.load('./description_embedding_historical.npy',allow_pickle=True)

description_embedding = description_embedding[:,2:].astype('float32')

AI_patent_embedding = np.load('./AI_patent_embedding.npy',allow_pickle=True)     

AI_patent_embedding = AI_patent_embedding[:,1:].astype('float32')

similarity = cosine_similarity(description_embedding,AI_patent_embedding)


np.save('./app_AI_similarity_historical.npy',similarity)   




#%% SIMILARITY W/ TREATHHOLD

os.chdir(r'/Users/justin/Dropbox/RA work/AI/patent')

data = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)

np.random.seed(0)

select_cor= np.random.rand(166403)

sub=data[np.where(select_cor<0.1)]

select_cor= np.random.rand(68970)

sub=sub[:,np.where(select_cor<0.1)[0]]

test=np.max(sub,axis=1)
test=np.sort(test)
test=np.quantile(sub,0.999,axis=1)

test = pd.DataFrame(test)
test.describe()

np.quantile(data,0.5)
np.quantile(data,0.7)
np.quantile(test,0.9)
np.quantile(data,0.1)

app_id = np.load('./description_embedding_historical.npy',allow_pickle=True)

app_id = app_id[:,0:2]

average_similarity = data.sum(axis=1)

data[data<0] = 0

average_similarity0 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum0 = data.sum(axis=1)


data = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)

data[data<0.1] = 0

average_similarity1 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum1 = data.sum(axis=1)


data = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)

data[data<0.2] = 0

average_similarity2 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum2 = data.sum(axis=1)


data = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)

data[data<0.3] = 0

average_similarity3 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum3 = data.sum(axis=1)


data = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)

data[data<0.4] = 0

average_similarity4 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum4 = data.sum(axis=1)


data = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)

data[data<0.5] = 0

average_similarity5 = data.sum(axis=1)

data[data>0] = 1

average_similarity_dum5 = data.sum(axis=1)



average_similarity = np.column_stack((app_id,average_similarity,average_similarity0,average_similarity1,average_similarity2,average_similarity3,average_similarity4,average_similarity5,
                                      average_similarity_dum0,average_similarity_dum1,average_similarity_dum2,average_similarity_dum3,average_similarity_dum4,average_similarity_dum5))

df = pd.DataFrame(average_similarity,columns=['app_id','year','AI Exposure','AI Exposure (0)','AI Exposure (0.1)','AI Exposure (0.2)','AI Exposure (0.3)','AI Exposure (0.4)','AI Exposure (0.5)'
                                              ,'AI Exposure dummy (0)','AI Exposure dummy (0.1)','AI Exposure dummy (0.2)','AI Exposure dummy (0.3)','AI Exposure dummy (0.4)','AI Exposure dummy (0.5)'])

df.to_csv('../App_AI_Exposure_historical_threshold.csv',index=False)


df = pd.read_csv('../App_AI_Exposure_historical_threshold.csv')

df['AI Exposure (0)'].describe()


#%%WITHIN FIRM SIMILARITY

similarity = np.load('./app_AI_similarity_historical.npy',allow_pickle=True)


df = pd.read_excel('../firm/top2000_BVD.xlsx')

df_firm_list = df.dropna(subset='BvD ID number').drop_duplicates(subset='BvD ID number')

df_firm = pd.read_csv('/Users/justin/Desktop/SensorTower/Data_Cleaned/datasets/Firm_ranks.csv')

df = df.merge(df_firm,on='unified_publisher_id',how='inner').dropna(subset='BvD ID number')

df = df[['unified_publisher_id', 'BvD ID number']]

df_info = pd.read_csv('../app/final_app_info.csv')

df = df.merge(df_info,on='unified_publisher_id',how='inner')

df = df[['app_id','BvD ID number']].reset_index(drop=True)

df_patent = pd.read_csv('../AI_patent_firm.csv')


description_embedding = np.load('./description_embedding_historical.npy',allow_pickle=True)

app_year = description_embedding[:,0:2]

description_embedding = description_embedding[:,0]


AI_patent_embedding = np.load('./AI_patent_embedding.npy',allow_pickle=True)     

AI_patent_embedding = AI_patent_embedding[:,0].T


df_final = pd.DataFrame()



for firm in df_firm_list['BvD ID number']:
    
    app = df[df['BvD ID number']==firm].reset_index(drop=True)
    
    patent = df_patent[df_patent['Publisher BvD ID number']==firm].reset_index(drop=True)
    
    patent_id = list()
    
    for j in range(len(patent)):
                    
        patent_id.extend(np.where(AI_patent_embedding == patent.loc[j,'Publication number'])[0])
        
    if len(patent_id)>0:
        
        patent_id.sort()
        
        app_id = list()
        
        for j in range(len(app)):
            
            app_id.extend(np.where(description_embedding == app.loc[j,'app_id'])[0])
            
        app = app_year[app_id]
        
        similarity_firm = similarity[app_id,:]
        
        similarity_firm = similarity_firm[:,patent_id]
        
        
        data = similarity_firm.copy()
        
        average_similarity = data.mean(axis=1)
    
        data[data<0] = 0
    
        average_similarity0 = data.mean(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum0 = data.mean(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.1] = 0
    
        average_similarity1 = data.mean(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum1 = data.mean(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.2] = 0
    
        average_similarity2 = data.mean(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum2 = data.mean(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.3] = 0
    
        average_similarity3 = data.mean(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum3 = data.mean(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.4] = 0
    
        average_similarity4 = data.mean(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum4 = data.mean(axis=1)
    
    
        data = similarity_firm.copy()
    
        data[data<0.5] = 0
    
        average_similarity5 = data.mean(axis=1)
    
        data[data>0] = 1
    
        average_similarity_dum5 = data.mean(axis=1)
        
        
        average_similarity = np.column_stack((app,average_similarity,average_similarity0,average_similarity1,average_similarity2,average_similarity3,average_similarity4,average_similarity5,
                                          average_similarity_dum0,average_similarity_dum1,average_similarity_dum2,average_similarity_dum3,average_similarity_dum4,average_similarity_dum5))
    
        df_firm = pd.DataFrame(average_similarity,columns=['app_id','year','AI Exposure','AI Exposure (0)','AI Exposure (0.1)','AI Exposure (0.2)','AI Exposure (0.3)','AI Exposure (0.4)','AI Exposure (0.5)'
                                                  ,'AI Exposure dummy (0)','AI Exposure dummy (0.1)','AI Exposure dummy (0.2)','AI Exposure dummy (0.3)','AI Exposure dummy (0.4)','AI Exposure dummy (0.5)'])
                    
        df_final = pd.concat([df_final,df_firm])
        
        print(firm)
    

df = df.merge(df_final,on='app_id',how='outer')

df = df.fillna(0)
                
df.to_csv('../App_Within_AI_Exposure_historical_threshold.csv',index=False)


#%%  cleaning

df = pd.read_csv("../App_AI_Exposure_historical_threshold.csv",low_memory=False)

df_app_id = df.drop_duplicates(subset='app_id').reset_index(drop=True)

for year in range(2014,2021):
    
    df_app_id[year] = 1
    
df_app_id = df_app_id[['app_id',2014,2015,2016,2017,2018,2019,2020]]

df_app_id = df_app_id.melt(id_vars=['app_id'])

df_app_id.rename(columns={'variable':'year','value':'indicator'},inplace=True)

df = df_app_id.merge(df,how='outer')

df = df.sort_values(by=['app_id','year']).reset_index(drop=True)

df = df.drop(df[df['year']==0].index)

df_new = df.groupby(by=['app_id']).fillna(method='bfill')

df_new = df_new.merge(df['app_id'],left_index=True, right_index=True)

df_new = df_new.drop(['indicator'], axis=1)

col = df_new.columns.to_list()

col = col[-1:] + col[:-1]

df_new = df_new[col]

df_new.to_csv('../App_AI_Exposure_historical_threshold_clean.csv',index=False)


df = pd.read_csv("../App_Within_AI_Exposure_historical_threshold.csv",low_memory=False)

df_app_id = df.drop_duplicates(subset='app_id').reset_index(drop=True)

for year in range(2014,2021):
    
    df_app_id[year] = 1
    
df_app_id = df_app_id[['app_id',2014,2015,2016,2017,2018,2019,2020]]

df_app_id = df_app_id.melt(id_vars=['app_id'])

df_app_id.rename(columns={'variable':'year','value':'indicator'},inplace=True)

df = df_app_id.merge(df,how='outer')

df = df.sort_values(by=['app_id','year']).reset_index(drop=True)

df = df.drop(df[df['year']==0].index)

df_new = df.groupby(by=['app_id']).fillna(method='bfill')

df_new = df_new.merge(df['app_id'],left_index=True, right_index=True)

df_new = df_new.drop(['indicator','BvD ID number'], axis=1)

col = df_new.columns.to_list()

col = col[-1:] + col[:-1]

df_new = df_new[col]

df_new.to_csv('../App_Within_AI_Exposure_historical_threshold_clean.csv',index=False)



