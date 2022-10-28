from google.cloud import bigquery
import numpy as np
import pandas as pd
import re

AI_class = np.load('/home/gaoyukun316/bigquery-demo/bigquery-demo/AIClass.npy',allow_pickle=True)

block1 = AI_class[0,]

block2 = AI_class[1,]
        
block3_cpc = AI_class[2,]
    
block3_ipc = AI_class[3,]
    
block3_kw = AI_class[4,]

# Define critiria for AI selection

def AIblock1(data):
    data = data[['cpc_code','cpc_first']].dropna(how='all').reset_index(drop=True)
    ind = 0
    ind_primary = 0

    if len(data)>0:
        for i in range(len(data)):
            if data['cpc_code'][i] in block1:
                ind = 1
                if data['cpc_first'][i] is True:
                    ind_primary = 1        
    return ind, ind_primary


def AIblock2(data):
        
    data = data[['title','abstract','claims_text']].drop_duplicates().reset_index(drop=True)
    ind = 0
    keyword = []    
    if str(data['title'][0])!='None':
        keyword = re.findall(block2, data['title'][0])
    if str(data['abstract'][0])!='None' and str(data['abstract'][0])!='nan':
        keyword.extend(re.findall(block2, data['abstract'][0]))
    if str(data['claims_text'][0])!='None' and str(data['claims_text'][0])!='nan':
        keyword.extend(re.findall(block2, data['claims_text'][0]))
    if len(keyword) > 0:
        ind = 1
        
    return ind, keyword



def AIblock3(data_all):

    ind = 0
    ind_primary = 0
    keyword = []
    
    data = data_all[['cpc_code','cpc_first']].dropna(how='all').reset_index(drop=True)

    if len(data)>0:
        for i in range(len(data)):
            if data['cpc_code'][i] in block3_cpc:
                ind = 1
                if data['cpc_first'][i] is True:
                    ind_primary = 1  

    data = data_all[['ipc_code','ipc_first']].dropna(how='all').reset_index(drop=True)

    if len(data)>0:
        for i in range(len(data)):
            if data['ipc_code'][i] in block3_ipc:
                ind = 1
                if data['ipc_first'][i] is True:
                    ind_primary = 1 

    data = data_all[['title','abstract','claims_text']].drop_duplicates().reset_index(drop=True)

    if ind == 1:
        if str(data['title'][0])!='None' and str(data['title'][0])!='nan':
            keyword = re.findall(block3_kw, data['title'][0])
        if str(data['abstract'][0])!='None' and str(data['abstract'][0])!='nan':
            keyword.extend(re.findall(block3_kw, data['abstract'][0]))
        if str(data['claims_text'][0])!='None' and str(data['claims_text'][0])!='nan':
            keyword.extend(re.findall(block3_kw, data['claims_text'][0]))
        if len(keyword) == 0:
            ind = 0
            ind_primary =0
                    
    return ind, ind_primary, keyword
                
    
# query data from google patent

client = bigquery.Client()

#### title, abstract and claim
keywords_rows = 138900000 
mylimit = 50000
save = False

ipc_rows = keywords_rows
cpc_rows = keywords_rows
times = keywords_rows//mylimit
for g in range(0, times):
    
    patent_query="""

    SELECT * FROM `high-magpie-356723.test.keywords`
    ORDER BY publication_number
    LIMIT {}
    OFFSET {}

    """.format(mylimit, mylimit * g)
    results = client.query(patent_query)

    df_keyword = results.to_dataframe() # translate data from query job to dataframe


    patent_query="""

    SELECT * FROM `high-magpie-356723.test.cpc_expand` AS cpc
    RIGHT JOIN  (
    SELECT publication_number , abstract, claims_text, claims_language FROM `high-magpie-356723.test.keywords` 
    ORDER BY publication_number
    LIMIT {}
    OFFSET {}) AS ids
    ON cpc.publication_number = ids.publication_number

    """.format(mylimit, mylimit*g)

    results = client.query(patent_query)

    df_cpc = results.to_dataframe() # translate data from query job to dataframe


    patent_query="""

    SELECT * FROM `high-magpie-356723.test.ipc_expand` AS ipc
    RIGHT JOIN  (
    SELECT publication_number , abstract, claims_text, claims_language FROM `high-magpie-356723.test.keywords` 
    ORDER BY publication_number
    LIMIT {}
    OFFSET {}) AS ids
    ON ipc.publication_number = ids.publication_number
    """.format(mylimit, mylimit*g)

    results = client.query(patent_query)

    df_ipc = results.to_dataframe() # translate data from query job to dataframe

    # process data in dataframe

    col_name=['publication_number','block1','block1_primary','block2','block2_kw','block3','block3_primary','block3_kw','position']
    df = pd.DataFrame(columns=col_name)

    print('there are '+str(len(df_keyword))+' patents')
    print('there are '+str(len(df_cpc))+' cpc')
    print('there are '+str(len(df_ipc))+' ipc')

    i = 0
    step = 1000    
    flag = list(range(0,130000000,step))
	
	def cc(x):
		data = df_keyword[df_keyword['publication_number']==pub_id]
        data = data.merge(df_cpc[df_cpc['publication_number']==pub_id],how='outer',on='publication_number')
        data = data.merge(df_ipc[df_ipc['publication_number']==pub_id],how='outer',on='publication_number')

        block1_ind, block1_primary_ind=AIblock1(data)
        block2_ind, keywords = AIblock2(data)
        block3_ind, block3_primary_ind, keywords_3 = AIblock3(data)
		
		return block1_ind,block1_primary_ind,block2_ind,keywords,block3_ind, block3_primary_ind, keywords_3
	df_keyword[['block1_ind','block1_primary_ind','block2_ind','keywords,block3_ind', 'block3_primary_ind', 'keywords_3']] = df_keyword['publication_number'].apply(cc,axis=1)
	df_dd = df_keyword[df_keyword['block1_ind']+df_keyword['block2_ind']+df_keyword['block3_ind']>0]
    df_dd[['block1_ind','block1_primary_ind','block2_ind','keywords,block3_ind', 'block3_primary_ind', 'keywords_3']].to_csv('sss.csv')
        if (block1_ind+block2_ind+block3_ind)>0:
            temp = pd.DataFrame([pub_id,block1_ind, block1_primary_ind,block2_ind,keywords, block3_ind, block3_primary_ind,keywords_3,i]).T
            temp.columns=col_name
            df = pd.concat([df,temp], ignore_index=True)
		

    for pub_id in df_keyword['publication_number']:
        
        #retrieve data for a single publication

        data = df_keyword[df_keyword['publication_number']==pub_id]
        data = data.merge(df_cpc[df_cpc['publication_number']==pub_id],how='outer',on='publication_number')
        data = data.merge(df_ipc[df_ipc['publication_number']==pub_id],how='outer',on='publication_number')

        block1_ind, block1_primary_ind=AIblock1(data)
        block2_ind, keywords = AIblock2(data)
        block3_ind, block3_primary_ind, keywords_3 = AIblock3(data)
            
        if (block1_ind+block2_ind+block3_ind)>0:
            temp = pd.DataFrame([pub_id,block1_ind, block1_primary_ind,block2_ind,keywords, block3_ind, block3_primary_ind,keywords_3,i]).T
            temp.columns=col_name
            df = pd.concat([df,temp], ignore_index=True)
        
        print(i)
        i=i+1
        if save and i in flag:
            df.to_csv('/home/gaoyukun316/bigquery-demo/bigquery-demo/pubs/AI_publications{}.csv'.format(g),index=False)

    df.to_csv('/home/gaoyukun316/bigquery-demo/bigquery-demo/pubs/AI_publications{}.csv'.format(g),index=False)
