
Here are some example code for my current research assistant job at UofT. Note that the below brief introduction does not exactly correspond to the code in this folder. The codes are just examples.

The first version of the paper can be find here: https://www.nber.org/papers/w29980. 

Based on the first version of the research works, improvements are needed for further economic analysis. To be more specific, my job aims to find the AI patents that are most likely to be used in various APPs. The patents data were originated from Google Patents Public Data. By using specific identifers, AI related patents were identified and grouped using Bigquery from Google Cloud Platform. The APP data were obtained using web crawler from Google Play and APP Store. 

To find the patent that might be used in a certain app, we focused mainly on the patent abstract and app description. Since companies have the freedom to write what they want in the app description, cleaning is need to standardize the app description. We split each paragraph of app description and patent abstract into multiple sentences according to some rules, which could increase the quality of sentence embedding. By using multilingual pre-trained model of sentence bert(https://www.sbert.net/index.html), sentence-level embeddings were obtained. Using cosine similarity and by taking average according to various metrics, one unique simialrity can be obtained between each originial description and abstract. By using the magnitude of similairty, further economic anaysis could be done.
