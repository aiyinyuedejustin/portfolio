from google.cloud import bigquery

client = bigquery.Client()


query = """

SELECT publication_number,title,abstract FROM `patents-public-data.google_patents_research.publications_202111` LIMIT 10

"""

results = client.query(query)

df = results.to_dataframe()

print(df)

for row in results:
    pub_ids = row['publication_number']
    print(f'{pub_ids}')


table_id = 'patentinfo-340203.test_for_patent.test_select_table'

client.delete_dataset('patentinfo-340203.test_for_patent',not_found_ok=True)
client.delete_table(table_id, not_found_ok=True)

dataset = client.create_dataset('test_for_patent')
table = dataset.table('test_select_table')

job_config = bigquery.QueryJobConfig(destination=table_id)

query_job = client.query(query, job_config=job_config)  # Make an API request.
query_job.result()  # Wait for the job to complete.

print("Query results loaded to the table {}".format(table_id))