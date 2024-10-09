import pyterrier as pt
import os
import shutil
from pyterrier.measures import *
import matplotlib.pyplot as plt

dataset = pt.get_dataset('irds:antique/test')
index_path = './indices/antique'

# Check if index already exists
if os.path.exists(index_path):
    response = input(f"Index already exists at {index_path}. Do you want to delete it? (y/n): ")
    if response.lower() == 'y':
        shutil.rmtree(index_path)
        print("Index deleted.")
        # Proceed to index documents after deletion
        indexer = pt.IterDictIndexer(index_path)
        indexer.setProperty("terrier.index.meta.forward.keys", "docno,text")
        indexer.setProperty("terrier.index.meta.forward.keylens", "20,100000")
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text'])
    else:
        print("Index not deleted.")
        # Load existing index
        index_ref = pt.IndexFactory.of(index_path)  # Use IndexFactory to load the index
else:
    # Indexación de documentos
    indexer = pt.IterDictIndexer(index_path)
    indexer.setProperty("terrier.index.meta.forward.keys", "docno,text")
    indexer.setProperty("terrier.index.meta.forward.keylens", "20,100000")
    index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text'])

# Recuperación de tópicos
topics = dataset.get_topics()
print(topics)

# Recuperación de documentos
bm25 = pt.terrier.Retriever(index_ref, wmodel='BM25') >> pt.text.get_text(dataset, "text")

# Capture the results of the experiment
results = pt.Experiment(
    [bm25],
    dataset.get_topics(),
    dataset.get_qrels(),
    [MAP, nDCG@20]
)

# Print the results
print(results)

# Print column names
print("Column names:", results.columns)

# Plot the results
plt.figure(figsize=(10, 6))
metric_columns = [col for col in results.columns if col not in ['name', 'query']]
results.plot(x='name', y=metric_columns, kind='bar')
plt.title('BM25 Performance')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()


from IPython.display import display, HTML
import pandas as pd

# Test the BM25 model with a simple query
sample_query = "capital of france"
results = bm25.search(sample_query)

# Print the results
print("Results for the query:", sample_query)

# Print the shape and first few rows of the results
print("Shape of results:", results.shape)
print("First few rows of results:")
print(results.head())

# Create a function to display results with scrollable text
def display_results_with_scroll(df, max_rows=5):
    df_html = df.to_html(index=False, classes='table table-striped')
    display(HTML(f"""
    <style>
        .scrollable-cell {{
            max-height: 100px;
            overflow-y: auto;
        }}
    </style>
    {df_html.replace('<td>', '<td class="scrollable-cell">')}
    """))

# Display results without the index, and with scrollable text
display_results_with_scroll(results[['rank', 'query', 'text']].head(5))






