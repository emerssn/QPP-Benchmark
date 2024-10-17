import pyterrier as pt
import math
import csv
from collections import defaultdict
import ir_datasets
import sys
import os  # Importamos os para manejar rutas absolutas

if not pt.started():
    pt.init()

class IDFMetrica:
    def __init__(self, dataset, output_file):
        self.dataset = dataset
        self.doc_freqs = defaultdict(int)
        self.num_docs = len(list(dataset.docs_iter()))
        self.output_file = output_file

    def compute_idf(self, query):
        terms = query.split()
        idfs = []
        for term in terms:
            doc_freq = self.doc_freqs[term]
            if doc_freq > 0:
                idf = math.log(self.num_docs / (doc_freq + 1))
            else:
                idf = 0
            idfs.append(idf)
        return max(idfs), sum(idfs) / len(terms) if terms else 0

    def calculate_idf_for_dataset(self):
        # Calcula la frecuencia de términos en los documentos
        for doc in self.dataset.docs_iter():
            terms_in_doc = set(doc.text.split())
            for term in terms_in_doc:
                self.doc_freqs[term] += 1

        # Asegurarse de que la carpeta de salida existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Abre el archivo CSV para escribir los resultados
        with open(self.output_file, 'w', newline='') as csvfile:
            fieldnames = ['Query ID', 'Max IDF', 'Avg IDF']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for query in self.dataset.queries_iter():
                max_idf, avg_idf = self.compute_idf(query.text)
                writer.writerow({'Query ID': query.query_id, 'Max IDF': max_idf, 'Avg IDF': avg_idf})

# Capturar el argumento del dataset al ejecutar
if len(sys.argv) < 3:
    print("Usage: python idf_method.py <dataset-name> <output-file>")
    sys.exit(1)

dataset_name = sys.argv[1]  # El primer argumento es el nombre del dataset
output_file = sys.argv[2]  # El segundo argumento es el archivo donde se guardan los resultados

# Convertir el archivo de salida a una ruta absoluta
output_file = os.path.abspath(output_file)

# Cargar el dataset
dataset = ir_datasets.load(dataset_name)

# Ejecutar la métrica IDF y guardar los resultados
idf_metric = IDFMetrica(dataset, output_file)
idf_metric.calculate_idf_for_dataset()
