AVAILABLE_DATASETS = {
    "antique_test": "irds:antique/test",
    "iquique_small": "iquique_dataset",
    "cranfield": "irds:cranfield",
    "fiqa": "irds:beir/fiqa/dev",
    "car": "irds:beir/car/dev",
    # Add more datasets as needed
    # "msmarco": "irds:msmarco/...",
    # "trec": "irds:trec/...",
}

# Dataset format configurations updated for TREC compliance
DATASET_FORMATS = {
     "antique_test": {
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        # Fixed transformation with type safety
        "doc_id_transform": lambda x: str(x).strip(),
        "relevance_levels": {1: "Out of context", 2: "Not relevant", 3: "Marginal", 4: "Highly relevant"},
        "binary_threshold": 3,
        "gain_values": {1: 0, 2: 1, 3: 2, 4: 3}
    },
}