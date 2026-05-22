#!/usr/bin/env python3
"""
Compare NLTK preprocessing (QPP-Benchmark) vs Terrier's own preprocessing.
Stores full token/stem lists per pipeline into separate JSON files.
"""

import sys
import json
import re
from collections import Counter

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import pyterrier as pt
pt.init()
from jnius import autoclass
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

terrier_tokenizer = autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()
terrier_porter = autoclass('org.terrier.terms.PorterStemmer')()

nltk_stop = set(stopwords.words('english'))
snowball = SnowballStemmer('english')

def nltk_tok(text):
    clean = re.sub(r'[^\w\s]', '', text.lower())
    return [t for t in word_tokenize(clean) if t]

def nltk_stem(text):
    return [snowball.stem(t) for t in nltk_tok(text)]

def nltk_full(text):
    return [t for t in nltk_stem(text) if t not in nltk_stop]

def terrier_tok(text):
    return [str(t).lower() for t in terrier_tokenizer.getTokens(text)]

def terrier_stem(text):
    return [terrier_porter.stem(str(t).lower().strip()) for t in terrier_tokenizer.getTokens(text)]

def terrier_full(text):
    # Full = stem + stopword removal using NLTK's stoplist applied on Terrier stems
    return [t for t in terrier_stem(text) if t not in nltk_stop]

# --- Load dataset ---
import ir_datasets
dataset = ir_datasets.load('antique/test')

sample_doc_count = 200
docs = []
for doc in dataset.docs_iter():
    docs.append({'docno': doc.doc_id, 'text': doc.text})
    if len(docs) >= sample_doc_count:
        break

queries = []
for q in dataset.queries_iter():
    queries.append({'qid': q.query_id, 'text': q.text})

print(f"Loaded {len(docs)} docs, {len(queries)} queries\n")

# --- Process and collect ---
def process_all(items, id_key, out_prefix):
    results = {
        'nltk_tokens': {},
        'nltk_stems': {},
        'nltk_full': {},
        'terrier_tokens': {},
        'terrier_stems': {},
        'terrier_full': {},
    }

    vocab_nltk_tokens = Counter()
    vocab_nltk_stems = Counter()
    vocab_nltk_full = Counter()
    vocab_terrier_tokens = Counter()
    vocab_terrier_stems = Counter()
    vocab_terrier_full = Counter()

    for item in items:
        item_id = item[id_key]
        text = item['text']
        if not text or not text.strip():
            continue

        nt = nltk_tok(text)
        ns = nltk_stem(text)
        nf = nltk_full(text)
        tt = terrier_tok(text)
        ts = terrier_stem(text)
        tf = terrier_full(text)

        results['nltk_tokens'][item_id] = nt
        results['nltk_stems'][item_id] = ns
        results['nltk_full'][item_id] = nf
        results['terrier_tokens'][item_id] = tt
        results['terrier_stems'][item_id] = ts
        results['terrier_full'][item_id] = tf

        vocab_nltk_tokens.update(nt)
        vocab_nltk_stems.update(ns)
        vocab_nltk_full.update(nf)
        vocab_terrier_tokens.update(tt)
        vocab_terrier_stems.update(ts)
        vocab_terrier_full.update(tf)

    vocabs = {
        'nltk_tokens': dict(vocab_nltk_tokens.most_common()),
        'nltk_stems': dict(vocab_nltk_stems.most_common()),
        'nltk_full': dict(vocab_nltk_full.most_common()),
        'terrier_tokens': dict(vocab_terrier_tokens.most_common()),
        'terrier_stems': dict(vocab_terrier_stems.most_common()),
        'terrier_full': dict(vocab_terrier_full.most_common()),
    }

    with open(f'{out_prefix}_per_item.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved {out_prefix}_per_item.json ({len(results['nltk_tokens'])} items)")

    with open(f'{out_prefix}_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocabs, f, indent=2, ensure_ascii=False)
    print(f"  Saved {out_prefix}_vocab.json")

    return vocabs

print("Processing documents...")
doc_vocabs = process_all(docs, 'docno', 'docs')

print("\nProcessing queries...")
query_vocabs = process_all(queries, 'qid', 'queries')

# --- Summary stats ---
def print_vocab_summary(vocabs, label):
    print(f"\n{'='*60}")
    print(f"  {label} — Vocabulary sizes")
    print(f"{'='*60}")
    for pipeline, vocab in vocabs.items():
        total_freq = sum(vocab.values())
        print(f"  {pipeline:20s}: {len(vocab):>6,} unique terms, {total_freq:>8,} total tokens")

print_vocab_summary(doc_vocabs, "DOCUMENTS")
print_vocab_summary(query_vocabs, "QUERIES")

print("\nDone.")
