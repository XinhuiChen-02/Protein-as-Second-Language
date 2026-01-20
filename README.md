# Protein-as-Second-Language
## Introduction
Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the “**Protein-as-Second-Language**” framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence–question–answer triples that reveal functional cues without any parameter updates. To support this process we curate a bilingual corpus of 79,860 protein–QA instances spanning attribute prediction, descriptive understanding, and extended reasoning.

## Installation
Install dependencies
```
conda create -n env python=3.10 -y
conda activate env
pip install -r requirements.txt
```
Install MMseqs2
Enable channels first (outside conda env or inside, both ok):
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```
Then install:
```
conda install mmseqs2
```
## Query-Adaptive Context Construction

Both reference and query datasets are JSON structured as:
```
[
  {
    "id": "P30291",
    "conversations": [
      { "from": "system", "value": "..." },
      { "from": "user", "value": "Amino acid: <seq> M S F ... T I Y </seq> ..." },
      { "from": "assistant", "value": "..." }
    ]
  }
]
```
User turn must contain sequence wrapped as "\<seq> M S L ... V \</seq>". To build the MMseqs reference DB and merged QA corpus:
```
python build_ref_db.py \
  --input data/Attribute-based_QA.json/Knowledge-based_QA.json/... \
  --merged-json data/ref_merged.json \
  --out-fasta data/ref.fasta \
  --mmseqs-db data/refDB \
  --skip-mmseqs
```
Then build MMseqs DB:
```
mmseqs createdb data/ref.fasta data/refDB
```
With reference DB prepared, run the retrieval module.
```
python build_contextual_exemplars.py \
  --queries data/protein_test.json \
  --reference data/ref_merged.json \
  --mmseqs-db data/refDB \
  --tfidf-cache data/ref_tfidf.joblib \
  --output data/contextual_exemplars_nonself.csv \
  --num-shots 4 \
  --w-seq 0.5 \
  --sim-threshold 0.0
```
This avoids retrieving exemplars with nearly identical sequences. To disable this behavior and allow self matches, use `--allow-self`.
