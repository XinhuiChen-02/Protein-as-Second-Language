# Protein-as-Second-Language
## Introduction
Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the “**Protein-as-Second-Language**” framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence–question–answer triples that reveal functional cues without any parameter updates. To support this process we curate a bilingual corpus of 79,860 protein–QA instances spanning attribute prediction, descriptive understanding, and extended reasoning.

## Installation
```
conda create -n env python=3.10 -y
conda activate env
pip install -r requirements.txt
conda install -c conda-forge mmseqs2
（环境外）
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install mmseqs2
python build_ref_db.py \
  --input autodl-tmp/Attribute-based_QA.json \
  --merged-json data/ref_merged.json \
  --out-fasta data/ref.fasta \
  --mmseqs-db data/refDB \
  --skip-mmseqs
```
