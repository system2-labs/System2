#!/usr/bin/env python3
"""build_pure_electrical_dataset.py - Purge physics trivia and assemble pure electrical & vocational data."""

from datasets import load_from_disk, concatenate_datasets, DatasetDict
import os

# 1. Load original electrical dataset
orig_ds = load_from_disk('./data/electrical')

# 2. Filter out physics trivia (In-Domain Causal Purity Law)
pure_train = orig_ds['train'].filter(lambda x: x['source'] in ('ee_reason_bench', 'mmlu_electrical_engineering'))
pure_test = orig_ds['test'].filter(lambda x: x['source'] in ('ee_reason_bench', 'mmlu_electrical_engineering'))

print(f"Purged physics trivia: train {len(orig_ds['train'])} -> {len(pure_train)}, test {len(orig_ds['test'])} -> {len(pure_test)}")

# 3. Ingest vocational electrician dataset if present
voc_path = '../data-enhancement/data/electrical_vocational'
if os.path.exists(voc_path):
    voc_ds = load_from_disk(voc_path)
    combined_train = concatenate_datasets([pure_train, voc_ds['train']])
    combined_test = concatenate_datasets([pure_test, voc_ds['test']])
    print(f"Added vocational dataset: train {len(combined_train)}, test {len(combined_test)}")
else:
    combined_train = pure_train
    combined_test = pure_test

pure_ds = DatasetDict({'train': combined_train, 'test': combined_test})
out_path = './data/electrical_pure'
pure_ds.save_to_disk(out_path)
print(f"Saved pure dataset to {out_path}")
