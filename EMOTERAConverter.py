# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped EMOTERA dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--english', default="True", help='Convert English tweets to BERT friendly shape')
parser.add_argument('--filipino', default="False", help='Convert Filipino tweets to BERT friendly shape')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
INCLUDE_ENGLISH = args.english.upper() == "TRUE"
INCLUDE_FILIPINO = args.filipino.upper() == "TRUE"

if INCLUDE_ENGLISH and INCLUDE_FILIPINO:
    EMOTERA_dataframe = pd.read_csv(INPUT_PATH+"/EMOTERA-All.tsv", sep="\t")
elif INCLUDE_FILIPINO:
    EMOTERA_dataframe = pd.read_csv(INPUT_PATH+"/EMOTERA-Fil.tsv", sep="\t")
else:
    EMOTERA_dataframe = pd.read_csv(INPUT_PATH+"/EMOTERA-En.tsv", sep="\t")
    
EMOTERA_dataframe = pd.DataFrame({"dialogue": EMOTERA_dataframe["tweet"], "emotion": EMOTERA_dataframe["emotion"]})

fraction = 0.2

np.random.seed(seed=42)

test_indices = np.random.choice(EMOTERA_dataframe.index, size=int(round(fraction*EMOTERA_dataframe.shape[0])), replace=False)
train_indices = EMOTERA_dataframe.index.difference(test_indices)
dev_indices = np.random.choice(train_indices, size=int(round(fraction*len(train_indices))), replace=False)
train_indices = train_indices.difference(dev_indices)

EMOTERA_train = EMOTERA_dataframe.loc[train_indices,:]
EMOTERA_dev = EMOTERA_dataframe.loc[dev_indices,:]
EMOTERA_test = EMOTERA_dataframe.loc[test_indices,:]

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
    
EMOTERA_train.to_csv(OUTPUT_PATH+"/train.tsv", sep='\t', encoding="utf-8")
EMOTERA_dev.to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
EMOTERA_test.to_csv(OUTPUT_PATH+"/test.tsv", sep='\t', encoding="utf-8")