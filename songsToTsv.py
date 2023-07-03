import pandas as pd
import numpy as np
import csv

songs = pd.read_csv("csvs/songs.csv", index_col=0)
songs["song_name"] = songs["song_name"].apply(lambda x: x.strip("FREE\r\n\t") if "\n" in x else x)
songs["title_artist"] = songs["song_name"] + '-' + songs["artist"]
songs["None"] = " "
songs.drop(["lyrics"], axis=1, inplace=True)
songs.to_csv("metadata.tsv", sep="\t", index=False,)

corpus_embeddings = np.load('csvs/lyrics_embeddings1.npy')
with open('tensor1.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in corpus_embeddings:
        tsv_writer.writerow(i)

corpus_embeddings = np.load('csvs/lyrics_embeddings2.npy')
with open('tensor2.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in corpus_embeddings:
        tsv_writer.writerow(i)