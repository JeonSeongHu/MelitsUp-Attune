#https://github.com/jhgan00/ko-sentence-transformers
#https://www.sbert.net/
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import random

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda')
# Corpus with example sentences

songs = pd.read_csv("songs.csv")
songs.dropna(inplace=True)

#load embeddings
corpus_embeddings = np.load('dot_lyrics_embeddings.npy')
corpus_embeddings = torch.as_tensor(corpus_embeddings, device="cuda")

# save embeddings
# song_lyrics = (songs['lyrics'].str.replace('\n', ',')).to_list()
# corpus = song_lyrics
# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
# np.save('lyrics_embeddings', corpus_embeddings.cpu().numpy())

# Query sentences:
queries = ["mbti"]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 100
top_real = 30
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n")
    print("Query:", query)
    print("\nTop 100 most similar songs\n")

    for i, idx in enumerate(top_results[0:top_k]):
        # if int(songs["date"].iloc[int(idx)][:4]) >= 2010:
            # print(corpus[idx][:100], "(Score: %.4f)" % (cos_scores[idx]))
            print(str(i+1) + ': ' +songs["song_name"].iloc[int(idx)] + " - " + songs["artist"].iloc[int(idx)], "(Score: %.4f)" % (cos_scores[idx]))