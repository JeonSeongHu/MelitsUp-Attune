#https://github.com/jhgan00/ko-sentence-transformers
#https://www.sbert.net/
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda')

songs = pd.read_csv("songs.csv")
songs.dropna(inplace=True)

#load embeddings
corpus_embeddings = np.load('lyrics_embeddings.npy')
corpus_embeddings = torch.as_tensor(corpus_embeddings, device="cuda")

# save embeddings
# song_lyrics = (songs['lyrics'].str.replace('\n', ' ')).to_list()
# corpus = song_lyrics
# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
# np.save('lyrics_embeddings', corpus_embeddings.cpu().numpy())

# Query sentences
queries = [""]

top_k = 1000
top_real = 15
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]

    print("\n\n======================\n")
    print("Query:", query)
    print("\nTop 10 most similar songs\n")
    i = 0
    for idx in top_results[0:top_k]:
        song_date = songs["date"].iloc[int(idx)][:4]
        if song_date != '-' and int(song_date) >= 2020:
            i += 1
            print(f"{str(i)}: {songs['song_name'].iloc[int(idx)]} - {songs['artist'].iloc[int(idx)]} "
                  f"(Score: {cos_scores[idx]:.4f})")
            if i == top_real:
                break