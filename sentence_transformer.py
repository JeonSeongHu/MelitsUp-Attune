# -*- coding: utf-8 -*-
#https://www.sbert.net/
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from PromptGenerator import generate_prompt

#pre-trained 모델 사용. https://github.com/jhgan00/ko-sentence-transformers 참고.
# embedder1 = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda')
embedder = SentenceTransformer("BM-K/KoDiffCSE-RoBERTa", device='cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

songs = pd.read_csv("SBERT/songs_new2.csv")

# #load embeddings -이미 만들어진 임베딩을 불러올 때 사용
corpus_embeddings = np.load('csvs/new_lyrics_embeddings.npy')
corpus_embeddings = torch.as_tensor(corpus_embeddings, device=device)

# save embeddings - 임베딩을 새로 만들 때 사용
# song_lyrics = (songs['lines']).to_list()
# corpus = song_lyrics
# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
# np.save('csvs/new_lyrics_embeddings', corpus_embeddings.cpu().numpy())


# Query sentences / 여기에 검색할 문장 넣기. 여러개를 한 번에 넣어도 됨.
client_id = "UbbwWcuXQsRrA_AaQLiO" # 개발자센터에서 발급받은 Client ID 값
client_secret = "5WNeGXBPXR" # 개발자센터에서 발급받은 Client Secret 값

prompt = generate_prompt("there is a cat laying on a bed with a pillow", "happy", client_id, client_secret)
print(prompt)
queries = prompt
# queries = ["자유로운 우리를 봐 자유로워"]
top_k = 1000
top_real = 30 # 실제 출력할 개수

for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].cpu()

    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
    print("\n\n======================\n")
    print("Query:", query)
    print(f"Top {str(top_real)} most similar songs")
    i = 0
    for idx in top_results[0:top_k]:

        song_year = songs["date"].iloc[int(idx)][:4]
        # 필터링 조건, 해당 조건에서는 2000년부터 2022년 사이에 발매된 곡만 불러오게 했음.
        if song_year != '-' and int(song_year) >= 2012 :
            i += 1
            print(f"{str(i)}: {songs['song_name'].iloc[int(idx)]} - {songs['artist'].iloc[int(idx)]} "
                  f"(Score: {cos_scores[idx]:.4f}) - {songs['date'].iloc[int(idx)]}")

            # for k, line in enumerate(songs['lyrics'].iloc[int(idx)].strip().splitlines()):
            #     if not line:
            #         continue
            #     linenum = k+1
            #     print(line.strip(), end="\n" if linenum % 2 == 0 else " ")
            # print()
            if i == top_real:
                break
