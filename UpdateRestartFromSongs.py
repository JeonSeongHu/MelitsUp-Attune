#https://www.sbert.net/
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from SingersCrawler import get_artistID_in_new, get_songID_by_artist
from SongCrawler import get_song_by_id
import chromedriver_autoinstaller
from fake_useragent import UserAgent
from selenium import webdriver
import pandas as pd
import os

chromedriver_autoinstaller.install()
ua = UserAgent(verify_ssl=False)
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument(f"user-agent={ua.random}")
driver = webdriver.Chrome(options=options)


with open("csvs/songID.csv", 'r') as f:
    try :
        songId_list = list(map(int, f.read().splitlines()))
    except:
        f.seek(0)
        songId_list = list(map(int, f.read().splitlines()[:-1]))
        print(songId_list)

with open("csvs/artistID.csv", 'r') as f:
    artistId_list = list(map(int, f.read().splitlines()))

# new SongID로 가사 크롤링
with open("csvs/songID_new.csv", 'r') as f:
    new_songID = list(set(list(map(int, f.read().splitlines()))))
    new_songID.sort()
pd.Series(new_songID).to_csv("csvs/songID_new.csv", mode="w", index=False, header=False)

if os.path.isfile("csvs/songs_new.csv"):
    csv = pd.read_csv("csvs/songs_new.csv", index_col=0)
    startindex = new_songID.index(csv.tail(1).index)+1
else:
    startindex = 0

# songID로 song lyrics 크롤링
for i, songID in enumerate(new_songID[startindex:]):
    try:
        song_info = get_song_by_id(songID)
        print(f"{startindex+i+1}: ", end="")
        print(song_info)
        if i == 0 and not os.path.isfile("csvs/songs_new.csv"):
            song_info.to_csv("csvs/songs_new.csv", mode='w')
        else:
            song_info.to_csv("csvs/songs_new.csv", mode='a', header=False)

    except:
        print(f"ID {songID} 차단됨.")
        exit(1)

print("end")

# 임베딩 하기
embedder1 = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda')
embedder2 = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device='cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
new_songs = pd.read_csv("csvs/songs_new.csv", index_col=0)
songs = pd.read_csv("csvs/songs.csv", index_col=0)
songs.dropna(inplace=True)
new_songs.dropna(inplace=True)

#save embeddings - 임베딩을 새로 만들 때 사용
corpus_embeddings1 = np.load('csvs/lyrics_embeddings1.npy')
corpus_embeddings1 = torch.as_tensor(corpus_embeddings1, device=device)
corpus_embeddings2 = np.load('csvs/lyrics_embeddings2.npy')
corpus_embeddings2 = torch.as_tensor(corpus_embeddings2, device=device)
corpus_embeddings = (corpus_embeddings1 + corpus_embeddings2) / 2

song_lyrics = (new_songs['lyrics']).to_list()
corpus = song_lyrics

corpus_embeddings_new_1 = embedder1.encode(corpus, convert_to_tensor=True)
corpus_embeddings_new_2 = embedder2.encode(corpus, convert_to_tensor=True)
corpus_embeddings_new = (corpus_embeddings_new_1 + corpus_embeddings_new_2) / 2

corpus_embeddings1 = torch.cat([corpus_embeddings1, corpus_embeddings_new_1], dim=0)
corpus_embeddings2 = torch.cat([corpus_embeddings2, corpus_embeddings_new_2], dim=0)
corpus_embeddings = torch.cat([corpus_embeddings, corpus_embeddings_new], dim=0)

np.save('csvs/lyrics_embeddings1', corpus_embeddings1.cpu().numpy())
np.save('csvs/lyrics_embeddings2', corpus_embeddings2.cpu().numpy())
np.save('csvs/lyrics_embeddings', corpus_embeddings.cpu().numpy())
#
#
# csv 파일 합치기, songs
songs = pd.concat([songs, new_songs], axis=0)
songs.to_csv("csvs/songs.csv")

# csv 파일 합치기, songID
songId_list += new_songID
songId_list = list(set(songId_list))
pd.Series(songId_list).to_csv("csvs/songID.csv", index=False, header=False)

# csv 파일 합치기, artistID
f = open("csvs/artistID_new.csv", 'r')
artists_to_get_all_pages = list(map(int, f.read().splitlines()))
artistId_list += artists_to_get_all_pages
artistId_list = list(set(artistId_list))

pd.Series(artistId_list).to_csv("csvs/artistID.csv", index=False, header=False)


# Query sentences / 여기에 검색할 문장 넣기. 여러개를 한 번에 넣어도 됨.
queries = [""""산타는 엄마 아빠 둘 중 하나일 테고
죠지 죠지는 죽었지만
매년 이맘때 되면 아직 들뜨는 건
너 때문도 아니야
Ooh la 크리스마스다
외로웁고 좋은 날 oh
oh darling kiss me babe
그냥 혼자 해본 말이야
산타는 선물 같은 눈칫밥을 주고
캐빈 캐빈은 늙었지만
너도 기다렸잖아 아닌 척하지 마
lonely Christmas it’s okay
Ooh la 크리스마스다
외로웁고 좋은 날 oh
oh darling kiss me babe
그냥 혼자 해본 말이야
앙상한 가지 위에도
떠도는 공기 속에도
방구석 처박혀 보는 영화에도
made in Christmas time
love actually is all around you
Ooh la it’s Christmas again
Ooh la it’s Christmas again
눈이라도 내려줘
온 세상 하얗게 물들여줘
Ooh la 크리스마스다
외로웁고 좋은 날 oh
oh darling kiss me babe
그냥 해본 말은 아니야
그냥 해본 말은 아닌데
oh lonely Christmas it’s ok
Christmas Christmas
We made in Christmas time
Christmas Christmas
We made in Christmas time
Christmas Christmas
We made in Christmas time
Christmas Christmas
We made in Christmas time"""]

top_k = 1000
top_real = 15 # 실제 출력할 개수
for query in queries:
    query_embedding1 = embedder1.encode(query, convert_to_tensor=True)
    query_embedding2 = embedder2.encode(query, convert_to_tensor=True)
    query_embedding = (query_embedding2 + query_embedding1) / 2
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]

    print("\n\n======================\n")
    print("Query:", query)
    print(f"Top {str(top_real)} most similar songs")
    i = 0
    for idx in top_results[0:top_k]:
        song_year = songs["date"].iloc[int(idx)][:4]
        # 필터링 조건, 해당 조건에서는 2000년부터 2022년 사이에 발매된 곡만 불러오게 했음.
        if song_year != '-' and 2023 >= int(song_year) >= 2010:
            i += 1
            print(f"{str(i)} {idx}: {songs['song_name'].iloc[int(idx)]} - {songs['artist'].iloc[int(idx)]} "
                  f"(Score: {cos_scores[idx]:.4f})")
            if i == top_real:
                break