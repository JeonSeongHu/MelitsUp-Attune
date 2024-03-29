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

try:
    os.remove("csvs/artistID_new.csv")
    os.remove("csvs/songID_new.csv")
    os.remove("csvs/songs_new.csv")
except:
    pass

with open("csvs/songID.csv", 'r') as f:
    try :
        songId_list = list(map(int, f.read().splitlines()))
    except:
        f.seek(0)
        songId_list = list(map(int, f.read().splitlines()[:-1]))
        print(songId_list)

with open("csvs/artistID.csv", 'r') as f:
    artistId_list = list(map(int, f.read().splitlines()))

# new song artitst 찾기
arr = get_artistID_in_new(driver=driver)

# new Artist 여부 찾기
artists_to_get_all_pages, artists_to_get_one_page = [], []
for artist in arr:

    if artist not in artistId_list:
        artists_to_get_all_pages.append(artist)
    else:
        artists_to_get_one_page.append(artist)

artists_to_get_all_pages = list(set(artists_to_get_all_pages))
artists_to_get_one_page = list(set(artists_to_get_one_page))

# #### 원하는 가수 추가 시 ####
# f = open("csvs/artistID_new.csv", 'r')
# artists_to_get_all_pages = list(map(int, f.read().splitlines()))
# f.close()
# #################################

print(f"new artist: {len(artists_to_get_all_pages)}")
print(f"existing artist: {len(artists_to_get_one_page)}")
pd.Series(artists_to_get_all_pages).to_csv("csvs/artistID_new.csv", mode='a', index=False, header=False)

# Artist로 Song ID 크롤링, 새 가수
new_songID = []
for artist in artists_to_get_all_pages:
    try:
        id = get_songID_by_artist(artist,driver=driver,time_wait=0)
        pd.Series(id).to_csv("csvs/songID_new.csv", mode='a', index=False, header=False)
        if not id:
            print(f"artist ID {artist} 중 차단됨.")
            exit(1)
        print(f"artist ID {artist} Done")
    except:
        print(f"artist ID {artist} 중 차단됨.")
        exit(1)

# Artist로 Song ID 크롤링, 이미 있는 가수
tmp_songID = []
for artist in artists_to_get_one_page:
    try:
        id = get_songID_by_artist(artist,driver=driver,num_pages=1,time_wait=0)
        pd.Series(id).to_csv("csvs/songID_new.csv", mode='a', index=False, header=False)
        if not id:
            print(f"artist ID {artist} 중 차단됨.")
            exit(1)
        print(f"artist ID {artist} Done")
    except:
        print(f"artist ID {artist} 중 차단됨.")
        exit(1)

# new SongID로 가사 크롤링
with open("csvs/songID_new.csv", 'r') as f:
    new_songID = list(set(list(map(int, f.read().splitlines()))))
    print(len(new_songID))
    new_songID = [i for i in new_songID if i not in songId_list]
    print(len(new_songID))
    new_songID.sort()


# songID로 song lyrics 크롤링
for i, songID in enumerate(new_songID):
    try:
        song_info = get_song_by_id(songID)
        print(f"{i+1}: ", end="")
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
queries = ["""] """]

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