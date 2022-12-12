import urllib
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import re
import time as t
from fake_useragent import UserAgent


ua = UserAgent(verify_ssl=False)
headers = {
    'user-agent': ua.random,
}

def get_song_by_id(id):
        # t.sleep(0.5)
    # try:
        url = "https://www.melon.com/song/detail.htm?songId="+str(id)
        req = urllib.request.Request(url, headers = headers)

        html = urlopen(req)
        soup = BeautifulSoup(html, "lxml")
        if not soup.text.strip():

            return None

        song_name = soup.find('div', {"class" : "song_name"})

        adults = 1 if song_name.find('span', {"class" : "bullet_icons age_19 large"}) else 0
        song_name = song_name.text.lstrip('곡명19금\n\t').strip()
        artist = soup.find('div', {"class" : "artist"}).text.strip()
        tmp = soup.find('dl', {"class" : "list"})

        album_id = str(soup.find('dl', {"class" : "list"}).find("a"))
        album_id = re.findall('\(([^)]+)', album_id)[0]
        tmp = tmp.text.split('\n')
        date, genre = tmp[tmp.index('발매일')+1], tmp[tmp.index('장르')+1]
        lyrics = str(soup.find('div', {"class" : "lyric"}))[72:-6].strip().replace('<br/>', '\n')

        return pd.DataFrame({'song_name':song_name, 'adults':adults,
                             'artist':artist, 'album_id':album_id, 'date':date, 'genre':genre, 'lyrics':lyrics},
                            index=[str(id)])

    # except:
    #     return "error"
try:
    csv = pd.read_csv("songs.csv", index_col = 0)
    start = max(csv.index)
    print(start)
    fileNotExist = False
except:
    start = 2073
    fileNotExist = True

df_list = []
for i in range(start+1,100000000):
    print(i)
    tmp = get_song_by_id(i)
    if tmp is not None:
        if fileNotExist:
            tmp.to_csv("songs.csv")
            fileNotExist = False
        else:
            tmp.to_csv("songs.csv", mode='a', header=False)
        print(f"id {i} writed,")
        print(tmp)

