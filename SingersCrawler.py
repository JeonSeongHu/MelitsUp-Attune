from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import chromedriver_autoinstaller
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import pandas as pd
import time as t
import re

chromedriver_autoinstaller.install()
ua = UserAgent(verify_ssl=False)

options = uc.ChromeOptions()
options.add_argument(f"user-agent={str(ua.random)}")
driver = uc.Chrome(options=options)
driver.implicitly_wait(200)

def get_singer(artistId):
    songID_list = []
    idx = 1
    while True:
        t.sleep(1)
        url = f"https://www.melon.com/artist/song.htm?artistId={str(artistId)}#params%5BlistType%5D=A&params%5BorderBy" \
              f"%5D=ISSUE_DATE&params%5BartistId%5D={str(artistId)}&po=pageObj&startIndex={str(idx)}"
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        soup_list = soup.select(".input_check")[1:]
        now_list = list(map(lambda x : x['value'], soup_list))
        if len(now_list) == 0:
            return list(set(songID_list))
        song_list = list(map(lambda x : x['title'], soup_list))

        disabled = []
        for i in soup_list:
            try:
                disabled.append(True if i['disabled'] == 'disabled' else False)
            except KeyError:
                disabled.append(False)
        for i, name in enumerate(song_list):
            if '(Inst.)' in name or '(inst.)' in name or '(Instrumental)' in name or '[Instrumental]' in name\
                    or disabled[i]:
                now_list[i] = ''
        now_list = [i for i in now_list if i]

        if idx > 4000:
            raise ConnectionError
        songID_list.extend(now_list)
        idx += 50


def artistId_list_crawler(year, m, isopen=False):
    artistId_list = []
    d = (2029 - year) // 10 + 1
    if d == 1:
        y = 3 - int(str(year)[-1]) # 2023년 되면 4 - 로 바꾸어야 함
    else:
        y = 10 - int(str(year)[-1])

    if not isopen:
        url = 'https://www.melon.com/chart/index.htm'
        driver.get(url)
        #차트파인더 클릭
        driver.find_element(By.XPATH,'//*[@id="gnb_menu"]/ul[1]/li[1]/div/div/button/span').click()
        #월간차트 클릭
        driver.find_element(By.XPATH,'//*[@id="d_chart_search"]/div/h4[2]/a').click()

    for idx in [('1',str(d)), ('2',str(y)), ('3',str(m)), ('5','1' if year != 2022 else '3' )]:
        t.sleep(0.2)
        driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[{idx[0]}]/div[1]/ul/li[{idx[1]}]/span/label').click()
    driver.find_element(By.XPATH, '//*[@id="d_srch_form"]/div[2]/button/span/span').click()

    for i in range(2):
        t.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        now_list = list(map(str, soup.find_all("div", {"class", "ellipsis rank02"})))
        for j in range(len(now_list)):
            try:
                now_list[j] = int(re.findall(r'\([^)]*\)', now_list[j])[0][2:-2])
            except:
                now_list[j] = ''
        artistId_list += now_list
        if i == 0:
            driver.find_element(By.XPATH, '//*[@id="frm"]/div[2]/span/a').click()

    return artistId_list

# if __name__ == "__main__":
#     trg = False
#     artistId_list = []
#     for y in range(2000, 2023):
#         for m in range(1, 13):
#             csv = pd.Series(artistId_list_crawler(y, m, trg))
#             csv.to_csv("artistId.csv", mode='a')
#             trg = True

if __name__ == "__main__":
    f = open("artistId.csv", 'r')
    try:
        csv = pd.read_csv("songID.csv")
        id_start = int(csv.iloc[-1][0].split(':')[1])
        print(id_start)
    except (FileNotFoundError, AttributeError):
        id_start = 3114174

    artistId_list = list(map(lambda x : int(x.strip()), f.readlines()))
    print(artistId_list)
    for id in artistId_list[artistId_list.index(id_start):]:
        try:
            csv = pd.Series(get_singer(id))
            csv.to_csv("songId.csv", mode='a', index=False)
        except:
            csv = pd.Series([f"lastid:{id}"])
            csv.to_csv("songId.csv", mode='a', index=False)
            break

    f.close()