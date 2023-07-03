from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import chromedriver_autoinstaller
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import pandas as pd
import time as t
import re

# 한 가수의 곡 ID 가져오기
def get_songID_by_artist(artistId, num_pages=1000, driver=None, time_wait=1):
    t.sleep(time_wait)
    songID_list = []
    idx = 1

    if driver == None:
        chromedriver_autoinstaller.install()
        ua = UserAgent(verify_ssl=False)
        options = uc.ChromeOptions()
        options.add_argument(f"user-agent={str(ua.random)}")
        options.add_argument("--headless")
        driver = uc.Chrome(options=options)

    while True:
        url = f"https://www.melon.com/artist/song.htm?artistId={str(artistId)}#params%5BlistType%5D=A&params%5BorderBy" \
              f"%5D=ISSUE_DATE&params%5BartistId%5D={str(artistId)}&po=pageObj&startIndex={str(idx)}"
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        soup_list = soup.select(".input_check")[1:]
        now_list = list(map(lambda x : x['value'], soup_list))

        # 리스트가 비었을 시 (마지막 페이지)
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
                    or '(MR)' in name or disabled[i]:
                now_list[i] = ''
        now_list = [i for i in now_list if i]

        # 차단당했을 시
        if idx > 100000:
            raise ConnectionError

        num_pages -= 1
        songID_list.extend(now_list)
        idx += 50


        if num_pages == 0:
            return list(set(songID_list))


def get_artistID_in_monthly_chart(year, m, isopen=False, driver = None):
    if driver == None:
        chromedriver_autoinstaller.install()
        ua = UserAgent(verify_ssl=False)
        options = uc.ChromeOptions()
        options.add_argument(f"user-agent={str(ua.random)}")
        options.add_argument("headless")
        driver = uc.Chrome(options=options)

    artistId_list = []
    d = (2029 - year) // 10 + 1
    if d == 1:
        y = 4 - int(str(year)[-1]) # 2023년 되면 4 - 로 바꾸어야 함
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


def get_artistID_in_new(driver = None):
    if driver == None:
        chromedriver_autoinstaller.install()
        ua = UserAgent(verify_ssl=False)
        options = uc.ChromeOptions()
        options.add_argument(f"user-agent={str(ua.random)}")
        options.add_argument("--headless")
        driver = uc.Chrome(options=options)

    song_list = []
    for idx in range(100,801,100):
        url = f'https://www.melon.com/genre/song_list.htm?gnrCode=GN0{str(idx)}'
        driver.get(url)

        t.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        now_list = list(map(str, soup.find_all("div", {"class", "ellipsis rank02"})))
        for j in range(len(now_list)):
            try:
                now_list[j] = int(re.findall(r'\([^)]*\)', now_list[j])[0][2:-2])
            except:
                now_list[j] = ''
        song_list.extend(now_list)

    return song_list

# artist ID in monthly chart
# if __name__ == "__main__":
#     trg = False
#     artistId_list = []
#     for y in range(2000, 2023):
#         for m in range(1, 13):
#             csv = pd.Series(get_artistID_in_monthly_chart(y, m, trg))
#             csv.to_csv("csvs/artistID.csv", mode='a')
#             trg = True

# artist ID in New Song list
# if __name__ == "__main__":
#     artistId_list = []
#     csv = pd.Series(get_artistID_in_new())
#     csv.to_csv("csvs/artistID.csv", mode='a', index=False)


# song ID
if __name__ == "__main__":
    chromedriver_autoinstaller.install()
    ua = UserAgent(verify_ssl=False)
    options = uc.ChromeOptions()
    options.add_argument(f"user-agent={str(ua.random)}")
    driver = uc.Chrome(options=options)
    f = open("csvs/artistID.csv", 'r')
    try:
        csv = pd.read_csv("csvs/songID.csv")
        songID_start = int(csv.iloc[-1][0].split(':')[1])
        csv = csv[:-1]
        csv.to_csv("csvs/songID.csv", mode='w', index=False)
        print(songID_start)
    except (FileNotFoundError, AttributeError, IndexError, pd.errors.EmptyDataError):
        songID_start = 15

    artistId_list = list(map(lambda x : int(x.strip()), f.readlines()))
    print(artistId_list)
    for id in artistId_list[artistId_list.index(songID_start):]:
        try:
            csv = pd.Series(get_songID_by_artist(id, driver=driver))
            csv.to_csv("csvs/songID.csv", mode='a', index=False, header=False)
        except:
            csv = pd.Series([f"lastid:{id}"])
            csv.to_csv("csvs/songID.csv", mode='a', index=False, header=False)
            break

    f.close()