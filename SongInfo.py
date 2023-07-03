import urllib
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import re
import time as t
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import undetected_chromedriver as uc
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
from datetime import date
from typing import Iterable, List, Tuple


class Driver:
    def __init__(self, headless: bool = True):
        chromedriver_autoinstaller.install()
        ua = UserAgent(verify_ssl=False)
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument(f"user-agent={ua.random}")
        self.driver = webdriver.Chrome(options=options)


PATH = {"newSongID": "csvs/songID_new.csv", "newSongs": "csvs/songs_new.csv",
        "newArtistID": "csvs/artistID_new.csv", "songID": "csvs/songID.csv",
        "songs": "csvs/songs.csv", "artistID": "csvs/artistID.csv"}

ARTISTID_PAGE = {0: "newGenre", 1: "newAlbum", 2: "newSong", 3: "monthlyChart", 4: "weeklyChart"}


class InfoCrawler:
    def __init__(self, driver: Driver):
        self.driver = driver.driver

    def __getCSV__(self, path_key: str) -> List[int]:
        '''
        get CSV file and return as list, (songID, artistID)
        :param path_key: "newSongID", "newSongs", "newArtistID", "songID", "songs", "artistID"
        :return: list of csv content
        '''
        with open(PATH[path_key], 'r') as f:
            try:
                return_list = list(map(int, f.read().splitlines()))
            except ValueError:
                f.seek(0)
                return_list = list(map(int, f.read().splitlines()[:-1]))
            except FileNotFoundError:
                print(f"{PATH[path_key]} 경로가 존재하지 않습니다.")
        return return_list

class ArtistCrawler(InfoCrawler):
    def get_artistID(self, type: str, source: str, load: bool = False) -> List[int]:
        artistIDList = []
        if type == 'new_genre':
            for idx in range(100, 801, 100):
                url = f'https://www.melon.com/genre/song_list.htm?gnrCode=GN0{str(idx)}'
                artistIDList.extend(self.__get_artistID_from_url_new__(url=url))
        elif 'new_all' in type:
            


    def __get_artistID_from_html__(self, html: str) -> List[int]:
        soup = BeautifulSoup(html, 'html.parser')

        artist_list = list(map(str, soup.find_all("div", {"class", "ellipsis rank02"})))
        for j in range(len(artist_list)):
            try:
                artist_list[j] = int(re.findall(r'\([^)]*\)', artist_list[j])[0][2:-2])
            except:
                artist_list[j] = ''
        return artist_list

    def __get_artistID_from_url_new__(self, url: str) -> List[int]:
        '''
        get artistID by url
        :param url: url
        :return: list of artistID
        '''
        self.driver.get(url)
        html = self.driver.page_source
        artistID = self.__get_artistID_from_html__(html)
        return artistID

    # def __get_artistID_from_url_chartfinder__(self, chart: str = None, year: int = None,
    #                                            month: int = None, week: int = None,
    #                                            genre: int = 1) -> List[int]:
    #
    #     DECADE = {'2020': '1', '2010': '2', '2000': '3', '1990': '4', '1980': '5'}
    #     if (year < 1984) or (year == 1984 and month <= 3):
    #         print("차트가 존재하지 않음")
    #         raise ConnectionError
    #
    #     elif year == 1984:
    #         month -= 2
    #     elif year >= 2022 and chart == "monthly":
    #         genre += 32
    #     decade = DECADE[str(year)[:-1]+'0']
    #     # year = str(year)[-1] if decade != 1
    #
    #     #차트파인더로 이동
    #     if "https://www.melon.com/chart/search/index.htm" not in self.driver.current_url:
    #         # move to chartfinder
    #         self.driver.get("https://www.melon.com/chart/index.htm")
    #         self.driver.find_element(By.XPATH, '//*[@id="gnb_menu"]/ul[1]/li[1]/div/div/button/span').click()
    #
    #     # chart
    #     if chart == 'weekly':
    #         self.driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/h4[1]/a').click()
    #     elif chart == 'monthly':
    #         self.driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/h4[2]/a').click()
    #     else:
    #         print("차트가 존재하지 않음")
    #         raise ConnectionError
    #
    #     # decade
    #     selectList: List[Tuple[str, str]] = []
    #     selectList.append(('1', decade))
    #     selectList.append(('2', year))
    #     selectList.append(('3', str(month)))
    #     selectList.append(('4', str(week))) if chart == 'weekly' else 0
    #     selectList.append(('5', str(genre)))
    #
    #     t.sleep(1)
    #     elem = self.driver.find_element(By.CSS_SELECTOR, "input[name='p_chartType'][value='MO']")
    #     elem.click()
    #     elem =  self.driver.find_element(By.CSS_SELECTOR, "input[name='age'][value='2020']")
    #     self.driver.execute_script("arguments[0].checked = true;", elem)
    #     t.sleep(4)
    #     # for idx in selectList:
    #     #     t.sleep(0.4)
    #     #     self.driver.find_element(By.XPATH,
    #     #         f'//*[@id="d_chart_search"]/div/div/div[{idx[0]}]/div[1]/ul/li[{idx[1]}]/span/label').click()
    #
    #     #submit
    #     self.driver.find_element(By.XPATH, '//*[@id="d_srch_form"]/div[2]/button/span/span').click()
    #
    #     artistID = []
    #     for i in range(2):
    #         t.sleep(5)
    #         html = self.driver.page_source
    #         artistID.extend(self.__get_artistID_from_html__(html))
    #         if i == 0:
    #             try:
    #                 self.driver.find_element(By.XPATH, '//*[@id="frm"]/div[2]/span/a').click()
    #             except:
    #                 return artistID
    #         return artistID

if __name__ == "__main__":
    driver = Driver(headless=False)
    crawler = SongInfoCrawler(driver)
    # arr = crawler.__get_artistID_by_url_in_chartfinder__(chart='weekly',year=2023, month=1, week=1)