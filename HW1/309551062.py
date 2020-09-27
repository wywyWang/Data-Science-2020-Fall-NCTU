import sys
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

header_URL = 'https://www.ptt.cc'
headers = {'cookie': 'over18=1;', 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

# crawler output filename
all_articles = './all_articles.txt'
all_popular = './all_popular.txt'

# parameters
SLEEP_INTERVAL = 0.02
YEAR = '2019'
START = 2748    # start page is 2748
END = 3143      # end page is 3143

NEED_TO_CHECK_POST = ['https://www.ptt.cc/bbs/Beauty/M.1577354483.A.D9D.html?fbclid=IwAR0h3_kb3-pSiYHIowmiTneSmpElFzor0HNgY3IoKTVTOtERzuYM2KunoSo', 'https://www.ptt.cc/bbs/Beauty/M.1578210772.A.06E.html']


def get_post_year(post_url, page_index):
    time.sleep(SLEEP_INTERVAL)
    content_response = requests.get(post_url, headers=headers)
    content_soup = BeautifulSoup(content_response.text, "html.parser")
    # Filter like https://www.ptt.cc/bbs/Beauty/M.1549974705.A.611.html
    if content_soup.select("#main-content"):
        main_content = content_soup.select("#main-content")[0].get_text("|")
    else:
        return None
    if "※ 發信站" in main_content:
        if page_index == START or page_index == END:
            # ['Fri', 'Sep', '25', '10:10:33', '2020']
            time_index = main_content.split("|").index("時間")
            content_time = main_content.split("|")[time_index + 1].split(" ")
            return content_time[-1]
        else:
            return YEAR
    else:
        return None


def get_each_page(page_index):
    time.sleep(SLEEP_INTERVAL)
    cheat_page = '/bbs/Beauty/index'+ str(page_index) +'.html'
    page_url = header_URL + cheat_page
    response = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # open txt files
    articles_file = open(all_articles, 'a+')
    popular_file = open(all_popular, 'a+')
    
    # get post info in every page
    post_info = soup.select("div.title")
    post_popular = soup.select("div.nrec")
    post_date = soup.select("div.date")

    for idx, item in enumerate(post_info):
        # filter out no content url (like delete) and announcement
        if item.find("a") != None and '[公告]' not in item.text:
            post_link = header_URL + item.select_one("a").get("href")
            post_title = item.text.replace("\n", "")
            post_month, post_day = post_date[idx].text.strip().split('/')
            post_date_convert = post_month + post_day

            post_year = get_post_year(post_link, page_index)
            if post_year == YEAR:
                # check have popular count
                if post_popular[idx].find("span") != None:
                    post_count = post_popular[idx].find("span").text
                else:
                    post_count = None
                # get popular post info
                if post_count == '爆':
                    popular_file.write(post_date_convert + ',' + post_title + ',' + post_link + '\n')
                articles_file.write(post_date_convert + ',' + post_title + ',' + post_link + '\n')
            else:
                pass
    articles_file.close()
    popular_file.close()


def crawl_PTT():
    if os.path.exists(all_articles):
        os.remove(all_articles)
    if os.path.exists(all_popular):
        os.remove(all_popular)

    record_count = 0
    for page in range(START, END+1):
        if record_count % 50 == 0:
            print("STATUS: {}, PAGE: {}".format(record_count, page))
        get_each_page(page)
        record_count += 1
        # if record_count == 21:
        #     break

if __name__ == '__main__':
    #get parameters
    # print(sys.argv)
    functions = sys.argv[1]
    
    # function crawl
    if functions == 'crawl':
        crawl_PTT()
        # print("DONEEEEEE")
    # function push
    
    # function popular
    
    # function keyword