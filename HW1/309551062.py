import sys
import os
import time
import csv
import threading
import requests
import pandas as pd
from tqdm import tqdm
from collections import Counter
from bs4 import BeautifulSoup

header_URL = 'https://www.ptt.cc'
headers = {'cookie': 'over18=1;', 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

# crawler output filename
all_articles = './all_articles.txt'
all_popular = './all_popular.txt'
raw_articles = './raw_articles.txt'

# parameters
SLEEP_INTERVAL = 0.1
YEAR = '2019'
START = 2748    # start page is 2748
END = 3143      # end page is 3143
THREAD = 6
STEP = (END - START + 1) // THREAD
SEP = '#,#'

NEED_TO_CHECK_POST = ['https://www.ptt.cc/bbs/Beauty/M.1549974705.A.611.html', 'https://www.ptt.cc/bbs/Beauty/M.1577354483.A.D9D.html?fbclid=IwAR0h3_kb3-pSiYHIowmiTneSmpElFzor0HNgY3IoKTVTOtERzuYM2KunoSo', 'https://www.ptt.cc/bbs/Beauty/M.1578210772.A.06E.html']


def get_post_year(post_url, page_index):
    time.sleep(SLEEP_INTERVAL)
    content_response = requests.get(post_url, headers=headers)
    content_soup = BeautifulSoup(content_response.text, "html.parser")
    # Filter like https://www.ptt.cc/bbs/Beauty/M.1549974705.A.611.html
    if content_soup.select("#main-content"):
        main_content = content_soup.select("#main-content")[0].get_text("|")
    else:
        return None
    # if "※ 發信站" in main_content:
    if page_index == START or page_index == END:
        # ['Fri', 'Sep', '25', '10:10:33', '2020']
        time_index = main_content.split("|").index("時間")
        content_time = main_content.split("|")[time_index + 1].split(" ")
        return content_time[-1]
    else:
        return YEAR
    # else:
    #     return None


def get_article_push_count(article_link, push_count, boo_count):
    time.sleep(SLEEP_INTERVAL)
    content_response = requests.get(article_link, headers=headers)
    content_soup = BeautifulSoup(content_response.text, "html.parser")
    push_type = content_soup.find_all('span', {'class': 'push-tag'})
    push_user = content_soup.find_all('span', {'class': 'push-userid'})

    if content_soup.select("#main-content"):
        main_content = content_soup.select("#main-content")[0].get_text("|")
    else:
        main_content = []
    if "※ 發信站" in main_content:
        for each_type, each_user in zip(push_type, push_user):
            if each_type.text == '推 ':
                push_count.update([each_user.text])
            elif each_type.text == '噓 ':
                boo_count.update([each_user.text])
            else:
                pass
    else:
        return push_count, boo_count
    return push_count, boo_count


def get_each_page(page_index):
    time.sleep(SLEEP_INTERVAL)
    cheat_page = '/bbs/Beauty/index' + str(page_index) + '.html'
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

            if page_index == START:
                if post_date_convert == '101':
                    # check have popular count
                    if post_popular[idx].find("span") != None:
                        post_count = post_popular[idx].find("span").text
                    else:
                        post_count = None
                    # get popular post info
                    if post_count == '爆':
                        popular_file.write(post_date_convert + SEP + str(page_index) + SEP + str(idx) + SEP + post_title + SEP + post_link + '\n')
                    articles_file.write(post_date_convert + SEP + str(page_index) + SEP + str(idx) + SEP + post_title + SEP + post_link + '\n')                    
                else:
                    pass
            elif page_index == END:
                if post_date_convert == '1231':
                    # check have popular count
                    if post_popular[idx].find("span") != None:
                        post_count = post_popular[idx].find("span").text
                    else:
                        post_count = None
                    # get popular post info
                    if post_count == '爆':
                        popular_file.write(post_date_convert + SEP + str(page_index) + SEP + str(idx) + SEP + post_title + SEP + post_link + '\n')
                    articles_file.write(post_date_convert + SEP + str(page_index) + SEP + str(idx) + SEP + post_title + SEP + post_link + '\n')                    
                else:
                    pass
            else:
                # check have popular count
                if post_popular[idx].find("span") != None:
                    post_count = post_popular[idx].find("span").text
                else:
                    post_count = None
                # get popular post info
                if post_count == '爆':
                    popular_file.write(post_date_convert + SEP + str(page_index) + SEP + str(idx) + SEP + post_title + SEP + post_link + '\n')
                articles_file.write(post_date_convert + SEP + str(page_index) + SEP + str(idx) + SEP + post_title + SEP + post_link + '\n')
    articles_file.close()
    popular_file.close()


def crawl_PTT(thread_num):
    # 396/6 = 66, 2748 ~ 2814
    start = START + STEP * thread_num
    end = start + STEP
    # print(start, end - 1)
    for page in tqdm(range(start, end), desc='Thread: {}, {} ~ {}'.format(thread_num, start, end)):
        get_each_page(page)


def push(start_date, end_date):
    articles_file = open(all_articles, 'r')
    article_info = articles_file.readlines()
    article_match_link = []

    # get link between start and end date
    for article in article_info:
        article_date = int(article.split(',')[0])
        if article_date >= start_date and article_date <= end_date:
            article_link = article.split(',')[-1].replace('\n', '')
            article_match_link.append(article_link)
    
    push_count = Counter()
    boo_count = Counter()
    for article_link in tqdm(article_match_link, desc='Progress'):
        push_count, boo_count = get_article_push_count(article_link, push_count, boo_count)

    # get top 10 and sort
    top_10_push = sorted(push_count.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
    top_10_boo = sorted(boo_count.items(), key=lambda pair: (-pair[1], pair[0]))[:10]

    # wrtie into txt
    filename = './push[' + str(start_date) + '-' + str(end_date) + '].txt'
    push_file = open(filename, 'w')
    push_file.write('all like: {}'.format(sum(push_count.values())) + '\n')
    push_file.write('all boo: {}'.format(sum(boo_count.values())) + '\n')
    # write push and boo ranks
    rank = 1
    for userid, freq in top_10_push:
        push_file.write('like #{}: {} {}'.format(rank, userid, freq) + '\n')
        rank += 1
    rank = 1
    for userid, freq in top_10_boo:
        push_file.write('boo #{}: {} {}'.format(rank, userid, freq) + '\n')
        rank += 1
    push_file.close()


def popular(start_date, end_date):
    popular_file = open(all_popular, 'r')
    popular_info = popular_file.readlines()
    popular_match_link = []
    popular_count = 0
    accepted_types = ['jpg', 'jpeg', 'png', 'gif']

    # get link between start and end date
    for article in popular_info:
        article_date = int(article.split(',')[0])
        if article_date >= start_date and article_date <= end_date:
            article_link = article.split(',')[-1].replace('\n', '')
            popular_match_link.append(article_link)
            popular_count += 1

    # wrtie into txt
    filename = './popular[' + str(start_date) + '-' + str(end_date) + '].txt'
    popular_file = open(filename, 'w')
    popular_file.write('number of popular articles: {}'.format(popular_count) + '\n')

    for article_link in tqdm(popular_match_link, desc='Progress'):
        time.sleep(SLEEP_INTERVAL)
        content_response = requests.get(article_link, headers=headers)
        content_soup = BeautifulSoup(content_response.text, "html.parser")
        
        if content_soup.select("#main-content"):
            main_content = content_soup.select("#main-content")[0].get_text("|")
        else:
            main_content = []
        if "※ 發信站" in main_content:
            href_link = content_soup.find_all('a', href=True)
            for each_link in href_link:
                original_link = each_link.text
                for check_type in accepted_types:
                    if original_link.lower().endswith(check_type):
                        popular_file.write(original_link + '\n')

    popular_file.close()


def keyword(keyword, start_date, end_date):
    articles_file = open(all_articles, 'r')
    article_info = articles_file.readlines()
    article_match_link = []
    accepted_types = ['jpg', 'jpeg', 'png', 'gif']

    # get link between start and end date
    for article in article_info:
        article_date = int(article.split(',')[0])
        if article_date >= start_date and article_date <= end_date:
            article_link = article.split(',')[-1].replace('\n', '')
            article_match_link.append(article_link)

    # wrtie into txt
    filename = './keyword(' + keyword + ')[' + str(start_date) + '-' + str(end_date) + '].txt'
    keyword_file = open(filename, 'w')

    for article_link in tqdm(article_match_link, desc='Progress'):
        time.sleep(SLEEP_INTERVAL)
        content_response = requests.get(article_link, headers=headers)
        content_soup = BeautifulSoup(content_response.text, "html.parser")
        
        if content_soup.select("#main-content"):
            main_content = content_soup.select("#main-content")[0].get_text()
        else:
            main_content = []
        if "※ 發信站" in main_content:
            test_content = main_content.split('※ 發信站')[0]
            if '--' in test_content:
                test_content = main_content.split('※ 發信站')[0].split('--')
            test_keyword = any(keyword in every_string for every_string in test_content)
            if test_keyword:
                href_link = content_soup.find_all('a', href=True)
                for each_link in href_link:
                    original_link = each_link.text
                    for check_type in accepted_types:
                        if original_link.lower().endswith(check_type):
                            keyword_file.write(original_link + '\n')


if __name__ == '__main__':
    #get parameters
    functions = sys.argv[1]
    
    # function crawl
    if functions == 'crawl':
        # delete file if it exists
        if os.path.exists(all_articles):
            os.remove(all_articles)
        if os.path.exists(all_popular):
            os.remove(all_popular)

        # create multi-threads
        thread_list = []
        for thread_num in range(THREAD):
            thread_list.append(threading.Thread(target=crawl_PTT, args=(thread_num, )))
            thread_list[thread_num].start()
        for thread_num in range(THREAD):
            thread_list[thread_num].join()
        
        # sort by date and time of two files and rewrite in order
        df_articles = pd.read_csv(all_articles, sep=SEP, header=None, engine='python')
        df_articles.columns = ['date', 'page', 'idx', 'title', 'url']
        df_articles = df_articles.sort_values(by=['date', 'page', 'idx'], ascending=[True, True, True])
        articles_file = open(all_articles, 'w+')
        
        for content in df_articles[['date', 'title', 'url']].values:
            articles_file.write(str(content[0]) + ',' + content[1] + ',' + content[2] + '\n')
        articles_file.close()
        
        df_popular = pd.read_csv(all_popular, sep=SEP, header=None, engine='python')
        df_popular.columns = ['date', 'page', 'idx', 'title', 'url']
        df_popular = df_popular.sort_values(by=['date', 'page', 'idx'], ascending=[True, True, True])
        popular_file = open(all_popular, 'w+')
        
        for content in df_popular[['date', 'title', 'url']].values:
            popular_file.write(str(content[0]) + ',' + content[1] + ',' + content[2] + '\n')
        popular_file.close()
    # function push
    if functions == 'push':
        assert len(sys.argv) == 4
        push(int(sys.argv[2]), int(sys.argv[3]))
    # function popular
    if functions == 'popular':
        assert len(sys.argv) == 4
        popular(int(sys.argv[2]), int(sys.argv[3]))    
    # function keyword
    if functions == 'keyword':
        assert len(sys.argv) == 5
        keyword(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))  