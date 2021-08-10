# 1단계 : 필요한 것이 어디 있는지 확인하기
# 2단계 : 뭘로 가져올 수 있는지 확인하기
# 3단계 : 가져올수 있는지 테스트
import urllib.request
from bs4 import BeautifulSoup
import time

# 모듈 추출
# 기사목록 추출
#url = "https://map.naver.com/v5/search/%EB%B2%A0%EC%96%B4%EB%A7%81?c=14342917.2004221,4196015.2629655,13,0,0,0,dh"
#url = "https://finance.naver.com/marketindex/"
url = "http://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=105"
response = urllib.request.urlopen(url)

soup = BeautifulSoup(response,"html.parser")
#soup.select_one()
results = soup.select("#section_body a")
for result in results:
    print(result.attrs["title"])
    url_artice = result.attrs["href"]
    response = urllib.request.urlopen(url_artice)
    soup_article = BeautifulSoup(response,"html.parser")
    content = soup_article.select("#articleBodyContents")
    print(content.contents)
    time.sleep(30)

