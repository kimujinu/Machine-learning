#https://search.naver.com/search.naver
#?where=nexearch&
#sm=top_hty&
#fbm=1&
#ie=utf8&
#query=%ED%8C%8C%EC%9D%B4%EC%8D%AC

#방식 : GET, POST, PUT, DELETE
#대상 : https://search.naver.com -> 호스트이름
# 추가적인 정보
# - 경로 : /search.naver
# - 데이터 : ?where=nexearch&
# sm=top_hty&
# fbm=1&
# ie=utf8&
# query=%ED%8C%8C%EC%9D%B4%EC%8D%AC

import urllib.request
import urllib.parse

api = "https://search.naver.com/search.naver"

values = {
    "where" : "nexearch",
    "sm" : "top_hty",
    "fbm" : "1",
    "ie" : "utf8",
    "query":"파이썬"
}

params = urllib.parse.urlparse(values)
url = api +"?"+ params

data = urllib.request.urlopen(url).read()
text = data.decode("utf-8")
print(text)