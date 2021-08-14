import requests
from bs4 import BeautifulSoup

# 세션 만들기
session = requests.session()
# 로그인
url = "http://www.hanbit.co.kr/member/login_proc.php"
data = {
    "return_url":"http://www.hanbit.co.kr/index.html",
    "m_id":"<아이디>",
    "m_passwd":"<비밀번호>"
}
response = session.get(url,data=data)
response.raise_for_status()
# 이코인 들고와보기
url = "http://www.hanbit.co.kr/myhanbit/myhanbit.html"
response = session.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.text,"html.parser")
text = soup.select_one(".mileage_section2 span")
print("마일리지: ",text)


#print(response.text)
