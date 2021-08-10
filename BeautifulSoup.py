# HTML, XML을 분석(파싱)해주는 라이브러리
from bs4 import BeautifulSoup

# 태그선택자
"h1"
"html"
# id 선택자
"#<아이디 이름>"
# 클래스 선택자
".<클래스 이름>.<클래스 이름>"
# 후손 선택자
"#html li"
# 자식 선택자
"ul.items > li"

html = """
<html><body>
<div id = "meigen">
    <h1>위키북스 도서</h1>
    <ul class = "items">
        <li>유니티 게임 이펙트 입문</li>
        <li>스위프트로 시작하는 아이폰 앱 개발 교과서</li>
        <li>모던 웹사이트 디자인의 정석<li>
    </ul>
</div>
</body></html>
"""

soup = BeautifulSoup(html,'html.parser')
# css선택자를 알아야함
header = soup.select_one("body > div > h1") # 요소
list_items = soup.select("ul.items > li") # 요소의 배열

header.string
#header.attrs["title"]
print(soup.select_one("ul").attrs)
for i in list_items :
    print(i.string)