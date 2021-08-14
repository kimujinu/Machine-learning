# docker pull ubuntu:16.04
# docker run -it ubuntu:16.04
# apt-get update
# apt-get install -y python3 python3-pip
# pip3 install selenium
# pip3 install beautifulsoup4
# apt-get install -y wget libfontconfig
# mkdir -p /home/root/src && cd $_
# wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2
# tar jxvf phantomjs-2.1.1-linux-x86_64.tar.bz2
# cd phantomjs-2.1.1-linux-x86_64/bin
# cp phantomjs /usr/local/bin/
# apt-get install -y fonts-nanum*
# docker ps -a
# docker commit <아이디> ubuntu-phantomjs
# docker run -i -t -v /c/Users/sdfdsf/sample:/sample -e ko_KR.UTF-8 -e PYTHONIOENCODING=utf_8 ubuntu-phantomjs /bin/bash
from selenium import webdriver

url = "https://smarket365.com/customer/login"

# PhantomJS 드라이버 추출하기
browser = webdriver.PhantomJS()
# 3초 대기
browser.implicitly_wait(5)
# URL 읽어 들이기
browser.get(url)
element_id = browser.find_element_by_id("loginId") # 아이디 텍스트 입력 상자
element_id.clear()
element_id.send_keys("test")
element_pw = browser.find_element_by_id("loginPw") # 비밀번호 텍스트 입력 상자
element_pw.clear()
element_pw.send_keys("1111")

buttton = browser.find_element_by_css_selector("button.btn-login[type=button]")
buttton.click()
# 화면을 캡처해서 저장
browser.save_screenshot("Website4.png")
# 브라우저 종료
browser.quit()