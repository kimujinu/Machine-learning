#도커 명령어 :
# docker pull continuumio/miniconda3
# docker run -i -t continuumio/miniconda3 /bin/bash
# pip install beautifulsoup4
# pip install requests
# docker ps -a
# docker commit 2623e05a7966
# docker run -i -t -v /c/Users/sdfdsf/sample:/sample mlearn:init /bin/bash
# python download_png1.py,
#웹 스크레이핑
import urllib.request

url = "http://api.aoikujira.com/ip/ini"
savename = "test.png"

# 다운로드
mem = urllib.request.urlopen(url).read()
print(mem.decode("utf-8"))
#파일로 저장
#with open(savename,mode="wb") as f:
 #   f.write(mem)
  #  print("저장되었습니다!")

#urllib.request.urlretrieve(url,savename)