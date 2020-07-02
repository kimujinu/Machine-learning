#일반코드

colors = ["a","b","c","d","e"]
result = ""

for s in colors :
    result += s

print("일반코드 : "+result)


#pythonic code

colors = ["red","blue","green","yellow"]
result = "".join(colors)

print("pythonic code : "+result)
