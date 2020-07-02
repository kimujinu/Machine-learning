# String Type의 값을 나눠서 List 형태로 변환

items = 'zero one two three'.split() # 빈칸을 기준으로 문자열 나누기
print(items)
example='python,jquery,javascript'
example.split(",")
a,b,c = example.split(",") # 리스트에 있는 각 값을 a,b,c 변수로 unpacking
example='cs50.gachon.edu'
subdomain, domain, tld = example.split(".")
