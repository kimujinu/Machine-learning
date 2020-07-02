# 기존 스타일
result = []
for i in range(10):
    result.append(i)

print(result)

# List_comprehensions

result = [i for i in range(10)]

print(result)

#ex..

result = [i for i in range(10) if i%2==0] # filter 기법 적용 

print(result)
