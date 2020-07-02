# 두 개의 list의 값을 병렬적으로 추출함

alist=['a1','a2','a3']
blist=['b1','b2','b3']

for a,b in zip(alist,blist): # 병렬적으로 값을 추출
    print(a,b)

result = [sum(x) for x in zip((1,2,3),(10,20,30),(100,200,300))] # 각 튜플 같은 index를 묶어 합을 list로 변환
print(result)

