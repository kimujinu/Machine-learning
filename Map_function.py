# Sequence 자료형 각 element에 동일한 function을 적용함

ex = [1,2,3,4,5]

f = lambda x: x**2

print(list(map(f,ex)))

f1 = lambda x,y:x+y

print(list(map(f1,ex,ex)))

list(map(lambda x: x**2 if x%2==0 else x,ex))
