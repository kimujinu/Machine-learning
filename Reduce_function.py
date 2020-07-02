# map function과 달리 list에 똑같은 함수를 적용해서 통합

from functools import reduce

print(reduce(lambda x, y:x+y,[1,2,3,4,5]))

def factorial(n):
    return reduce(
            lambda x,y:x*y,range(1,n+1))

print(factorial(5))

# Lambda, map, reduce는 간단한 코드로 다양한 기능을 제공
# 그러나 코드의 직관성이 떨어져서 lambda나 reduce는 파이썬3에서 사용권장 x
# Legacy library나 다양한 머신러닝 코드에서 여전히 사용중


