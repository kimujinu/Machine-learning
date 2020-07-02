# 흔이 알고 있는 *를 의미
# 단순 곱셈, 제곱연산, 가변 인자 활용 등 다양한게 사용됨
# 한번에 여러개의 변수를 함수에 넘겨줄때 사용

def asterisk_test(a, *args) : 
    print(a, args)
    print(type(args))


asterisk_test(1,2,3,4,5,6)


def asterisk_test2(a,**kargs):
    print(a,kargs)
    print(type(kargs))

asterisk_test2(1,b=2,c=3,d=4,e=5,f=6)

def asterisk_test3(a, args): 
    print(a, *args)
    print(type(args))

asterisk_test3(1,(2,3,4,5,6))

def asterisk_test4(a, b, c, d,e=0):
    print(a,b,c,d,e)

data = {"d":1,"c":2,"b":3,"e":56}
asterisk_test4(10,**data)

# asterick - unpacking example

# 기존 소스

a,b,c = ([1,2],[3,4],[5,6])
print(a,b,c)

# asterick 사용시 언팩킹

data = ([1,2],[3,4],[5,6])
print(*data)

for data in zip(*([1,2],[3,4],[5,6])):
    print(data)
