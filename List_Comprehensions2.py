word_1 = "Hello"
word_2 = "world"

result = [i+j for i in word_1 for j in word_2] # for i in word_1:
                                                    #for j in word+2:  이중 반복문과 같음
print(result)


result2 = [i+j for i in word_1 for j in word_2 if not (i!=j)]

print(result2)
