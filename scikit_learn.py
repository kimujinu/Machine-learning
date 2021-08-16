# pip3 install -U scikit-learn scipy matplotlib scikit-image
# pip install pandas
# docker commit <아이디> mlearn:init
# docker run -i -t -v /c/user/sdfdsf/sample:/sample mlearn:init /bin/bash
from sklearn import svm, metrics

clf = svm.SVC()
# clf.fit(데이터,답)
#clf.fit([
#    [0,0],
#    [1,0],
#    [0,1],
#    [1,1]
#], [0,1,1,0])

#results = clf.predict([
#    [0,0],
#    [1,0]
#])

#print(results)

datas = [
    [0,0],
    [1,0],
    [0,1],
    [1,1]
]
example = [[0,0],[1,0]]
example_label = [0,1]

labels = [0,1,1,0]
clf.fit(datas,labels)
results = clf.predict(example)
print(results)

score = metrics.accuracy_score(example_label,results)
print("정답률: ",score)