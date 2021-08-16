from sklearn import model_selection,svm,metrics
import pandas

train_csv = pandas.read_csv(" ", header=None)
tk_csv = pandas.read_csv(" ",header=None)

train_csv_data = list(map(train_csv.iloc[:, 1:].values))
tk_csv_data = list(map(tk_csv.iloc[:,1:].values))
print(tk_csv_data)

train_csv_label = train_csv[0].values
tk_csv_label = tk_csv[0].values

clf = svm.SVC()
clf.fit(train_csv_data,train_csv_label)
predict = clf.predict(tk_csv_data)
score = metrics.accuracy_score(tk_csv_label,predict)
print("정답률 : ",score)