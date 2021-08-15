import csv,codecs
# CSV(Comma-Seperated Values)

#csv = """\
#1,김정수,2017-01-19 11:30:00,25
#2,박민구,2017-02-07 10:22:00,35
#3,정순미,2017-03-22 09:10:00,33\
#"""

#splitted = csv.split("\n")
#for item in splitted:
#    list_data = item.split(",")
filename = "test.csv"
file = codecs.open(filename,"r","euc-kr")

reader = csv.reader(file,delimiter=",")

for cells in reader:
    print(cells[1],cells[2])