import csv

csv_w = open('rock_label_2.csv','w', encoding = 'gb2312')
csv_r = open('rock_label_1.csv','r', encoding = 'gb2312')
writer = csv.writer(csv_w)
reader = csv.reader(csv_r)
for line in reader:
    for i in range(47):
        writer.writerows(line)
csv_r.close()
csv_w.close()