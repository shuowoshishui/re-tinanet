with open('test.txt') as reader, open('val.txt', 'w') as writer:
    for index, line in enumerate(reader):
        if index % 20 ==0:
            writer.write(line)
           # writer.write('\n')
list1=[]
for line1 in open("val.txt"):
    list1.append(line1)
print(list1)
with open('test.txt') as reader, open('train.txt', 'w') as writer:
    for index, line in enumerate(reader):
        if line not in list1:
            writer.write(line)
            #writer.write('\n')