from os import walk
import os

mypath = "DataImg/" # path thu muc

listIMG = [] # mang chua path IMG

SumFile = 0 #sum file

for (root, dirs, file) in os.walk(mypath): # lap lay danh sach
    for f in file:
        SumFile += 1
        FileIMG = root+"/"+f
        listIMG.append(FileIMG)


print(SumFile)
print(listIMG)