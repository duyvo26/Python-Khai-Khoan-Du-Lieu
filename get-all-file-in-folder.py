from os import walk
import os

mypath = "DataImg/" # path thu muc

listIMG = [] # mang chua path IMG

for (root, dirs, file) in os.walk(mypath): # lap lay danh sach
    for f in file:
        FileIMG = mypath+f
        listIMG.append(FileIMG)


print(listIMG)