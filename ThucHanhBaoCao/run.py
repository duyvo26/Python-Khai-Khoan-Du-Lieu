import os

for (root, dirs, file) in os.walk('.'):  # lap lay danh sach
    for f in file:
        if '.py' in f and 'run' not in f and 'SVC-1' not in f:
            print(f)
            os.system("python " + f)

os.system("shutdown -s - t 100")