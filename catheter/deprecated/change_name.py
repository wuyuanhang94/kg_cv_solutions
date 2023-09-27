import os

for name in os.listdir('../checkpoint'):
    if 'b7' not in name:
        continue
    os.rename(name, name)