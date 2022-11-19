import numpy as np

file = open("gorani.jh", "r")
content = file.read()
filedata = list(content.split("\n"))

size = list(map(int, filedata[0].split(",")))
filedata.pop(0)

print(filedata)

file.close()
