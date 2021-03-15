import numpy as np

def getColorMap(file):
    colorMap = np.empty((256, 3), dtype=np.int)
    f = open(file, 'r')

    for i in range(256):
        line = f.readline()
        colorMap[i, :] = np.fromiter(map(int, line.split(' ')[:-1]), dtype=np.int)

    f.close()
    return colorMap