import numpy as np

def getColorMap(file):
    colorMap = np.empty((256, 3), dtype=np.int)
    f = open(file, 'r')

    for i in range(256):
        line = f.readline()
        colorMap[i, :] = np.fromiter(map(int, line.split(' ')[:-1]), dtype=np.int)

    f.close()
    return colorMap


def segmentationColorize(img, colorMap):
    outputColorMap = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            outputColorMap[i, j, 0] = colorMap[img[i, j], 0]
            outputColorMap[i, j, 1] = colorMap[img[i, j], 1]
            outputColorMap[i, j, 2] = colorMap[img[i, j], 2]

    return outputColorMap
