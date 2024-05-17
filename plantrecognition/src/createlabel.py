import os
import sys
import tarfile
import scipy.io
import numpy as np


def getlength(tgzfile):
    with tarfile.open(tgzfile, 'r') as tar:
        filelist = tar.getnames()
        count = 0
        for file in filelist:
            if file.endswith('.jpg'):
                count += 1
        return count

def create_label(length, num):
    labels = []
    category = 1
    for i in range(1, length+1):
        if i % num == 0:
            labels.append(category)
            category += 1
        else:
            labels.append(category)
    return labels

def savemat(label, filename):
    path = os.path.join('data/', filename)
    labeldict = {'labels': label}
    scipy.io.savemat(path, labeldict)

def main():
    datafile = sys.argv[1]
    labelfile = sys.argv[2]
    iteration = int(sys.argv[3])
    tgzpath = 'data/'+datafile
    length = getlength(tgzpath)
    label = create_label(length, iteration)
    savemat(label, labelfile)  

if __name__ == '__main__':
    main()      
        
    