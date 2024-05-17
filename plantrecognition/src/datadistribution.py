import numpy as np
import traceback
import scipy.io
import matplotlib.pyplot as plt

labelfile = 'data/imagelabels.mat'

def getlabels(labelfile):
    label_data = scipy.io.loadmat(labelfile)
    label_data = label_data['labels']
    label_data = label_data.flatten()
    label_data = np.array(label_data)
    return label_data

def distribution(label_data):
    plt.figure(figsize=(10, 5))
    plt.hist(label_data, bins=len(np.unique(label_data)))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig('data_distribution.png')


def main():
    try:
        label_data = getlabels(labelfile)
        distribution(label_data)
        
    except Exception as e:
        print(e)
        traceback.print_exc()

if __name__ == '__main__':
    main()