# Installation
-python
-cv2
-tensorflow
-numpy
-matplotlib
-keras
-sklearn

# usage
## objectrec.py
run by following the command
python objectrec.py --order a --data b --label c --output d
a presents the method to create a model or models with a default value of cnn 
the choices are cnn, colorshapecnn, esemblecnn
b is the data file name inside the data folder
c is the label file name inside the data folder
d is the output folder name which will be created in the output folder

this will create the model based on your choice and provide the accuracy 

## datadistribution.py
run by python datadistribution.py
by providing the label data file name to labelfile, it will create the graph to see distribution of data

## createlabel.py
run by python createlabel.py a b c
a is the data file in data folder
b is the name of the label file you want to create
c is the iteration to assign the label
use the file when there is data without a category
data needs to have same number of each group and list in order

# Data
Data is from the Oxford Flowers dataset, get the data from the link below
https://www.robots.ox.ac.uk/~vgg/data/flowers/
Labels for 102 flowers data are attached
Labels for 17 flowers are created by createlabel.py

