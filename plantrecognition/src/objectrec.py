#cd c:\project\AI\objectrecognition\plantrecognition
#activate my_env\Scripts\activate
#deactivate (exit)

#python src\objectrec.py

#data 
# https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html


import sys
import numpy as np
import cv2
import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
#from sklearn.ensemble import VotingRegressor
#from sklearn.svm import SVC
import warnings
import tarfile
import scipy.io
import os
import traceback
import io
import argparse
from scipy.stats import mode

#datafile = 'data/test/test.jpg'



def extractimg(datafile, labelfile):
    image = []
    label = []
    
    with tarfile.open(datafile, 'r') as tar:
        filelist = tar.getnames()
        for file in filelist:
            if file.endswith('.jpg'):
                imgex = tar.extractfile(file)
                img = Image.open(io.BytesIO(imgex.read()))
                #img = load_img(io.BytesIO(imgex.read()), datafile, imgex)
                img = img.resize((224, 224))
                imgar = img_to_array(img)
                image.append(imgar)
                    
    label_data = scipy.io.loadmat(labelfile)
    labels = label_data['labels']
    label = labels.flatten()
    return image, label
      
def savemodel(model, history, accuracy,outdir):
    directory = 'output'
    if os.path.exists(os.path.join('output', outdir)) or os.path.exists(outdir):
        print('Output directory already exists')
        exit()
    path = os.path.join(directory, outdir)
    os.makedirs(path)
    
    with open(os.path.join(path, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    csv_log = CSVLogger('traininglog.csv', separator=',', append=False)
    model.save(os.path.join(path, 'model.h5'))
    with open(os.path.join(path, 'model_history.txt'), 'w') as f:
        f.write(str(history.history))
    with open(os.path.join(path, 'model_accuracy.txt'), 'w') as f:
        f.write(str(accuracy))
        
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'accuracy.png'))
    plt.close()
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 10])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()

#input_shape = dimension of image, num_classes = number of color channels
# https://faroit.com/keras-docs/1.1.0/getting-started/sequential-model-guide/
# https://www.tensorflow.org/tutorials/images/cnn
# https://analyticsindiamag.com/topics/what-is-dense-layer-in-neural-network/
# https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
# https://keras.io/guides/sequential_model/
def createcnnmodel(input_shape):
    model = models.Sequential([
        #layers.Conv2D(a, (b, c)) a = number of filters, b = filter size
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        #layers.Dropout(0.2),
        layers.Flatten(),
        # units is the size of output from dense layer
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        #layers.Dense(64, activation='relu'),
        #layers.Dense(32, activation='relu'),
        #layers.Dropout(0.3),
        #layers.BatchNormalization(),
        layers.Dense(103, activation='softmax')
    ])
    return model
    
def cnnmodel(train_data, test_data, train_label, test_label, name):
    try:
        #model achitecture
        input_shape = train_data[0].shape
        model = createcnnmodel(input_shape)
        
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
        #https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        #https://stackoverflow.com/questions/69783897/compute-class-weight-function-issue-in-sklearn-library-when-used-in-keras-cl
        classweight = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)
        weightdict = dict(enumerate(classweight))
        
        #epochs = number of iterations ::::::::::::::::::::change
        fitted = model.fit(train_data, train_label, epochs=10, validation_data=(test_data, test_label), class_weight=weightdict)
        test_loss, test_acc = model.evaluate(test_data, test_label, verbose=2)
        """
        plt.plot(fitted.history['accuracy'], label='accuracy')
        plt.plot(fitted.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig(name)
        """
        savemodel(model, fitted, test_acc, name)
        print('Test accuracy:', test_acc)
        
        #accuary: measure of how often the model makes correct predictions. It is calculated as the number of correct predictions divided by the total number of predictions.
        #loss: measures the error between predicted and actual values
        #val_accuracy: accuracy of the model evaluated on a separate validation dataset. During training, a portion of the training data is typically set aside as a validation dataset.
        #val_loss: loss of the model evaluated on a separate validation dataset. During training, a portion of the training data is typically set aside as a validation dataset.
    except Exception as e:
        print(e)
        traceback.print_exc()
    
def extractcolor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    hist_hue /= hist_hue.sum()
    hist_saturation /= hist_saturation.sum()
    hist_value /= hist_value.sum()
    color_features = np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])
    return color_features

def extractshape(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.uint8(gray_image)
    _, gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hu_moments = cv2.HuMoments(cv2.moments(largest_contour)).flatten()
    return hu_moments

# Define function to create CNN model for color features
def create_color_cnn_model():
    model = models.Sequential([
        #layers.Dense(512, activation='relu'),
        #layers.Dense(256, activation='relu'),
        #layers.Dense(128, activation='relu'),
        #layers.Dense(108, activation='softmax')
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(18, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define function to create CNN model for shape features
def create_shape_cnn_model():
    model = models.Sequential([
        #only dense layer as no width or height
        #layers.Dense(512, activation='relu'),
        #layers.Dense(256, activation='relu'),
        #layers.Dense(128, activation='relu'),
        #layers.Dense(108, activation='softmax')
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(18, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define function to combine predictions from color and shape models
def color_shape_predictions(color_predictions, shape_predictions):
    # Initialize an empty list to store the combined predictions
    cosh_predictions = []
    
    # Iterate over each pair of color and shape predictions
    for color_pred, shape_pred in zip(color_predictions, shape_predictions):
        
        # Find the index of the class with the maximum count
        cosh_pred = np.argmax(color_pred + shape_pred)
        
        # Append the combined prediction to the list
        cosh_predictions.append(cosh_pred)
    
    return np.array(cosh_predictions)


# Define function to evaluate the ensemble model
def evaluate_model(predictions, true_labels):
    return np.mean(predictions == true_labels)

def colorshapecnn(train_data, train_label, test_data, test_label, outputfile):
    try:
        colortrain_features = [extractcolor(img) for img in train_data]
        shapetrain_features = [extractshape(img) for img in train_data]
        colortest_features = [extractcolor(img) for img in test_data]
        shapetest_features = [extractshape(img) for img in test_data]

        # Convert lists to numpy arrays
        colortrain_features = np.array(colortrain_features)
        shapetrain_features = np.array(shapetrain_features)
        colortest_features = np.array(colortest_features)
        shapetest_features = np.array(shapetest_features)
        
        # Train CNN model for color features
        color_model = create_color_cnn_model()
        #color_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        colorhistory = color_model.fit(colortrain_features, train_label, epochs=10)

        # Train CNN model for shape features
        shape_model = create_shape_cnn_model()
        shapehistry = shape_model.fit(shapetrain_features, train_label, epochs=10)

        color_predictions = color_model.predict(colortest_features)
        shape_predictions = shape_model.predict(shapetest_features)
           
        # Combine predictions from both models using ensemble techniques (e.g., averaging)
        ensemble_predictions = color_shape_predictions(color_predictions, shape_predictions)

        # Evaluate the ensemble model
        # You can use metrics like accuracy, precision, recall, etc.
        accuracy = evaluate_model(ensemble_predictions, test_label)
        
        directory = 'output'
        path = os.path.join(directory, outputfile)
        if os.path.exists(path) or os.path.exists(outputfile):
            print('Output directory already exists')
        else:
            os.makedirs(path)
            with open(os.path.join(path, 'shapemodel_summary.txt'), 'w', encoding='utf-8') as f:
                shape_model.summary(print_fn=lambda x: f.write(x + '\n'))
            with open(os.path.join(path, 'shapehistory.txt'), 'w') as f:
                f.write(str(shapehistry.history))
            with open(os.path.join(path, 'shapemodel_accuracy.txt'), 'w') as f:
                f.write(str(shape_predictions))
            
            with open(os.path.join(path, 'colormodel_summary.txt'), 'w', encoding='utf-8') as f:
                color_model.summary(print_fn=lambda x: f.write(x + '\n'))
            with open(os.path.join(path, 'colorhistory.txt'), 'w') as f:
                f.write(str(colorhistory.history))
            with open(os.path.join(path, 'colormodel_accuracy.txt'), 'w') as f:
                f.write(str(color_predictions))
                
            with open(os.path.join(path, 'ensemblemodel_accuracy.txt'), 'w') as f:
                f.write(str(accuracy))    
            
            plt.plot(shapehistry.history['accuracy'], label='shape_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.1, 1])
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(path, 'shape_accuracy.png'))
            plt.close()
            
            plt.plot(colorhistory.history['accuracy'], label='color_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.1, 1])
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(path, 'color_accuracy.png'))
            plt.close()
            
        
        print("Shape Color Model Accuracy:", accuracy)    
        return accuracy
    except Exception as e:
        print(e)
        traceback.print_exc()

def combine_predictions(color_predictions, shape_predictions, imgs):
    combine_pred = []
    for color_pred, shape_pred, img in zip(color_predictions, shape_predictions, imgs):
        combined_pred = np.argmax(np.vstack([color_pred, shape_pred, img]), axis=0)
        combine_pred.append(combined_pred)
    combined_pred = np.array(combined_pred)
    final_predictions = mode(combined_pred, axis=0)[0]
    return final_predictions

def esemblecnn(train_data, train_label, test_data, test_label, outputfile):
    try:

        normalize_train_data = train_data/ 255.0
        normalize_test_data = test_data / 255.0
        input_shape = train_data[0].shape
          
        colortrain_features = [extractcolor(img) for img in train_data]
        shapetrain_features = [extractshape(img) for img in train_data]
        colortest_features = [extractcolor(img) for img in test_data]
        shapetest_features = [extractshape(img) for img in test_data]

        # Convert lists to numpy arrays
        colortrain_features = np.array(colortrain_features)
        shapetrain_features = np.array(shapetrain_features)
        colortest_features = np.array(colortest_features)
        shapetest_features = np.array(shapetest_features)
        
        # Train CNN model for color features
        color_model = create_color_cnn_model()
        color_model.fit(colortrain_features, train_label, epochs=5)

        # Train CNN model for shape features
        shape_model = create_shape_cnn_model()
        shape_model.fit(shapetrain_features, train_label, epochs=5)

        model = createcnnmodel(input_shape)
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

        classweight = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)
        weightdict = dict(enumerate(classweight))
        
        #epochs = number of iterations
        fitted = model.fit(normalize_train_data, train_label, epochs=5, validation_data=(test_data, test_label), class_weight=weightdict)
        
        color_predictions = color_model.predict(colortest_features)
        shape_predictions = shape_model.predict(shapetest_features)
        model_predictions = model.predict(normalize_test_data)
        predictions = combine_predictions(color_predictions, shape_predictions, model_predictions)
        accuracy = evaluate_model(predictions, test_label)
        
        directory = 'output'
        path = os.path.join(directory, outputfile)
        if os.path.exists(path):
            print('Output directory already exists')
        
        else:
            os.makedirs(path)
            with open(os.path.join(path, 'esemblecnn.txt'), 'w') as f:
                f.write(str(accuracy))
            
            
        print("Ensemble Model Accuracy:", accuracy)
        
    except Exception as e:
        print(e)
        traceback.print_exc()



def main(order,tgz, labelfile, outputfile):
    try:
        imgdata, imglabel = extractimg(tgz, labelfile)
        imgdata = np.array(imgdata)
        imglabel = np.array(imglabel)
        train_data, test_data, train_label, test_label = train_test_split(imgdata, imglabel, test_size=0.2, random_state=555)
        normalize_train_data = train_data/ 255.0
        normalize_test_data = test_data / 255.0
        
        if order == 'cnn':
            cnnmodel(normalize_train_data, normalize_test_data, train_label, test_label, outputfile)
        elif order == 'colorshapecnn':
            colorshapecnn(train_data, train_label, test_data, test_label, outputfile)
        elif order == 'esemblecnn':
            esemblecnn(train_data, train_label, test_data, test_label, outputfile)
        else:
            print("Invalid order")
        
    except Exception as e:
        print(e)
        traceback.print_exc()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plant recognition')
    
    parser.add_argument('--order', type=str, help='cnn, colorshapecnn, esemblecnn', default='cnn')
    parser.add_argument('--tgz', type=str, help='image tar.gz file', default='17flowers.tgz')
    parser.add_argument('--mat', type=str, help='image mat file', default='imagelabels17.mat')
    parser.add_argument('output', type=str, help='output directory')
    
    args = parser.parse_args()
    
    tgzpath = 'data/'+args.tgz
    matpath = 'data/'+args.mat

    main(args.order, tgzpath, matpath, args.output)