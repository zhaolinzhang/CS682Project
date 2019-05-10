'''
Spencer Riggins, Yajie Yang, Yiting Liu, Zhaolin Zhang
GMU CS 682
Computer Vision Final Project
Professor Duric
Spring 2019
'''

import numpy as np
import os
import pandas as pd
import cv2
import copy
import operator
import random

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, RNN, SimpleRNNCell, Embedding, Reshape
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from keras.models import Sequential
from keras import backend as K
from sklearn.datasets.base import Bunch

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from sklearn.datasets.base import Bunch



#Returns percentage of labeled samples(learned_data) are labeled correctly in data object
def test_against_true_vals(true_data, learned_data):
    num_learned_samples = learned_data['data'].shape[0]
    num_correctly_labeled = 0
    for i in range(learned_data['data'].shape[0]):
        true_index = -1
        for j in range(true_data['data'].shape[0]):
            if(np.all(learned_data['data'][i] == true_data['data'][j])):
                true_index = j
            if(true_index != -1):
                break
        if(i % 1000 == 0):
            print("Validated " + str(i) + " out of " + str(num_learned_samples))
        if(np.all(learned_data['labels'][i] == true_data['labels'][j])):
            num_correctly_labeled = num_correctly_labeled + 1
    return num_correctly_labeled, num_learned_samples






#Paths to preprocessed images
pwd = os.path.dirname(os.path.realpath(__file__))
germany_train_csv_path = os.path.join(pwd, "filepath_class_mapping", "germany_training_processed.csv")
germany_test_csv_path = os.path.join(pwd, "filepath_class_mapping", "germany_test_processed.csv")
italy_train_csv_path = os.path.join(pwd, "filepath_class_mapping", "italy_training_processed.csv")
italy_test_csv_path = os.path.join(pwd, "filepath_class_mapping", "italy_test_processed.csv")
belgium_train_csv_path = os.path.join(pwd, "filepath_class_mapping", "belgium_training_processed.csv")
belgium_test_csv_path = os.path.join(pwd, "filepath_class_mapping", "belgium_training_processed.csv")


#Data in files contains many classes. This maps those classes down to a set of three classes used for our project.
#Traffic Controls and Traffic Priority -> 0
#Speed Limit and Environmental Factors -> 1
#Vehicle Information and Pedestrian Signs -> 2
#Throwaway samples are not used because they do not correspond to any sign in the German data set
germany_class_mapping = {"0": [14, 17, 26, 32, 33, 34, 35, 36, 37, 38, 39, 40, 18, 11, 12, 13, 24], "1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 30, 31], "2": [9, 10, 41, 42, 15, 16, 25, 27, 28, 29]}
italy_class_mapping = {"0":[45, 25, 46, 28, 10, 11, 35, 19, 9, 33, 21, 3], "1":[39, 40, 41, 42, 43, 44, 57, 58, 7, 4, 0], "2":[26, 50, 5, 48, 49, 36, 55], "throwaway":[1, 2, 6, 8, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 27, 29, 30, 31, 32, 34, 37, 38, 47, 51, 52, 53, 54, 56]}


#Object that holds all of the data used for training and testing models
class data_object:

    def __init__(self, source, percent_unlabeled): #percent_unlabeled is the percent of data to leave unlabeled in training data
        train_source = source + "_Train"
        test_source = source + "_Test"
        #Pass source of labeled data and percent of data to label (10%)
        train_data_imgs, train_data_labels = self.get_data(train_source)
        self.train_data = {"data": train_data_imgs, "labels": train_data_labels}
        train_data_copy = copy.deepcopy(self.train_data)
        self.labeled_data = {"data": np.empty([0, train_data_imgs.shape[1], train_data_imgs.shape[2]]), "labels": np.empty([0, train_data_labels.shape[1]])} #Labels are held for reference but should not be used.
        print("Performing Random Sampling")
        self.uniform_random_sampling(self.train_data, self.labeled_data, 1 - percent_unlabeled) #Populate unlabeled_data
        self.unlabeled_data = {"data": self.train_data['data'], "labels": self.train_data['labels']} #Populate labeled_data
        test_data_imgs, test_data_labels = self.get_data(test_source)#<percent_test> percent of self.original_data with labels (held for testing purposes) 10%
        self.test_data = {"data": test_data_imgs, "labels": test_data_labels}
        self.train_data = train_data_copy



    #Reads data into data object from csv_file_path and applies labels
    def read_data(self, csv_file_path, class_mappings):
        annotations = pd.read_csv(csv_file_path)
        rows, cols = annotations.shape


        image_path_collection = []
        image_class_collection = []

        for i in range(rows):
            image_path = annotations.at[i, 'file']
            split_image_path = []
            while True:
                head, tail = os.path.split(image_path)
                image_path = head
                if(tail != "."):
                    split_image_path.insert(0, tail)
                if(tail == "."):
                    break
            image_path = pwd
            for path_component in split_image_path:
                image_path = os.path.join(image_path, path_component)
            image_class = annotations.at[i, 'class']
            class_array = np.zeros(len(class_mappings.keys()))
            if("throwaway" in class_mappings.keys()):
                class_array = np.zeros(len(class_mappings.keys())-1)
            mapped_class = -1
            throwaway = False
            for class_key in class_mappings.keys():
                if(image_class in class_mappings[class_key]):
                    if(class_key == "throwaway"):
                        throwaway = True
                        break
                    mapped_class = int(float(class_key))
            if(throwaway):
                pass
            class_array[int(mapped_class)] = 1
            if(mapped_class != -1):
                image_path_collection.append(image_path)
                image_class_collection.append(class_array)


        image_collection = []
        for i in range(len(image_path_collection)):
            image = cv2.imread(image_path_collection[i])
            image_array = np.array(image[:, :, 0])
            image_collection.append(image_array)

        if len(image_collection) == len(image_class_collection):
            return np.array(image_collection), np.array(image_class_collection)
        else:
            raise ImportError


    #Determines which data set to read into data object
    def get_data(self, source):
        #Fetch and return preprocessed data (with labels) for training/co-training
        if(source == "Germany_Train"):
            data = self.read_data(germany_train_csv_path, germany_class_mapping)
        elif(source == "Germany_Test"):
            data = self.read_data(germany_test_csv_path, germany_class_mapping)
        elif(source == "Italy_Train"):
            data = self.read_data(italy_train_csv_path, italy_class_mapping)
        elif(source == "Italy_Test"):
            data = self.read_data(italy_test_csv_path, italy_class_mapping)
        else:
            pass
        return data

    #Randomly sample percent of data from source data and put it in destination data (data object generation)
    def uniform_random_sampling(self, source_data, destination_data, percent_of_data):
        def perform_update(indices, source_data, destination_data):
            count = 0
            for index_list in indices:
                index = index_list[0]
                label = index_list[1]
                image = np.expand_dims(source_data['data'][index], axis=0)
                destination_data['data'] = np.append(destination_data['data'], image, axis = 0)
                temp_label_arr = np.zeros(destination_data['labels'].shape[1])
                temp_label_arr[label] = 1
                temp_label_arr = np.expand_dims(temp_label_arr, axis = 0)
                destination_data['labels'] = np.append(destination_data['labels'], temp_label_arr, axis = 0)
                count = count + 1
            indices = sorted(indices, key=operator.itemgetter(0), reverse=True) #Sort by largest index to lowest so data doesn't get messed up when deleting
            for index_list in indices:
                index = index_list[0]
                label = index_list[1]
                source_data['data'] = np.delete(source_data['data'], index, 0)
                source_data['labels'] = np.delete(source_data['labels'], index, axis = 0)


        num_source_samples = source_data['data'].shape[0]
        num_to_sample = num_source_samples * percent_of_data
        index_and_labels = []
        index_list = []
        while(len(index_list) < num_to_sample):
            possible_index = random.randint(0, num_source_samples - 1)
            if(possible_index not in index_list):
                index_list.append(possible_index)
        for index in index_list:
            label_index = np.argwhere(source_data['labels'][index] == 1)[0][0]
            index_and_labels.append([index, label_index])
        perform_update(index_and_labels, source_data, destination_data)


    #Take in predictions from one classifier and move the <num_to_update> most confident samples into
    #The labeled data array with their predicted label appended
    def update_data_single_classifier(self, predicted_label_arr, num_to_update):

        #Pass in array of predicted labels and remove <num_to_update> most confident predictions from <self.unlabeled_data>, placing those into <self.labeled_data> with predicted labels
        def generate_indices(predicted_label_array, number_to_update):
            if(number_to_update > predicted_label_array.shape[0]):
                number_to_update = predicted_label_array.shape[0]
            index_prob_pred = []
            for i in range(predicted_label_array.shape[0]):
                cur_pred = np.argmax(predicted_label_array[i])
                cur_max = predicted_label_array[i][cur_pred]
                index_prob_pred.append([i, cur_max, cur_pred])
            index_prob_pred = sorted(index_prob_pred, key=operator.itemgetter(1), reverse=True)
            update_arr = []
            for k in range(number_to_update):
                update_arr.append([index_prob_pred[k][0], index_prob_pred[k][2]])
            return update_arr


        #Helper method that places the source_data samples at indices in the indices list into the destination data array
        def perform_update(indices, source_data, destination_data):
            for index_list in indices:
                index = index_list[0]
                label = index_list[1]
                image = np.expand_dims(source_data['data'][index], axis=0)
                destination_data['data'] = np.append(destination_data['data'], image, axis = 0)
                temp_label_arr = np.zeros(destination_data['labels'].shape[1])
                temp_label_arr[label] = 1
                temp_label_arr = np.expand_dims(temp_label_arr, axis = 0)
                destination_data['labels'] = np.append(destination_data['labels'], temp_label_arr, axis = 0)
            indices = sorted(indices, key=operator.itemgetter(0), reverse=True) #Sort by largest index to lowest so data doesn't get messed up when deleting
            for index_list in indices:
                index = index_list[0]
                label = index_list[1]
                source_data['data'] = np.delete(source_data['data'], index, 0)
                source_data['labels'] = np.delete(source_data['labels'], index, axis = 0)

        print("GENERATING INDICES")
        items_to_update = generate_indices(predicted_label_arr, num_to_update)
        #perform_update(items_to_update, source_data, destination_data)
        print("PERFORMING UPDATES")
        perform_update(items_to_update, self.unlabeled_data, self.labeled_data)


    #Take in predictions from two classifiers and move the <num_to_update> most confident samples from
    #each classifier from unlabeled_data to labeled_data
    def update_data(self, predicted_label_arrA, predicted_label_arrB, num_to_update):
        #Do the same thing as update_data_single_classifier except with two arrays of predictions from cotraining

        #Helper method: Finds all the object values in a dictionary
        def union_of_dict(dictA):
            union = []
            for key in dictA.keys():
                union = union + dictA[key]
            return union

        #Helper method: Finds if given value occurs in a dictionary
        # def index_in_dict(dictionary, keys):
        #     for key in dictionary.keys():
        #         if index in dictionary[keys]:
        #             return True
        #     return False

        #Count the number of elements summed over all array value lists
        def sum_over_dictionary(dictionary):
            total = 0
            for key in dictionary.keys():
                total = total + len(dictionary[key])
            return total

        #Pass in two dictionaries whose values are lists. Returns list of values which are contained within lists in both dictionaries.
        def dictionary_collisions(dictA, dictB):
            collisions = []
            for keyA in dictA.keys():
                for value in dictA[keyA]:
                    for keyB in dictB.keys():
                        if((value in dictB[keyB]) and (value not in collisions)):
                            collisions.append(value)
            return collisions

            #Returns true if all indices from [0, num_classes-1] are included somehwere in either dictA or dictB
        def all_indices_in_dicts(dictA, dictB, num_classes):
            assert(dictA.keys() == dictB.keys())
            dictA_union = union_of_dict(dictA)
            dictB_union = union_of_dict(dictB)
            for i in range(num_classes):
                if((i not in dictA_union) and (i not in dictB_union)):
                    return False
            return True

        #Checks if classifier A is more confident in prediction at index
        def check_for_sureness(prob_arrA, prob_arrB, predictionA, predictionB, index):
            if(prob_arrA[index][predictionA] > prob_arrB[index][predictionB]):
                return True
            elif(prob_arrA[index][predictionA] < prob_arrB[index][predictionB]):
                return False
            else: #If a tie pick randomly who gets it
                if(random.uniform(0,1)):
                    return True
                else:
                    return False

        #Returns key as int where index is in value list
        def find_prediction(dictA, index):
            for key in dictA.keys():
                if(index in dictA[key]):
                    return int(key)
            return -1

        #Picks the top <max_size> most confident predictions that are not contained in the prohibited_indices list
        def add_indices_to_dict(dictA, prob_arrA, max_size, prohibited_indices):
            #Empty all arrays in dictA to be refilled
            for key in dictA.keys():
                dictA[key] = []
            #max_locs will look like max_locs = [[x1,y1,z1], [x2,y2,z2], ...] where x is the index of the sample, y is the subindex (class) of the max value in sample, z is the value of max value in sample
            max_locs = []
            sub_arr = []
            for i in range(prob_arrA.shape[0]):
                sub_arr.append(i)
                max_index = np.argmax(prob_arrA[i])
                sub_arr.append(max_index)
                max_value = prob_arrA[i][max_index]
                sub_arr.append(max_value)
                max_locs.append(sub_arr)
                sub_arr = []
            max_locs = sorted(max_locs, key=operator.itemgetter(2), reverse=True) #max_locs is now sorted from highest to lowest in terms of max_value
            all_labeled = (union_of_dict(dictA) + prohibited_indices).sort() == list(range(prob_arrA.shape[0])) #If we have assigned a label to every index in the array of probabilities
            max_locs_cur_index = 0
            while((sum_over_dictionary(dictA) != max_size) and (not all_labeled)):
                if(max_locs[max_locs_cur_index][0] not in prohibited_indices):
                    dictA[str(max_locs[max_locs_cur_index][1])].append(max_locs[max_locs_cur_index][0])
                all_labeled = set((union_of_dict(dictA) + prohibited_indices)) == set(range(prob_arrA.shape[0]))
                max_locs_cur_index = max_locs_cur_index + 1
            return dictA

        #Returns list of indices and labels that can be consumed by perform update
        def get_final_indices(dictA, dictB):
            final_indices = []
            for keyA in dictA.keys():
                for value in dictA[keyA]:
                    final_indices.append([value, int(keyA)])
            for keyB in dictB.keys():
                for value in dictB[keyB]:
                    final_indices.append([value, int(keyB)])
            return final_indices #final_indices ends up looking like [[x1,y1], [x2,y2],  ...] where x is the index of the selection and y is the value assigned to it

        #Moves data at indices from source data to destination data with their labels
        def perform_update(indices, source_data, destination_data):
            for index_list in indices:
                index = index_list[0]
                label = index_list[1]
                image = np.expand_dims(source_data['data'][index], axis=0)
                destination_data['data'] = np.append(destination_data['data'], image, axis = 0)
                temp_label_arr = np.zeros(destination_data['labels'].shape[1])
                temp_label_arr[label] = 1
                temp_label_arr = np.expand_dims(temp_label_arr, axis = 0)
                destination_data['labels'] = np.append(destination_data['labels'], temp_label_arr, axis = 0)
            indices = sorted(indices, key=operator.itemgetter(0), reverse=True) #Sort by largest index to lowest so data doesn't get messed up when deleting
            for index_list in indices:
                index = index_list[0]
                label = index_list[1]
                source_data['data'] = np.delete(source_data['data'], index, 0)
                source_data['labels'] = np.delete(source_data['labels'], index, axis = 0)



        #Input: Arrays of probabilities and number of classifications each classifier is allowed to make
        #Returns each classifiers top <per_classifier> most confident predictions that do not share the same index
        #When classifiers disagree on a sample's label, the tie is broken by which classifier is most confident
        def find_indices(prob_arrA, prob_arrB, per_classifier):
            assert(prob_arrA.shape == prob_arrB.shape)
            assert(len(prob_arrA.shape) == 2)
            num_samples = prob_arrA.shape[0]
            num_classes = prob_arrA.shape[1]
            indices_a = {} #Dicts where keys are in range str(num_classes) and values are indices to be assigned label from the key
            indices_b = {}
            for label_num in range(num_classes):
                indices_a[str(label_num)] = []
                indices_b[str(label_num)] = []
            #Initialize all the checks for the while loop
            a_empty = sum_over_dictionary(indices_a) == 0
            b_empty = sum_over_dictionary(indices_b) == 0
            a_full = sum_over_dictionary(indices_a) == per_classifier
            b_full = sum_over_dictionary(indices_b) == per_classifier
            both_full = a_full and b_full
            collisions = dictionary_collisions(indices_a, indices_b)
            all_indices_counted = all_indices_in_dicts(indices_a, indices_b, num_samples)
            #Remember that an empty list from collections evaluates to False
            #Starts with both index dicts empty
            #Ends when no index collisions and either both classifiers have classified per_classifier number of indices or all indices have been classified
            while((a_empty and b_empty) or collisions or (not (both_full or all_indices_counted))):#(not (both_full or collisions) or not (all_indices_counted or collisions))):
                #Rectify any instances where both classifiers want to supply a prediction for one instance
                for collision in collisions:
                    a_pred = find_prediction(indices_a, collision)
                    b_pred = find_prediction(indices_b, collision)
                    a_more_confident = check_for_sureness(prob_arrA, prob_arrB, a_pred, b_pred, collision)
                    if(a_more_confident): #If a is more confident about the prediction for the colliding index
                        indices_b[str(b_pred)].remove(collision)
                    else:
                        indices_a[str(a_pred)].remove(collision)
                #Recalculate indices for dictionaries
                indices_a = add_indices_to_dict(indices_a, prob_arrA, per_classifier, union_of_dict(indices_b))
                indices_b = add_indices_to_dict(indices_b, prob_arrB, per_classifier, union_of_dict(indices_a))
                #Update all checks for the while loop
                a_empty = sum_over_dictionary(indices_a) == 0
                b_empty = sum_over_dictionary(indices_b) == 0
                a_full = sum_over_dictionary(indices_a) == per_classifier
                b_full = sum_over_dictionary(indices_b) == per_classifier
                both_full = a_full and b_full
                collisions = dictionary_collisions(indices_a, indices_b)
                all_indices_counted = all_indices_in_dicts(indices_a, indices_b, num_samples)
            return get_final_indices(indices_a, indices_b)

        #Find indexes and labels to update
        items_to_update =  find_indices(predicted_label_arrA, predicted_label_arrB, num_to_update)
        #Update those samples from unlabeled data to labeled date
        perform_update(items_to_update, self.unlabeled_data, self.labeled_data)


    #Find accuracy of predictions
    #Class_array is true label array, prediction_array is predicted class array
    def test_predictions(self, class_array, prediction_array):
        def getAccuracy(classes, predictions):
            assert(classes.shape == predictions.shape)
            num_correct = 0
            for i in range(classes.shape[0]):
                true_class = np.argwhere(classes[i] == 1)
                pred_class = np.argmax(predictions[i])
                if(true_class == pred_class):
                    num_correct = num_correct + 1
            return num_correct/classes.shape[0]
        return getAccuracy(class_array, prediction_array)

    #Checks whether all of the data in the object is labeled or not
    def all_data_labeled(self):
        assert(self.unlabeled_data['data'].shape[0] == self.unlabeled_data['labels'].shape[0])
        if(self.unlabeled_data['data'].shape[0] == 0):
            return True
        return False

#Extend this class and implement train_model and get_predictions for each type of classifier we will use
#This is the template for classes that will hold a classifier
class model_object(object):
    '''
    Do Any Initialization Needed to Set up Model For Later Training
    Note: May Want to save an extra copy of unused model to be used when resetting model in reset_model() method (See exampleNN_model_class)
    '''
    def __init__(self):
        pass

    '''
    Input: data: NumPy array of size (x, 40, 40) where x == number of images in training set, labels: (x, y) NumPy array where
    Note: labels looks like this:
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0.]
     ...
     [0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 1.]]
     Output: Model is trained using data as training data and labels as training labels
     '''
    def train_model(self, data, labels):
        pass

    '''
    Input: data: NumPy array of size (x, 40, 40) where x == number of images in training set
    Output: Returns predictions: NumPy array of shape (x, y) which contains predicted class probabilities of each predicted class where
    y is the number of class labels
    '''
    def get_predictions(self, data):
        predictions = None
        return predictions

    '''
    Input: None
    Output: Reinitialize model to original untrained version (i.e. The same version of the model that calling the __init__ method generates)
    '''
    def reset_model(self):
        self.model = copy.deepcopy(self.original_model)


#Class that contains CNN classifier
class CNN_model_class(model_object):

    def __init__(self):
        num_classes = 3 #Three Possible Output Classes (From Class Mapping Dicts)
        input_shape = (48, 48, 1) #Images are 48 x 48 pixels

        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='same', activation=keras.activations.relu))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.1))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same', activation=keras.activations.relu))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same', activation=keras.activations.relu))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Nadam(lr=0.0002, schedule_decay=1e-4),
                      metrics=['accuracy'])

        self.model = model
        self.original_model = model

    #Perform training on data and labels
    def train_model(self, data, labels):

        data = np.expand_dims(data, axis=3)

        self.model.fit(data, labels, batch_size=500, epochs=15, verbose=1)

    def get_predictions(self, data):
        data = np.expand_dims(data, axis=3)
        return(self.model.predict(data))

    def reset_model(self):
        self.model.reset_states()


#Class that holds Eigen-Face classifier
class Eigen_model_class(model_object):

    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=500, verbose=True, early_stopping=True, max_iter=30)
        self.original_model = self.model

    def train_model(self, data, labels):
        n, nx, ny = data.shape
        data_2 = data.reshape((n, nx * ny))
        n_components = 100
        self.pca = PCA(n_components=n_components, whiten=True).fit(data_2)
        data_pca = self.pca.transform(data_2)
        self.model.fit(data_pca, labels)


    def get_predictions(self, data):
        # data = np.expand_dims(data, axis=3)
        n, nx, ny = data.shape
        data_2 = data.reshape((n, nx * ny))
        test_pca = self.pca.transform(data_2)
        scaled_preds = []
        preds = self.model.predict_proba(test_pca)
        for i in range(preds.shape[0]):
          scaled_preds.append(minmax_scale(preds[i]))

        scaled_preds = np.asarray(scaled_preds)
        return(scaled_preds)


#Class that holds co-training classifier
class cotraining_object:

    #Holds two different model objects
    def __init__(self, modelA, modelB):
        self.modelA = modelA
        self.modelB = modelB

    def train_models(self, data, labels):
        self.modelA.train_model(data, labels)
        self.modelB.train_model(data, labels)
        return

    #Return predictions from both sub-classifiers
    def get_cotraining_predictions(self, data):
        return self.modelA.get_predictions(data), self.modelB.get_predictions(data)

    #Return actual co-training predictions (Product of sub-classifier predictions)
    def get_final_cotraining_predictions(self, data):
        return np.multiply(self.modelA.get_predictions(data), self.modelB.get_predictions(data))

    #Untrain model (Not used)
    def reset_models(self):
        self.modelA.reset_model()
        self.modelB.reset_model()

#The methods below contain the full experiment for each type of classifier
################################################################################

def perform_CNN():

    #################Train on 20% of German Data... Learn other 80%#############
    print("CREATING GERMANY DATA OBJECT")
    germany_data_object_CNN = data_object("Germany", .8)

    model_object_CNN = CNN_model_class()

    #Start with 20% of German data labeled... learn other 80% with CNN
    round_count = 0
    germany_accuracy_array_CNN = []
    germany_correctly_labeled_array_CNN = []
    while(not germany_data_object_CNN.all_data_labeled()):
        print("ON ROUND " + str(round_count))
        #Keep track of how many labeled samples are labeled correctly
        num_correctly_labeled, total_num_labeled = test_against_true_vals(germany_data_object_CNN.train_data, germany_data_object_CNN.labeled_data)
        germany_correctly_labeled_array_CNN.append(num_correctly_labeled/total_num_labeled)
        with open("cotrain_validation_germany_CNN.txt", "a+") as output:
            output.write("ON ROUND " + str(round_count) + " " + str(num_correctly_labeled) + " out of " + str(total_num_labeled) + " labeled correctly\n")
            output.write("PERCENT LABELED CORRECTLY PERCENTAGE ARRAY IS NOW: " + str(germany_correctly_labeled_array_CNN) + "\n\n")

        #Train CNN model on labeled data and label 2000 unlabeled samples per round
        model_object_CNN.train_model(germany_data_object_CNN.labeled_data['data'], germany_data_object_CNN.labeled_data['labels'])
        predictions = model_object_CNN.get_predictions(germany_data_object_CNN.unlabeled_data['data'])
        germany_data_object_CNN.update_data_single_classifier(predictions, 2000)

        #Keep track of the accuracy of the CNN classifier on the German test data at the end of each round
        test_predictions = model_object_CNN.get_predictions(germany_data_object_CNN.test_data['data'])
        test_accuracy = germany_data_object_CNN.test_predictions(germany_data_object_CNN.test_data['labels'], test_predictions)
        germany_accuracy_array_CNN.append(test_accuracy)
        with open("result_accuracies_germany_CNN.txt", "a+") as output:
            output.write("TEST ACCURACY ON ROUND " + str(round_count) + ": " + str(germany_accuracy_array_CNN) + "\n\n")

        #Increment Round Counter
        round_count = round_count + 1


    ####Use classifiers trained on German data to start with 20% of Italian data and learn other 80%###
    italy_data_object_CNN = data_object("Italy", .8)

    #Start with 20% of Italian data labeled... learn other 80% with pre-trained CNN
    round_count = 0
    italy_accuracy_array_CNN = []
    italy_correctly_labeled_array_CNN = []
    while(not italy_data_object_CNN.all_data_labeled()):

        #Keep track of how many labeled samples are labeled correctly
        num_correctly_labeled, total_num_labeled = test_against_true_vals(italy_data_object_CNN.train_data, italy_data_object_CNN.labeled_data)
        italy_correctly_labeled_array_CNN.append(num_correctly_labeled/total_num_labeled)
        with open("cotrain_validation_italy_CNN.txt", "a+") as output:
            output.write("ON ROUND " + str(round_count) + " " + str(num_correctly_labeled) + " out of " + str(total_num_labeled) + " labeled correctly\n")
            output.write("PERCENT LABELED CORRECTLY PERCENTAGE ARRAY IS NOW: " + str(italy_correctly_labeled_array_CNN) + "\n\n")

        #Train CNN model on labeled data and label 2000 unlabeled samples per round
        model_object_CNN.train_model(italy_data_object_CNN.labeled_data['data'], italy_data_object_CNN.labeled_data['labels'])
        predictions = model_object_CNN.get_predictions(italy_data_object_CNN.unlabeled_data['data'])
        italy_data_object_CNN.update_data_single_classifier(predictions, 2000)

        #Keep track of the accuracy of the CNN classifier on the German test data at the end of each round
        test_predictions = model_object_CNN.get_predictions(italy_data_object_CNN.test_data['data'])
        test_accuracy = italy_data_object_CNN.test_predictions(italy_data_object_CNN.test_data['labels'], test_predictions)
        italy_accuracy_array_CNN.append(test_accuracy)
        with open("result_accuracies_italy_CNN.txt", "a+") as output:
            output.write("TEST ACCURACY ON ROUND " + str(round_count) + ": " + str(italy_accuracy_array_CNN) + "\n\n")

        #Increment Round Counter
        round_count = round_count + 1


def perform_EIGEN():

    #################Train on 10% of German Data... Learn other 90%#############
    germany_data_object_EIGEN = data_object("Germany", .8)
    model_object_EIGEN = Eigen_model_class()


    #Start with 10% of German data labeled... learn other 90% with Eigen-Face Algorithm
    round_count = 0
    germany_accuracy_array_EIGEN = []
    germany_correctly_labeled_array_EIGEN = []
    while(not germany_data_object_EIGEN.all_data_labeled()):

        #Keep track of how many labeled samples are labeled correctly
        num_correctly_labeled, total_num_labeled = test_against_true_vals(germany_data_object_EIGEN.train_data, germany_data_object_EIGEN.labeled_data)
        germany_correctly_labeled_array_EIGEN.append(num_correctly_labeled/total_num_labeled)
        with open("cotrain_validation_germany_EIGEN.txt", "a+") as output:
            output.write("ON ROUND " + str(round_count) + " " + str(num_correctly_labeled) + " out of " + str(total_num_labeled) + " labeled correctly\n")
            output.write("PERCENT LABELED CORRECTLY PERCENTAGE ARRAY IS NOW: " + str(germany_correctly_labeled_array_EIGEN) + "\n\n")

        #Train EIGEN model on labeled data and label 2000 unlabeled samples per round
        model_object_EIGEN.train_model(germany_data_object_EIGEN.labeled_data['data'], germany_data_object_EIGEN.labeled_data['labels'])
        predictions = model_object_EIGEN.get_predictions(germany_data_object_EIGEN.unlabeled_data['data'])
        germany_data_object_EIGEN.update_data_single_classifier(predictions, 2000)

        #Keep track of the accuracy of the EIGEN classifier on the German test data at the end of each round
        test_predictions = model_object_EIGEN.get_predictions(germany_data_object_EIGEN.test_data['data'])
        test_accuracy = germany_data_object_EIGEN.test_predictions(germany_data_object_EIGEN.test_data['labels'], test_predictions)
        germany_accuracy_array_EIGEN.append(test_accuracy)
        with open("result_accuracies_germany_EIGEN.txt", "a+") as output:
            output.write("TEST ACCURACY ON ROUND " + str(round_count) + ": " + str(germany_accuracy_array_EIGEN) + "\n\n")

        #Increment Round Counter
        round_count = round_count + 1



    ####Use classifiers trained on German data to start with 20% of Italian data and learn other 80%###
    italy_data_object_EIGEN = data_object("Italy", .8)

    #Start with 20% of Italian data labeled... learn other 80% with pre-trained Eigen-Face Classifier
    round_count = 0
    italy_accuracy_array_EIGEN = []
    italy_correctly_labeled_array_EIGEN = []
    while(not italy_data_object_EIGEN.all_data_labeled()):

        #Keep track of how many labeled samples are labeled correctly
        num_correctly_labeled, total_num_labeled = test_against_true_vals(italy_data_object_EIGEN.train_data, italy_data_object_EIGEN.labeled_data)
        italy_correctly_labeled_array_EIGEN.append(num_correctly_labeled/total_num_labeled)
        with open("cotrain_validation_italy_EIGEN.txt", "a+") as output:
            output.write("ON ROUND " + str(round_count) + " " + str(num_correctly_labeled) + " out of " + str(total_num_labeled) + " labeled correctly\n")
            output.write("PERCENT LABELED CORRECTLY PERCENTAGE ARRAY IS NOW: " + str(italy_correctly_labeled_array_EIGEN) + "\n\n")

        #Train EIGEN model on labeled data and label 2000 unlabeled samples per round
        model_object_EIGEN.train_model(italy_data_object_EIGEN.labeled_data['data'], italy_data_object_EIGEN.labeled_data['labels'])
        predictions = model_object_EIGEN.get_predictions(italy_data_object_EIGEN.unlabeled_data['data'])
        italy_data_object_EIGEN.update_data_single_classifier(predictions, 2000)

        #Keep track of the accuracy of the EIGEN classifier on the German test data at the end of each round
        test_predictions = model_object_EIGEN.get_predictions(italy_data_object_EIGEN.test_data['data'])
        test_accuracy = italy_data_object_EIGEN.test_predictions(italy_data_object_EIGEN.test_data['labels'], test_predictions)
        italy_accuracy_array_EIGEN.append(test_accuracy)
        with open("result_accuracies_italy_EIGEN.txt", "a+") as output:
            output.write("TEST ACCURACY ON ROUND " + str(round_count) + ": " + str(italy_accuracy_array_EIGEN) + "\n\n")

        #Increment Round Counter
        round_count = round_count + 1


def perform_COTRAIN():

    #################Train on 20% of German Data... Learn other 90%#############
    germany_data_object_COTRAIN = data_object("Germany", .8)

    model_object_CNN_COTRAIN = CNN_model_class()
    model_object_EIGEN_COTRAIN = Eigen_model_class()
    model_object_COTRAIN = cotraining_object(model_object_CNN_COTRAIN, model_object_EIGEN_COTRAIN)


    #Start with 20% of German data labeled... learn other 80% with COTRAIN Algorithm
    round_count = 0
    germany_accuracy_array_COTRAIN = []
    germany_correctly_labeled_array_COTRAIN = []
    while(not germany_data_object_COTRAIN.all_data_labeled()):

        #Keep track of how many labeled samples are labeled correctly
        num_correctly_labeled, total_num_labeled = test_against_true_vals(germany_data_object_COTRAIN.train_data, germany_data_object_COTRAIN.labeled_data)
        germany_correctly_labeled_array_COTRAIN.append(num_correctly_labeled/total_num_labeled)
        with open("cotrain_validation_germany_COTRAIN.txt", "a+") as output:
            output.write("ON ROUND " + str(round_count) + " " + str(num_correctly_labeled) + " out of " + str(total_num_labeled) + " labeled correctly\n")
            output.write("PERCENT LABELED CORRECTLY PERCENTAGE ARRAY IS NOW: " + str(germany_correctly_labeled_array_COTRAIN) + "\n\n")

        #Train COTRAIN model on labeled data and label 2000 unlabeled samples per round
        model_object_COTRAIN.train_models(germany_data_object_COTRAIN.labeled_data['data'], germany_data_object_COTRAIN.labeled_data['labels'])
        predictions_CNN, predictions_EIGEN = model_object_COTRAIN.get_cotraining_predictions(germany_data_object_COTRAIN.unlabeled_data['data'])
        germany_data_object_COTRAIN.update_data(predictions_CNN, predictions_EIGEN, 1000)

        #Keep track of the accuracy of the COTRAIN classifier on the German test data at the end of each round
        test_predictions = model_object_COTRAIN.get_final_cotraining_predictions(germany_data_object_COTRAIN.test_data['data'])
        test_accuracy = germany_data_object_COTRAIN.test_predictions(germany_data_object_COTRAIN.test_data['labels'], test_predictions)
        germany_accuracy_array_COTRAIN.append(test_accuracy)
        with open("result_accuracies_germany_COTRAIN.txt", "a+") as output:
            output.write("TEST ACCURACY ON ROUND " + str(round_count) + ": " + str(germany_accuracy_array_COTRAIN) + "\n\n")

        #Increment Round Counter
        round_count = round_count + 1


    ####Use classifiers trained on German data to start with 20% of Italian data and learn other 90%###
    italy_data_object_COTRAIN = data_object("Italy", .8)

    #Start with 20% of Italian data labeled... learn other 80% with COTRAIN Algorithm
    round_count = 0
    italy_accuracy_array_COTRAIN = []
    italy_correctly_labeled_array_COTRAIN = []
    while(not italy_data_object_COTRAIN.all_data_labeled()):

        #Keep track of how many labeled samples are labeled correctly
        num_correctly_labeled, total_num_labeled = test_against_true_vals(italy_data_object_COTRAIN.train_data, italy_data_object_COTRAIN.labeled_data)
        italy_correctly_labeled_array_COTRAIN.append(num_correctly_labeled/total_num_labeled)
        with open("cotrain_validation_italy_COTRAIN.txt", "a+") as output:
            output.write("ON ROUND " + str(round_count) + " " + str(num_correctly_labeled) + " out of " + str(total_num_labeled) + " labeled correctly\n")
            output.write("PERCENT LABELED CORRECTLY PERCENTAGE ARRAY IS NOW: " + str(italy_correctly_labeled_array_COTRAIN) + "\n\n")

        #Train COTRAIN model on labeled data and label 2000 (1000 per classifier) unlabeled samples per round
        model_object_COTRAIN.train_models(italy_data_object_COTRAIN.labeled_data['data'], italy_data_object_COTRAIN.labeled_data['labels'])
        predictions_CNN, predictions_EIGEN = model_object_COTRAIN.get_cotraining_predictions(italy_data_object_COTRAIN.unlabeled_data['data'])
        italy_data_object_COTRAIN.update_data(predictions_CNN, predictions_EIGEN, 1000)

        #Keep track of the accuracy of the COTRAIN classifier on the German test data at the end of each round
        test_predictions = model_object_COTRAIN.get_final_cotraining_predictions(italy_data_object_COTRAIN.test_data['data'])
        test_accuracy = italy_data_object_COTRAIN.test_predictions(italy_data_object_COTRAIN.test_data['labels'], test_predictions)
        italy_accuracy_array_COTRAIN.append(test_accuracy)
        with open("result_accuracies_italy_COTRAIN.txt", "a+") as output:
            output.write("TEST ACCURACY ON ROUND " + str(round_count) + ": " + str(italy_accuracy_array_COTRAIN) + "\n\n")

        #Increment Round Counter
        round_count = round_count + 1


###########################################################################################################
###########################################################################################################
###########################################################################################################


#Perform experiment for all three classifiers in succession
def main():
    perform_CNN()
    perform_EIGEN()
    perform_COTRAIN()


if __name__ == "__main__": main()
