import pandas as pd
import numpy as np



# Define functions for trainin and predicting classes from categorical variables.
def NaiveBayes_train_categorical(dataset, targets):
	p_c_1 = ((targets == 1).sum() / len(targets)).item()
	p_c_0 = 1 - p_c_1

	class1_data = dataset[targets.iloc[:,0] == 1]
	class0_data = dataset[targets.iloc[:,0] == 0]
	
	def compute_categorical_probs(data):
		feature_probs = {}
		for col in data.columns:
			feature_probs[col] = data[col].value_counts(normalize=True).to_dict()
		return(feature_probs)
	
	class1_probs = compute_categorical_probs(class1_data)
	class0_probs = compute_categorical_probs(class0_data)
	return(p_c_1, p_c_0, class1_probs, class0_probs)


def NaiveBayes_predict_categorical(X_test, Y_test, train_out):
	import numpy as np
	import math

	predictions = []
	for sample_index in range(len(X_test)):
		sample = X_test.iloc[sample_index, :]
		class_1_prob, class_0_prob, class1_feature_prob, class0_feature_prob = train_out

		class1_score = np.log(class_1_prob)
		class0_score = np.log(class_0_prob)
		for feature_name, feature_value in sample.items():
			class1_score += np.log(class1_feature_prob[feature_name].get(feature_value, 1e-12))
			class0_score += np.log(class0_feature_prob[feature_name].get(feature_value, 1e-12))

		prediction = 1 if class1_score > class0_score else 0

		predictions.append(prediction)
	
	acc = ((np.array(predictions) == Y_test.iloc[:,0].to_list())).sum() / len(np.array(Y_test))
	return(predictions, round(acc, 2))




# Define a function to split the data into training and testing sets
def split_train_test(X,Y, train_percentage):
	import random
    
	len_dataset = len(X)
	test_n_indices = int(train_percentage * len_dataset)

	train_idx = random.sample(range(len_dataset), test_n_indices)
	test_idx = [x for x in range(len_dataset) if x not in train_idx]

	return(X.iloc[train_idx,:], Y.iloc[train_idx,:], X.iloc[test_idx,:], Y.iloc[test_idx,:])


# Define functions for training and predicting classes from continuous variables.
def NaiveBayes_train_continuous(dataset, targets):
	p_c_1 = ((targets == 1).sum() / len(targets)).item()
	p_c_0 = 1 - p_c_1

	class1_data = dataset[targets.iloc[:,0] == 1]
	class0_data = dataset[targets.iloc[:,0] == 0]
	
	def compute_continuous_probs(data):
		return(data.mean(), data.var())

	return(p_c_1, p_c_0, compute_continuous_probs(class1_data), compute_continuous_probs(class1_data))




def NaiveBayes_predict_continuous(X_test, Y_test, train_out):
	import numpy as np
	import math
	def gaussian_prob(x, mean, var):
		return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

	predictions = []
	for sample_index in range(len(X_test)):
		sample = X_test.iloc[sample_index, :]
		class_1_prob, class_0_prob, class1_mean_variance, class0_mean_variance = train_out

		class1_score = np.log(class_1_prob)
		class0_score = np.log(class_0_prob)
		for feature_index, feature_value in enumerate(sample):
			class1_score += np.log(gaussian_prob(x = feature_value, mean = class1_mean_variance[0][feature_index], var= class1_mean_variance[1][feature_index]))
			class0_score += np.log(gaussian_prob(x = feature_value, mean = class0_mean_variance[0][feature_index], var= class0_mean_variance[1][feature_index]))

		prediction = 1 if class1_score > class0_score else 0

		predictions.append(prediction)
	
	acc = ((np.array(predictions) == Y_test.iloc[:,0].to_list())).sum() / len(np.array(Y_test))
	return(predictions, round(acc, 2))



# Put everything together in one main function.
def NaiveBayes(X,Y, training_percentage, feature_type):
	train_data, train_targets, test_data, test_targets = split_train_test(X= X,Y=Y, train_percentage=training_percentage)
	if feature_type == 'categorical':
		train_out = NaiveBayes_train_categorical(dataset = train_data, targets = train_targets)
		predictions, acc = NaiveBayes_predict_categorical(X_test = test_data, Y_test = test_targets, train_out = train_out)

	elif feature_type == 'continuous':
		train_out = NaiveBayes_train_continuous(dataset = train_data, targets = train_targets)
		predictions, acc = NaiveBayes_predict_continuous(X_test = test_data, Y_test = test_targets, train_out = train_out)
	
	return(predictions, acc)


