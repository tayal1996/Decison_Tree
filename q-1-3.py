import numpy as np
import pandas as pd

def count_column_pandn(data_frame, attr_output):
	num_data = len(data_frame)
	p_data_frame = data_frame[data_frame[attr_output] == 1]
	return float(len(p_data_frame)), float(num_data - len(p_data_frame))

def info_gini(df, predict_attr):
	p, n = count_column_pandn(df, predict_attr)
	if p  == 0 or n == 0:
		gini = 0
	else:
		gini = 2*((p)/(p + n))*((n)/(p + n))
	return gini

def info_entropy(df, predict_attr):
	p, n = count_column_pandn(df, predict_attr)
	if p  == 0 or n == 0:
		entropy = 0
	else:
		entropy = ((-1*p)/(p + n))*np.log2(p/(p+n)) + ((-1*n)/(p + n))*np.log2(n/(p+n))
	return entropy

def info_misclass(df, predict_attr):
	p, n = count_column_pandn(df, predict_attr)
	if p  == 0 or n == 0:
		misclass = 0
	else:
		misclass = min((p/(p + n)),(n/(p + n)))
	return misclass

def info_gain(data_frame, attribute, threshold, attr_output, info_func):
	lendf = len(data_frame)
	sub1 = data_frame[data_frame[attribute] <= threshold]
	len1 = len(sub1)
	sub2 = data_frame[data_frame[attribute] > threshold]
	len2 = len(sub2)
	gain = info_func(data_frame, attr_output) -\
	float(len1/lendf)*info_func(sub1, attr_output) -\
	float(len2/lendf)*info_func(sub2, attr_output)
	return gain

def select_best_threshold(data_frame, attribute, attr_output, info_func):
	column = data_frame[attribute].tolist()
	column = list(set(column))
	column.sort()
	max_gain = float("-inf")
	best_thres = float(0)
	for i in range(1, len(column)):
		thres = float(column[i-1] + column[i])/2
		gain = info_gain(data_frame, attribute, thres, attr_output, info_func)
		if max_gain < gain:
			max_gain, best_thres = gain, thres
	return best_thres, max_gain

def choose_attr(data_frame, cols, attr_output, info_func):
	max_gain = float("-inf")
	best_thres = float(0)
	best_attr = None
	for col in cols:
		thres, gain = select_best_threshold(data_frame, col, attr_output, info_func)
		gain = info_gain(data_frame, col, thres, attr_output, info_func)
		if max_gain < gain:
			max_gain, best_thres, best_attr = gain, thres, col
	return best_attr, best_thres


def build_tree(data_frame, cols, attr_output, info_func, level=0):
	p, n = count_column_pandn(data_frame, attr_output)
	if p == 0 or n == 0 or level==15:
		if p < n:
			tree = TreeNode(0)
		else:
			tree = TreeNode(1)
	else:
		best_attr, threshold = choose_attr(data_frame, cols, attr_output, info_func)
		tree = TreeNode(None, best_attr, threshold)
		sub1 = data_frame[data_frame[best_attr] <= threshold]
		tree.left = build_tree(sub1, cols, attr_output, info_func, level+1)
		sub2 = data_frame[data_frame[best_attr] > threshold]
		tree.right = build_tree(sub2, cols, attr_output, info_func ,level+1)
	return tree

def predict(node, row):
	if node.pred != None:
		return node.pred
	else:
		if row[node.attr] <= node.thres:
			return predict(node.left, row)
		else:
			return predict(node.right, row)

def result_calculator(root, data_frame, attr_output, formulaname):
	TP,TN,FP,FN=0,0,0,0
	for index,row in data_frame.iterrows():
	# for row in data_frame:
		prediction = predict(root, row)
		if prediction == row[attr_output] == 1:
			TP += 1
		elif prediction == row[attr_output] == 0:
			TN += 1
		elif row[attr_output] == 0 and prediction == 1:
			FP += 1
		else:
			FN += 1
	print("TP = {} TN = {} FP = {} FN = {} ".format(TP,TN,FP,FN))
	accuracy,precision,recall,f1_score = 0,0,0,0
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	if TP+FP != 0:
		precision = TP/(TP+FP)
	if TP+FN != 0:
		recall = TP/(TP+FN)
	if precision+recall != 0:
		f1_score = (2*precision*recall)/(precision+recall)
	print("\nFor ",formulaname)
	print("TP = {} TN = {} FP = {} FN = {} ".format(TP,TN,FP,FN))
	print("Accuracy : %0.2f" %(accuracy*100),"%")
	print("Precision : %0.2f" %(precision*100),"%")
	print("Recall : %0.2f" %(recall*100),"%")
	print("f1-score : %0.2f" %(f1_score*100),"%\n")

class TreeNode():
	def __init__(self, p = None, a = None, t = None):
		self.pred = p
		self.attr = a
		self.thres = t
		self.left = None
		self.right = None

def main():
	data_frame = pd.read_csv('train.csv')
	data_frame['last_outcome_row'] = 0
	data_frame.loc[data_frame['left'] == 1, 'last_outcome_row'] = 1
	data_frame.drop(['left'], axis=1 )
	for col in ['sales','salary']:
		data_frame[col] = pd.factorize(data_frame[col])[0]
	df_train = data_frame.sample(frac = 0.2)
	df_test = data_frame.drop(df_train.index)
	attributes =  ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']
	# attributes =  ['Work_accident','promotion_last_5years','sales','salary']

	root_misclass = build_tree(df_train, attributes, 'last_outcome_row', info_misclass)
	root_gini = build_tree(df_train, attributes, 'last_outcome_row', info_gini)
	root_entropy = build_tree(df_train, attributes, 'last_outcome_row', info_entropy)

	result_calculator(root_misclass, df_test, 'last_outcome_row', "MisClassification Rate")
	result_calculator(root_gini, df_test, 'last_outcome_row', "Gini")
	result_calculator(root_entropy, df_test, 'last_outcome_row', "Entropy")

if __name__ == '__main__':
	main()