import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def count_column_pandn(data_frame, attr_output):
	num_data = len(data_frame)
	p_data_frame = data_frame[data_frame[attr_output] == 1]
	return float(len(p_data_frame)), float(num_data - len(p_data_frame))

def info_entropy(data_frame, attr_output):
	p, n = count_column_pandn(data_frame, attr_output)
	if p  != 0 and n != 0:
		entropy = ((-1*p)/(p + n))*np.log2(p/(p+n)) + ((-1*n)/(p + n))*np.log2(n/(p+n))
	else:
		entropy = 0
	return entropy

def info_gain(data_frame, attribute, threshold, attr_output):
	lendf = len(data_frame)
	sub1 = data_frame[data_frame[attribute] <= threshold]
	len1 = len(sub1)
	sub2 = data_frame[data_frame[attribute] > threshold]
	len2 = len(sub2)
	gain = info_entropy(data_frame, attr_output) -\
	float(len1/lendf)*info_entropy(sub1, attr_output) -\
	float(len2/lendf)*info_entropy(sub2, attr_output)
	return gain

def select_best_threshold(data_frame, attribute, attr_output):
	column = data_frame[attribute].tolist()
	column = list(set(column))
	column.sort()
	max_gain = float("-inf")
	best_thres = float(0)
	for i in range(1, len(column)):
		thres = float(column[i-1] + column[i])/2
		gain = info_gain(data_frame, attribute, thres, attr_output)
		if max_gain < gain:
			max_gain, best_thres = gain, thres
	return best_thres, max_gain

def choose_attr(data_frame, cols, attr_output):
	max_gain = float("-inf")
	best_thres = float(0)
	best_attr = None
	for col in cols:
		thres, gain = select_best_threshold(data_frame, col, attr_output)
		gain = info_gain(data_frame, col, thres, attr_output)
		if max_gain < gain:
			max_gain, best_thres, best_attr = gain, thres, col
	return best_attr, best_thres


def build_tree(data_frame, cols, attr_output, level=0):
	p, n = count_column_pandn(data_frame, attr_output)
	if p == 0 or n == 0 or level==15:
		if p < n:
			tree = TreeNode(0)
		else:
			tree = TreeNode(1)
	else:
		best_attr, threshold = choose_attr(data_frame, cols, attr_output)
		tree = TreeNode(None, best_attr, threshold)
		sub1 = data_frame[data_frame[best_attr] <= threshold]
		tree.left = build_tree(sub1, cols, attr_output, level+1)
		sub2 = data_frame[data_frame[best_attr] > threshold]
		tree.right = build_tree(sub2, cols, attr_output, level+1)
	return tree

def predict(node, row):
	if node.pred != None:
		return node.pred
	else:
		if row[node.attr] <= node.thres:
			return predict(node.left, row)
		else:
			return predict(node.right, row)

def result_calculator(root, data_frame, attr_output):
	x_name = root.attr
	x_axis = list(data_frame[root.attr])
	if root.attr != root.right.attr:
		y_name = root.right.attr
		y_axis = list(data_frame[root.right.attr])
	else:
		y_name = root.left.attr
		y_axis = list(data_frame[root.left.attr])
	output = data_frame[attr_output]
	color= ['red' if value == 0 else 'blue' for value in output]
	plt.rcParams['figure.figsize'] = [6, 4]
	plt.title('Red for Left = 0 and Green for Left=1')
	plt.xlabel(x_name)
	plt.ylabel(y_name)
	plt.scatter(x_axis, y_axis, color=color)
	plt.show()

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
	root = build_tree(df_train, attributes, 'last_outcome_row')
	
	result_calculator(root, df_test, 'last_outcome_row')

if __name__ == '__main__':
	main()