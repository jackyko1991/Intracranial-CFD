import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Model:
	def __init__(self,
		X_train=[],
		X_test=[],
		y_train=[],
		y_test=[],
		method="SVM",
		plot=False
		):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		assert method in ["LogisticRegression","RandomForest","SVM_Linear","SVM_RBF","NN"], \
			"Model method only allow LogisticRegression, RandomForest, SVM_Linear, SVM_RBF or NN"
		self.method = method
		self.plot = plot

	def train(self):
		if self.method == "RandomForest":
			clf = RandomForestClassifier(max_depth=3, random_state=0)
		elif self.method == "SVM_Linear":
			clf = svm.SVC(kernel='linear')
		elif self.method == "SVM_RBF":
			clf = svm.SVC(kernel='rbf')
		elif self.method == "LogisticRegression":
			clf = LogisticRegression(random_state=0)
		elif self.method == "NN":
			clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32),random_state=1, max_iter=1000)

		clf = OneVsRestClassifier(clf)
		clf.fit(self.X_train, self.y_train)
		self.y_train_fit = clf.predict(self.X_train)
		self.y_test_fit = clf.predict(self.X_test)

		# acc_train = accuracy_score(self.y_train, self.y_train_fit)
		# acc_test = accuracy_score(self.y_test, self.y_test_fit)

		# print("Training Accuracy: {:.2f}%".format(acc_train*100))
		# print("Testing Accuracy: {:.2f}%".format(acc_test*100))

		# target_names = ['non stroke', 'stroke']
		# # print("Training report:")
		# # print(classification_report(self.y_train, self.y_train_fit, target_names=target_names))
		# # print("Testing report:")
		# # print(classification_report(self.y_test, self.y_test_fit, target_names=target_names))

		# # y_true = np.array([[0, 0, 1],
		# # 					[0, 1, 0],
		# # 					[1, 1, 0]])
		# # y_pred = np.array([[0, 1, 0],
		# # 					[0, 0, 1],
		# # 					[1, 1, 0]])
		# # mcm = metrics.multilabel_confusion_matrix(y_true, y_pred)
		# # print(mcm)
		# # print("***")
		# # print(mcm[:,0,0])

		# mcm = metrics.multilabel_confusion_matrix(self.y_train, self.y_train_fit)

		# tn = mcm[:, 0, 0]
		# tp = mcm[:, 1, 1]
		# fn = mcm[:, 1, 0]
		# fp = mcm[:, 0, 1]
		# sensitivity = tp / (tp + fn)
		# specificity = tn / (tn + fp)
		# print("Training sensitivity: {:.2f}%".format(sensitivity[1]*100))
		# print("Training specificity: {:.2f}%".format(specificity[1]*100))

		# mcm = metrics.multilabel_confusion_matrix(self.y_test, self.y_test_fit)

		# tn = mcm[:, 0, 0]
		# tp = mcm[:, 1, 1]
		# fn = mcm[:, 1, 0]
		# fp = mcm[:, 0, 1]
		# sensitivity = tp / (tp + fn)
		# specificity = tn / (tn + fp)

		# print("TP: {}, FP: {}, TN: {}, FN: {}".format(tp,fp,tn,fn))

		# print("Testing sensitivity: {:.2f}%".format(sensitivity[1]*100))
		# print("Testing specificity: {:.2f}%".format(specificity[1]*100))

		return clf

	def get_metric(self):
		return

def main():
	# result_csv = "Z:/projects/intracranial/results.csv"
	result_csv = "/Volumes/shared/projects/intracranial/results.csv"
	result = pd.read_csv(result_csv)
	method = "RandomForest"
	# method = "LogisticRegression"
	# method = "SVM_Linear"
	# method = "SVM_RBF"
	# method = "NN"

	result_X = result[[
		"radius mean(mm)",
		"radius min(mm)",
		"pressure mean(mmHg)",
		"max pressure gradient(mmHg)",
		"in/out pressure gradient(mmHg)",
		"velocity mean(ms^-1)",
		"peak velocity(ms^-1)",
		"max velocity gradient(ms^-1)",
		"vorticity mean(s^-1)",	
		"peak vorticity(s^-1)"
		]]

	result_Y = result[["Stroke","Severity","ICAD"]]

	result_X_array = result_X.to_numpy()
	# severity
	# classes = [0,1,2]
	# classnames = ["normal","moderate","severe"]
	classes = [0,1]
	classnames = ["normal","ICAD"]
	n_classes = len(classes)

	# stroke: [:,0], severity: [:,1], "icad": [:,2]
	result_Y_array = label_binarize(result_Y.to_numpy()[:,1],classes=classes)
	result_Y_array = label_binarize(result_Y.to_numpy()[:,2],classes=classes)

	# model
	model = Model(method=method)

	# K fold
	kf = KFold(n_splits=5,shuffle=True,random_state=0)

	# plot
	fig, axs = plt.subplots(2,n_classes,figsize=(5*n_classes, 10))
	fig.suptitle("ROC of Classifier: {}".format(method))

	for i, ax in enumerate(axs.flatten()):
		ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
		if np.floor(i/n_classes) == 0:
			ax.set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',title="{} ({})".format(classnames[i%n_classes],"Train"))
		else:
			ax.set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',title="{} ({})".format(classnames[i%n_classes],"Test"))

	tprs_train = []
	fprs_train = []
	aucs_train = []
	tprs_test = []
	fprs_test = []
	aucs_test = []
	mean_fpr = np.linspace(0, 1, 100)

	for i, (train_index, test_index) in enumerate(kf.split(result_X_array)):
		X_train, X_test = result_X_array[train_index], result_X_array[test_index]
		y_train, y_test = result_Y_array[train_index], result_Y_array[test_index]

		model.X_train = X_train
		model.X_test = X_test
		model.y_train = y_train
		model.y_test = y_test

		# perform training
		clf = model.train()

		# Compute ROC curve and ROC area for each class
		fpr_train = dict()
		tpr_train = dict()
		roc_auc_train = dict()
		fpr_test = dict()
		tpr_test = dict()
		roc_auc_test = dict()

		for j in range(n_classes):
			# training
			if "SVM" in method or "LogisticRegression" in method:
				if n_classes >2:
					fpr_train[j], tpr_train[j], _ = roc_curve(y_train[:, j], clf.decision_function(X_train)[:,j])
				else:
					if j == 0:
						fpr_train[j], tpr_train[j], _ = roc_curve(1 - y_train, 1- clf.decision_function(X_train))
					else:
						fpr_train[j], tpr_train[j], _ = roc_curve(y_train, clf.decision_function(X_train))
			# elif "RandomForest" in method or "NN" in method:
			else:
				if n_classes >2:
					fpr_train[j], tpr_train[j], _ = roc_curve(y_train[:, j], clf.predict_proba(X_train)[:,j])
				else:
					if j ==0:
						fpr_train[j], tpr_train[j], _ = roc_curve(1 - y_train, clf.predict_proba(X_train)[:,j])
					else:
						fpr_train[j], tpr_train[j], _ = roc_curve(y_train, clf.predict_proba(X_train)[:,j])
			# else:
			# 	if n_classes > 2:
			# 		fpr_train[j], tpr_train[j], _ = roc_curve(y_train[:, j], clf.predict(X_train)[:,j])					
			# 	else:
			# 		fpr_train[j], tpr_train[j], _ = roc_curve(y_train[:, j], label_binarize(clf.predict(X_train),classes=classes)[:,j])

			roc_auc_train[j] = auc(fpr_train[j], tpr_train[j])
			axs[0,j].plot(fpr_train[j],tpr_train[j], label="fold {} (AUC = {:.2f})".format(i, roc_auc_train[j]),alpha=0.3, lw=1)

			# testing
			if "SVM" in method or "LogisticRegression" in method:
				if n_classes >2:
					fpr_test[j], tpr_test[j], _ = roc_curve(y_test[:, j], clf.decision_function(X_test)[:,j])
				else:
					if j == 0:
						fpr_test[j], tpr_test[j], _ = roc_curve(1 - y_test, 1-clf.decision_function(X_test))
					else:
						fpr_test[j], tpr_test[j], _ = roc_curve(y_test, clf.decision_function(X_test))
			# elif "RandomForest" in method or "NN" in method:
			else:
				if n_classes >2:
					fpr_test[j], tpr_test[j], _ = roc_curve(y_test[:, j], clf.predict_proba(X_test)[:,j])
				else:
					if j == 0:
						fpr_test[j], tpr_test[j], _ = roc_curve(1 - y_test, clf.predict_proba(X_test)[:,j])
					else:
						fpr_test[j], tpr_test[j], _ = roc_curve(y_test, clf.predict_proba(X_test)[:,j])
			# else:
			# 	if n_classes >2:
			# 		fpr_test[j], tpr_test[j], _ = roc_curve(y_test[:, j], clf.predict(X_test)[:,j])
			# 	else:
			# 		fpr_test[j], tpr_test[j], _ = roc_curve(y_test[:, j], label_binarize(clf.predict(X_test),classes=classes)[:,j])

			roc_auc_test[j] = auc(fpr_test[j], tpr_test[j])
			axs[1,j].plot(fpr_test[j],tpr_test[j], label="fold {} (AUC = {:.2f})".format(i, roc_auc_test[j]),alpha=0.3, lw=1)

		tprs_train.append(tpr_train)
		fprs_train.append(fpr_train)
		aucs_train.append(roc_auc_train)

		tprs_test.append(tpr_test)
		fprs_test.append(fpr_test)
		aucs_test.append(roc_auc_test)

	# average
	mean_fpr = np.linspace(0,1,100);
	for i in range(n_classes):
		# train
		tprs = []
		for tpr, fpr in zip(tprs_train,fprs_train):
			interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
			tprs.append(interp_tpr)
		mean_tpr = np.mean(tprs,axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = metrics.auc(mean_fpr, mean_tpr)
		aucs = [auc[i] for auc in aucs_train]
		std_auc = np.std(aucs)
		axs[0,i].plot(mean_fpr, mean_tpr, color='b',
			label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
			lw=2, alpha=.8)

		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		axs[0,i].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
			label=r'$\pm$ 1 std. dev.')

		# test
		tprs = []
		for tpr, fpr in zip(tprs_test,fprs_test):
			interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
			tprs.append(interp_tpr)
		mean_tpr = np.mean(tprs,axis=0)
		mean_tpr[0] = 0.0
		mean_tpr[-1] = 1.0
		mean_auc = metrics.auc(mean_fpr, mean_tpr)
		aucs = [auc[i] for auc in aucs_test]
		std_auc = np.std(aucs)
		axs[1,i].plot(mean_fpr, mean_tpr, color='b',
			label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
			lw=2, alpha=.8)

		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		axs[1,i].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
			label=r'$\pm$ 1 std. dev.')

	for i, ax in enumerate(axs.flatten()):
		ax.legend(loc="lower right")
	plt.show()

if __name__ == "__main__":
	main()