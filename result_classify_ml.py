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
from tqdm import tqdm
import os

class Model:
	def __init__(self,
		X_train=[],
		X_test=[],
		y_train=[],
		y_test=[],
		method="SVM",
		plot=False,
		mlp_iter=100
		):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		assert method in ["LogisticRegression","RandomForest","SVM_Linear","SVM_RBF","MLP"], \
			"Model method only allow LogisticRegression, RandomForest, SVM_Linear, SVM_RBF or MLP"
		self.method = method
		self.plot = plot
		self.mlp_iter = mlp_iter

	def train(self):
		if self.method == "RandomForest":
			clf = RandomForestClassifier(max_depth=3, random_state=0)
		elif self.method == "SVM_Linear":
			clf = svm.SVC(kernel='linear')
		elif self.method == "SVM_RBF":
			clf = svm.SVC(kernel='rbf')
		elif self.method == "LogisticRegression":
			clf = LogisticRegression(random_state=0)
		elif self.method == "NN" or self.method == "MLP":
			clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32),random_state=1, max_iter=self.mlp_iter)

		clf = OneVsRestClassifier(clf)
		clf.fit(self.X_train, self.y_train)

		return clf

	def get_metrics():
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


class Classify:
	def __init__(self,
		X,
		y,
		method="LogisticRegression",
		classnames=[],
		show_plot=False,
		save_plot=False,
		plot_dir="",
		n_folds = 5,
		):

		self.X = X
		self.y = y

		self.method = method
		self.classnames = classnames
		self.show_plot = show_plot
		self.save_plot = save_plot
		self.plot_dir = plot_dir
		self.n_folds = n_folds

	@property
	def classnames(self):
		return self.__classnames

	@classnames.setter
	def classnames(self, classnames):
		self.n_classes = len(classnames)
		assert self.n_classes>0, "length of classnames should be > 0"

		self.__classnames = classnames

	def run(self):
		# model
		model = Model(method=self.method)

		# K fold
		kf = KFold(n_splits=self.n_folds,shuffle=True,random_state=0)

		# plot
		if self.n_classes < 3:
			fig, axs = plt.subplots(1, self.n_classes,figsize=(5*self.n_classes, 5))
		else:
			fig, axs = plt.subplots(2,self.n_classes,figsize=(5*self.n_classes, 10))
		fig.suptitle("ROC of Classifier: {}".format(self.method))

		if self.n_classes < 3:
			axs[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
			axs[0].set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',title="Train")
			axs[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
			axs[1].set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',title="Test")
		else:
			for i, ax in enumerate(axs.flatten()):
				ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
				if np.floor(i/self.n_classes) == 0:
					ax.set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',title="{} ({})".format(self.classnames[i%self.n_classes],"Train"))
				else:
					ax.set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',title="{} ({})".format(self.classnames[i%self.n_classes],"Test"))

		tprs_train = []
		fprs_train = []
		aucs_train = []
		tprs_test = []
		fprs_test = []
		aucs_test = []
		mean_fpr = np.linspace(0, 1, 100)

		columnNames = ["Method","Train/Test","Fold"]
		if self.n_classes <3:
			columnNames.append("AUC") 
		else:
			for i in range(self.n_classes):
				columnNames.append("AUC {}".format(str(i))) 
			
		fold_aucs = {columnName: [] for columnName in columnNames}

		columnNames = ["Method","Train/Test"]
		if self.n_classes <3:
			columnNames.append("AUC") 
			columnNames.append("SD")
		else: 
			for i in range(self.n_classes):
				columnNames.append("AUC {}".format(str(i)))
				columnNames.append("SD {}".format(str(i)))
		macro_aucs = {columnName: [] for columnName in columnNames}

		for i, (train_index, test_index) in enumerate(kf.split(self.X)):
			X_train, X_test = self.X[train_index], self.X[test_index]
			y_train, y_test = self.y[train_index], self.y[test_index]

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

			if self.n_classes > 2:
				for j in range(self.n_classes):
					if "SVM" in self.method or "LogisticRegression" in self.method:
						fpr_train[j], tpr_train[j], _ = roc_curve(y_train[:, j], clf.decision_function(X_train)[:,j])
						fpr_test[j], tpr_test[j], _ = roc_curve(y_test[:, j], clf.decision_function(X_test)[:,j])
					else:
						fpr_train[j], tpr_train[j], _ = roc_curve(y_train[:, j], clf.predict_proba(X_train)[:,j])
						fpr_test[j], tpr_test[j], _ = roc_curve(y_test[:, j], clf.predict_proba(X_test)[:,j])

					# auc
					roc_auc_train[j] = auc(fpr_train[j], tpr_train[j])
					roc_auc_test[j] = auc(fpr_test[j], tpr_test[j])

					# plot
					axs[0,j].plot(fpr_train[j],tpr_train[j], label="fold {} (AUC = {:.2f})".format(i, roc_auc_train[j]),alpha=0.3, lw=1)
					axs[1,j].plot(fpr_test[j],tpr_test[j], label="fold {} (AUC = {:.2f})".format(i, roc_auc_test[j]),alpha=0.3, lw=1)
			else:
				if "SVM" in self.method or "LogisticRegression" in self.method:
					fpr_train, tpr_train, _ = roc_curve(y_train, clf.decision_function(X_train))
					fpr_test, tpr_test, _ = roc_curve(y_test, clf.decision_function(X_test))
				else:
					if self.n_classes == 1:
						fpr_train, tpr_train, _ = roc_curve(y_train, clf.predict_proba(X_train))
						fpr_test, tpr_test, _ = roc_curve(y_test, clf.predict_proba(X_test))
					else:
						fpr_train, tpr_train, _ = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
						fpr_test, tpr_test, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

				# auc
				roc_auc_train = auc(fpr_train, tpr_train)
				roc_auc_test = auc(fpr_test, tpr_test)

				# plot 
				axs[0].plot(fpr_train,tpr_train, label="fold {} (AUC = {:.2f})".format(i, roc_auc_train),alpha=0.3, lw=1)
				axs[1].plot(fpr_test,tpr_test, label="fold {} (AUC = {:.2f})".format(i, roc_auc_test),alpha=0.3, lw=1)
				
			tprs_train.append(tpr_train)
			fprs_train.append(fpr_train)
			aucs_train.append(roc_auc_train)

			tprs_test.append(tpr_test)
			fprs_test.append(fpr_test)
			aucs_test.append(roc_auc_test)

			fold_aucs["Method"].append(self.method)
			fold_aucs["Train/Test"].append("Train")
			fold_aucs["Fold"].append(i + 1)

			if self.n_classes <3:
				fold_aucs["AUC"].append(roc_auc_train)
			else:
				for j in range(self.n_classes):
					fold_aucs["AUC {}".format(j)].append(roc_auc_train[j])

			fold_aucs["Method"].append(self.method)
			fold_aucs["Train/Test"].append("Test")
			fold_aucs["Fold"].append(i + 1)
			if self.n_classes <3:
				fold_aucs["AUC"].append(roc_auc_test)
			else:
				for j in range(self.n_classes):
					fold_aucs["AUC {}".format(j)].append(roc_auc_test[j])

		# average
		mean_fpr = np.linspace(0,1,100)

		if self.n_classes <3:
			# train
			macro_aucs["Method"].append(self.method)
			macro_aucs["Train/Test"].append("Train")
			tprs = []
			for tpr, fpr in zip(tprs_train,fprs_train):
				interp_tpr = np.interp(mean_fpr, fpr, tpr)
				tprs.append(interp_tpr)
			mean_tpr = np.mean(tprs,axis=0)
			mean_tpr[-1] = 1.0
			mean_auc = metrics.auc(mean_fpr, mean_tpr)
			aucs = [auc for auc in aucs_train]
			std_auc = np.std(aucs)
			axs[0].plot(mean_fpr, mean_tpr, color='b',
				label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
				lw=2, alpha=.8)

			std_tpr = np.std(tprs, axis=0)
			tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
			tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
			axs[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				label=r'$\pm$ 1 std. dev.')

			macro_aucs["AUC"].append(mean_auc)
			macro_aucs["SD"].append(std_auc)

			# test
			macro_aucs["Method"].append(self.method)
			macro_aucs["Train/Test"].append("Test")
			tprs = []
			for tpr, fpr in zip(tprs_test,fprs_test):
				interp_tpr = np.interp(mean_fpr, fpr, tpr)
				tprs.append(interp_tpr)
			mean_tpr = np.mean(tprs,axis=0)
			mean_tpr[-1] = 1.0
			mean_auc = metrics.auc(mean_fpr, mean_tpr)
			aucs = [auc for auc in aucs_test]
			std_auc = np.std(aucs)
			axs[1].plot(mean_fpr, mean_tpr, color='b',
				label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
				lw=2, alpha=.8)

			std_tpr = np.std(tprs, axis=0)
			tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
			tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
			axs[1].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				label=r'$\pm$ 1 std. dev.')

			macro_aucs["AUC"].append(mean_auc)
			macro_aucs["SD"].append(std_auc)
		else:
			# train
			macro_aucs["Method"].append(self.method)
			macro_aucs["Train/Test"].append("Train")
			for i in range(self.n_classes):
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

				macro_aucs["AUC {}".format(str(i))].append(mean_auc)
				macro_aucs["SD {}".format(str(i))].append(std_auc)

			# test
			macro_aucs["Method"].append(self.method)
			macro_aucs["Train/Test"].append("Test")

			for i in range(self.n_classes):
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

				macro_aucs["AUC {}".format(str(i))].append(mean_auc)
				macro_aucs["SD {}".format(str(i))].append(std_auc)

		for i, ax in enumerate(axs.flatten()):
			ax.legend(loc="lower right")

		if self.save_plot and (not self.plot_dir == ""):
			os.makedirs(self.plot_dir,exist_ok=True)
			tqdm.write(("Saving plot..."))
			plt.savefig(os.path.join(self.plot_dir,"roc_{}.png".format(self.method)))

		if self.show_plot:
			plt.show()

		return fold_aucs, macro_aucs

def main():
	result_csv = "Z:/projects/intracranial/results.csv"
	plot_output_dir = "Z:/projects/intracranial/plots/results_radius_min"
	kfold_output_csv = "Z:/projects/intracranial/auc/results_radius_min_kfold_aucs.csv"
	macro_output_csv = "Z:/projects/intracranial/auc/results_radius_min_macro_aucs.csv"

	# result_csv = "Z:/data/intracranial/CFD_results/scores.csv"
	# # result_csv = "/Volumes/shared/projects/intracranial/results.csv"
	# plot_output_dir = "Z:/data/intracranial/CFD_results/plots/scores_sum"
	# kfold_output_csv = "Z:/data/intracranial/CFD_results/auc/scores_sum_kfold_aucs.csv"
	# macro_output_csv = "Z:/data/intracranial/CFD_results/auc/scores_sum_macro_aucs.csv"
	result = pd.read_csv(result_csv)

	show_plot = False
	mlp_iter = 200

	methods = [
		"RandomForest",
		"LogisticRegression",
		# "SVM_Linear",
		"SVM_RBF",
		"MLP"
	]

	# result_X = result[[
	# 	"radius mean(mm)",
	# 	"radius min(mm)",
	# 	"pressure mean(mmHg)",
	# 	"max pressure gradient(mmHg)",
	# 	"in/out pressure gradient(mmHg)",
	# 	"velocity mean(ms^-1)",
	# 	"peak velocity(ms^-1)",
	# 	"max velocity gradient(ms^-1)",
	# 	"vorticity mean(s^-1)",	
	# 	"peak vorticity(s^-1)"
	# 	]]

	# # selected by anova
	# result_X = result[[
	# 	"radius mean(mm)",
	# 	# "radius min(mm)",
	# 	# "pressure mean(mmHg)",
	# 	# "max pressure gradient(mmHg)",
	# 	# "in/out pressure gradient(mmHg)",
	# 	"velocity mean(ms^-1)",
	# 	"peak velocity(ms^-1)",
	# 	# "max velocity gradient(ms^-1)",
	# 	"vorticity mean(s^-1)",	
	# 	"peak vorticity(s^-1)"
	# 	]]

	# select by human
	# result_X = result[[
	# 	"radius min(mm)",
	# 	"in/out pressure gradient(mmHg)",
	# 	"peak velocity(ms^-1)",
	# 	"peak vorticity(s^-1)"
	# 	]]

	result_X = result[[
		"radius min(mm)",
		]]

	result_Y = result[["Stroke","Severity","ICAD"]]

	result_X_array = result_X.to_numpy()
	# severity
	# classes = [0,1,2]
	# classnames = ["normal","moderate","severe"]
	classes = [0,1]
	classnames = ["normal","Stroke"]
	# classnames = ["normal","ICAD"]

	# stroke: [:,0], severity: [:,1], "icad": [:,2]
	result_Y_array = label_binarize(result_Y.to_numpy()[:,0],classes=classes)
	# result_Y_array = label_binarize(result_Y.to_numpy()[:,1],classes=classes)
	# result_Y_array = label_binarize(result_Y.to_numpy()[:,2],classes=classes)

	classify = Classify(result_X_array,result_Y_array,classnames=classnames)
	classify.mlp_iter = mlp_iter
	classify.plot_dir = plot_output_dir
	classify.show_plot = show_plot

	# output df
	columnNames = ["Method","Train/Test","Fold"]
	if len(classes) <3:
		columnNames.append("AUC") 
	else:
		for i in range(len(classes)):
			columnNames.append("AUC {}".format(str(i))) 

	fold_aucs_df = pd.DataFrame(columns=columnNames)

	columnNames = ["Method","Train/Test"]
	if len(classes) <3:
		columnNames.append("AUC")
		columnNames.append("SD")
	else:
		for i in range(len(classes)):
			columnNames.append("AUC {}".format(str(i)))
			columnNames.append("SD {}".format(str(i)))

	macro_aucs_df = pd.DataFrame(columns=columnNames)

	# perform classification on different methods
	pbar = tqdm(methods)
	for method in pbar:
		pbar.set_description(method)
		classify.method = method
		classify.save_plot = True
		fold_aucs, macro_aucs = classify.run()

		fold_aucs_df = fold_aucs_df.append(pd.DataFrame.from_dict(fold_aucs), ignore_index=True)
		macro_aucs_df = macro_aucs_df.append(pd.DataFrame.from_dict(macro_aucs), ignore_index=True)

	# write csv
	tqdm.write("Writing output CSV...")
	if not os.path.exists(os.path.dirname(kfold_output_csv)):
		os.makedirs(os.path.dirname(kfold_output_csv))
	fold_aucs_df.to_csv(kfold_output_csv,index=False)

	if not os.path.exists(os.path.dirname(macro_output_csv)):
		os.makedirs(os.path.dirname(macro_output_csv))
	macro_aucs_df.to_csv(macro_output_csv,index=False)
	tqdm.write("Write output CSV complete")

if __name__ == "__main__":
	main()