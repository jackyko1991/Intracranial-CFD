import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
plt.style.use('ggplot')
import pickle

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
			clf = svm.SVC(kernel='linear',probability=True)
		elif self.method == "SVM_RBF":
			clf = svm.SVC(kernel='rbf',probability=True)
		elif self.method == "LogisticRegression":
			clf = LogisticRegression(random_state=0, multi_class='multinomial')
		elif self.method == "NN" or self.method == "MLP":
			clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32),random_state=1, max_iter=self.mlp_iter)

		clf = OneVsRestClassifier(clf)
		clf.fit(self.X_train, self.y_train)

		return clf

def Find_Optimal_Cutoff(TPR, FPR, threshold=[]):
	y = TPR - FPR
	Youden_index_idx = np.argmax(y)  # Only the first occurrence is returned.
	if len(threshold) == len(TPR):
		optimal_threshold = threshold[Youden_index_idx]
	else:
		optimal_threshold = None
	point = [FPR[Youden_index_idx], TPR[Youden_index_idx]]
	Youden_index = y[Youden_index_idx]
	return optimal_threshold, point, Youden_index

class Classify:
	def __init__(self,
		X,
		y,
		method="LogisticRegression",
		classnames=[],
		show_plot=False,
		save_plot=False,
		save_models=False,
		save_roc=False,
		plot_dir="",
		model_dir="",
		n_folds = 5,
		):

		self.X = X
		self.y = y

		self.method = method
		self.classnames = classnames
		self.show_plot = show_plot
		self.save_plot = save_plot
		self.save_models = save_models
		self.save_roc = save_roc
		self.plot_dir = plot_dir
		self.model_dir = model_dir
		self.n_folds = n_folds
		self.prob = {}

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

		# k fold macro average
		tprs_train = []
		fprs_train = []
		aucs_train = []
		sensitivity_train = []
		specificity_train = []
		precision_train = []
		accuracy_train = []
		f1_train = []

		tprs_test = []
		fprs_test = []
		aucs_test = []
		sensitivity_test = []
		specificity_test = []
		precision_test = []
		accuracy_test = []
		f1_test = []

		mean_fpr = np.linspace(0, 1, 100)

		columnNames = ["Method","Train/Test","Fold"]
		if self.n_classes <3:
			columnNames.append("AUC") 
			columnNames.append("Youden\'s index")
			columnNames.append("Sensitivity")
			columnNames.append("Specificity")
			columnNames.append("Precision")
			columnNames.append("Accuracy")
			columnNames.append("F1 score")
		else:
			for i in range(self.n_classes):
				columnNames.append("AUC {}".format(str(i))) 
				columnNames.append("Youden\'s index {}".format(str(i)))
				columnNames.append("Sensitivity {}".format(str(i)))
				columnNames.append("Specificity {}".format(str(i)))
				columnNames.append("Precision {}".format(str(i)))
				columnNames.append("Accuracy {}".format(str(i)))
				columnNames.append("F1 score {}".format(str(i)))
			
		fold_metrics = {columnName: [] for columnName in columnNames}

		columnNames = ["Method","Train/Test"]
		if self.n_classes <3:
			columnNames.append("AUC") 
			columnNames.append("SD")
			columnNames.append("Youden\'s index")
			columnNames.append("Sensitivity")
			columnNames.append("Specificity")
			columnNames.append("Precision")
			columnNames.append("Accuracy")
			columnNames.append("F1 score")
		else: 
			for i in range(self.n_classes):
				columnNames.append("AUC {}".format(str(i)))
				columnNames.append("SD {}".format(str(i)))
				columnNames.append("Youden\'s index {}".format(str(i)))
				columnNames.append("Sensitivity {}".format(str(i)))
				columnNames.append("Specificity {}".format(str(i)))
				columnNames.append("Precision {}".format(str(i)))
				columnNames.append("Accuracy {}".format(str(i)))
				columnNames.append("F1 score {}".format(str(i)))
		macro_metrics = {columnName: [] for columnName in columnNames}

		#model saving dir
		tqdm.write(self.model_dir)
		if self.save_models and self.model_dir != "":
			os.makedirs(self.model_dir,exist_ok=True)
			tqdm.write(self.model_dir)

		for i, (train_index, test_index) in enumerate(kf.split(self.X)):
			X_train, X_test = self.X[train_index], self.X[test_index]
			y_train, y_test = self.y[train_index], self.y[test_index]

			model.X_train = X_train
			model.X_test = X_test
			model.y_train = y_train
			model.y_test = y_test

			# perform training
			clf = model.train()

			# save the trained model
			if self.save_models:
				pickle.dump(clf, open(os.path.join(self.model_dir,"fold_{}.sav".format(i)), 'wb'))

			# classification report
			report_train = classification_report(y_train,clf.predict(X_train),target_names=self.classnames,output_dict=True)
			report_test = classification_report(y_test,clf.predict(X_test),target_names=self.classnames,output_dict=True)

			# Compute ROC curve and ROC area for each class
			fpr_train = dict()
			tpr_train = dict()
			thresholds_train = dict()
			roc_auc_train = dict()

			fpr_test = dict()
			tpr_test = dict()
			thresholds_test = dict()
			roc_auc_test = dict()

			if self.n_classes > 2:
				for j in range(self.n_classes):
					if False:
					# if "SVM" in self.method or "LogisticRegression" in self.method:
						fpr_train[j], tpr_train[j], thresholds_train[j]  = roc_curve(y_train[:, j], clf.decision_function(X_train)[:,j])
						fpr_test[j], tpr_test[j], thresholds_test[j] = roc_curve(y_test[:, j], clf.decision_function(X_test)[:,j])
					else:
						fpr_train[j], tpr_train[j], thresholds_train[j] = roc_curve(y_train[:, j], clf.predict_proba(X_train)[:,j])
						fpr_test[j], tpr_test[j], thresholds_test[j] = roc_curve(y_test[:, j], clf.predict_proba(X_test)[:,j])

					# auc
					roc_auc_train[j] = auc(fpr_train[j], tpr_train[j])
					roc_auc_test[j] = auc(fpr_test[j], tpr_test[j])

					# youdens index
					optimal_threshold_train, optimal_point_train, youden_idx_train[j] = Find_Optimal_Cutoff(TPR=tpr_train[j],FPR=fpr_train[j],threshold=thresholds_train[j])
					optimal_threshold_test, optimal_point_test, youden_idx_test[j] = Find_Optimal_Cutoff(TPR=tpr_test[j],FPR=fpr_test[j],threshold=thresholds_test[j])

					# plot
					axs[0,j].plot(fpr_train[j],tpr_train[j], label="fold {} (AUC = {:.2f})".format(i, roc_auc_train[j]),alpha=0.3, lw=1)
					#axs[0,j].plot(optimal_point_train[0],optimal_point_train[1], marker='o', color='r')
					#axs[0,j].text(optimal_point_train[0], optimal_point_train[1], f'Threshold:{thresholds_train[j]:.2f}')

					axs[1,j].plot(fpr_test[j],tpr_test[j], label="fold {} (AUC = {:.2f})".format(i, roc_auc_test[j]),alpha=0.3, lw=1)
					#axs[1,j].plot(optimal_point_test[0],optimal_point_test[1], marker='o', color='r')
					#axs[1,j].text(optimal_point_test[0], optimal_point_test[1], f'Threshold:{thresholds_test[j]:.2f}')

					# output probability
					prob = clf.predict_proba(self.X)
			else:
				if False:
				# if "SVM" in self.method or "LogisticRegression" in self.method:
					fpr_train, tpr_train, thresholds_train = roc_curve(y_train, clf.decision_function(X_train))
					fpr_test, tpr_test, threshold_test = roc_curve(y_test, clf.decision_function(X_test))

					prob = clf.decision_function(self.X)
				else:
					if self.n_classes == 1:
						fpr_train, tpr_train, thresholds_train = roc_curve(y_train, clf.predict_proba(X_train))
						fpr_test, tpr_test, thresholds_test = roc_curve(y_test, clf.predict_proba(X_test))

						if self.save_roc:
							roc_cols = ["FPR_0","TPR_0","Thresholds_0"]
							roc_train_df = pd.DataFrame(columns=roc_cols)
							roc_test_df = pd.DataFrame(columns=roc_cols)

							roc_train_df["FPR_0"] = np.array(fpr_train)
							roc_train_df["TPR_0"] = np.array(tpr_train)
							roc_train_df["Thresholds_0"] = np.array(thresholds_train)

							roc_test_df["FPR_0"] = np.array(fpr_test)
							roc_test_df["TPR_0"] = np.array(tpr_test)
							roc_test_df["Thresholds_0"] = np.array(thresholds_test)

							roc_train_df.to_csv(os.path.join(self.plot_dir,"roc_{}_fold{}_{}.csv".format("train",str(i),self.method)),index=False)
							roc_test_df.to_csv(os.path.join(self.plot_dir,"roc_{}_fold{}_{}.csv".format("test",str(i),self.method)),index=False)

						prob = clf.predict_proba(self.X)
					else:
						fpr_train, tpr_train, thresholds_train = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
						fpr_test, tpr_test, thresholds_test = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

						if self.save_roc:
							roc_cols = ["FPR_0","TPR_0","Thresholds_0"]
							roc_train_df = pd.DataFrame(columns=roc_cols)
							roc_test_df = pd.DataFrame(columns=roc_cols)

							roc_train_df["FPR_0"] = np.array(fpr_train)
							roc_train_df["TPR_0"] = np.array(tpr_train)
							roc_train_df["Thresholds_0"] = np.array(thresholds_train)

							roc_test_df["FPR_0"] = np.array(fpr_test)
							roc_test_df["TPR_0"] = np.array(tpr_test)
							roc_test_df["Thresholds_0"] = np.array(thresholds_test)

							roc_train_df.to_csv(os.path.join(self.plot_dir,"roc_{}_fold{}_{}.csv".format("train",str(i),self.method)),index=False)
							roc_test_df.to_csv(os.path.join(self.plot_dir,"roc_{}_fold{}_{}.csv".format("test",str(i),self.method)),index=False)

						prob = clf.predict_proba(self.X)[:,1]

				# auc
				roc_auc_train = auc(fpr_train, tpr_train)
				roc_auc_test = auc(fpr_test, tpr_test)

				# youdens index
				optimal_threshold_train, optimal_point_train, youden_idx_train = Find_Optimal_Cutoff(TPR=tpr_train,FPR=fpr_train,threshold=thresholds_train)
				optimal_threshold_test, optimal_point_test, youden_idx_test = Find_Optimal_Cutoff(TPR=tpr_test,FPR=fpr_test,threshold=thresholds_test)

				# plot 
				axs[0].plot(fpr_train,tpr_train, label="fold {} (AUC = {:.2f})".format(i, roc_auc_train),alpha=0.3, lw=1)
				#axs[0].plot(optimal_point_train[0], optimal_point_train[1], marker='o', color='r')
				#axs[0].text(optimal_point_train[0], optimal_point_train[1], 'Threshold: {:.2f}'.format(optimal_threshold_train))
				axs[1].plot(fpr_test,tpr_test, label="fold {} (AUC = {:.2f})".format(i, roc_auc_test),alpha=0.3, lw=1)
				#axs[1].plot(optimal_point_test[0], optimal_point_test[1], marker='o', color='r')
				#axs[1].text(optimal_point_test[0], optimal_point_test[1], f'Threshold:{optimal_threshold_test:.2f}')

			self.prob["fold_{}".format(i)] = prob
				
			tprs_train.append(tpr_train)
			fprs_train.append(fpr_train)
			aucs_train.append(roc_auc_train)

			tprs_test.append(tpr_test)
			fprs_test.append(fpr_test)
			aucs_test.append(roc_auc_test)

			fold_metrics["Method"].append(self.method)
			fold_metrics["Train/Test"].append("Train")
			fold_metrics["Fold"].append(i + 1)

			if self.n_classes <3:
				fold_metrics["AUC"].append(roc_auc_train)
				fold_metrics["Youden\'s index"].append(youden_idx_train)
				fold_metrics["Sensitivity"].append(optimal_point_train[1])
				fold_metrics["Specificity"].append(1 - optimal_point_train[0])
				fold_metrics["Precision"].append(report_train[self.classnames[1]]["precision"])
				fold_metrics["Accuracy"].append(report_train["accuracy"])
				fold_metrics["F1 score"].append(report_train[self.classnames[1]]["f1-score"])

				sensitivity_train.append(optimal_point_train[1])
				specificity_train.append(1-optimal_point_train[0])
				precision_train.append(report_train[self.classnames[1]]["precision"])
				accuracy_train.append(report_train["accuracy"])
				f1_train.append(report_train[self.classnames[1]]["f1-score"])
			else:
				for j in range(self.n_classes):
					fold_metrics["AUC {}".format(j)].append(roc_auc_train[j])
					fold_metrics["Youden\'s index {}".format(j)].append(youden_idx_train[j])
					# other metrics to be added

			fold_metrics["Method"].append(self.method)
			fold_metrics["Train/Test"].append("Test")
			fold_metrics["Fold"].append(i + 1)
			if self.n_classes <3:
				fold_metrics["AUC"].append(roc_auc_test)
				fold_metrics["Youden\'s index"].append(youden_idx_test)
				fold_metrics["Sensitivity"].append(optimal_point_test[1])
				fold_metrics["Specificity"].append(1 - optimal_point_test[0])
				fold_metrics["Precision"].append(report_test[self.classnames[1]]["precision"])
				fold_metrics["Accuracy"].append(report_test["accuracy"])
				fold_metrics["F1 score"].append(report_test[self.classnames[1]]["f1-score"])

				sensitivity_test.append(optimal_point_test[1])
				specificity_test.append(1-optimal_point_test[0])
				precision_test.append(report_test[self.classnames[1]]["precision"])
				accuracy_test.append(report_test["accuracy"])
				f1_test.append(report_test[self.classnames[1]]["f1-score"])
			else:
				for j in range(self.n_classes):
					fold_metrics["AUC {}".format(j)].append(roc_auc_test[j])
					fold_metrics["Youden\'s index {}".format(j)].append(youden_idx_test[j])

		# average
		mean_fpr = np.linspace(0,1,100)

		if self.n_classes <3:
			# train
			macro_metrics["Method"].append(self.method)
			macro_metrics["Train/Test"].append("Train")
			tprs = []
			for tpr, fpr in zip(tprs_train,fprs_train):
				interp_tpr = np.interp(mean_fpr, fpr, tpr)
				tprs.append(interp_tpr)
			mean_tpr = np.mean(tprs,axis=0)
			mean_tpr[-1] = 1.0
			mean_auc = metrics.auc(mean_fpr, mean_tpr)
			aucs = [auc for auc in aucs_train]
			std_auc = np.std(aucs)
			# youden's index
			_, optimal_point_train, youden_idx_train = Find_Optimal_Cutoff(TPR=mean_tpr,FPR=mean_fpr)

			axs[0].plot(mean_fpr, mean_tpr, color='b',
				label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
				lw=2, alpha=.8)
			axs[0].plot(optimal_point_train[0], optimal_point_train[1], marker='o', color='r')
			axs[0].text(optimal_point_train[0]+0.05, optimal_point_train[1]-0.05, 'J: {:.2f}'.format(youden_idx_train))

			std_tpr = np.std(tprs, axis=0)
			tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
			tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
			axs[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				label=r'$\pm$ 1 std. dev.')

			if self.save_roc:
				roc_cols = ["FPR_0","TPR_0","SD_Upper_0","SD_Lower_0"]
				roc_train_df = pd.DataFrame(columns=roc_cols)
				roc_train_df["FPR_0"] = np.array(mean_tpr)
				roc_train_df["TPR_0"] = np.array(mean_fpr)
				roc_train_df["SD_Upper_0"] = np.array(tprs_upper)
				roc_train_df["SD_Lower_0"] = np.array(tprs_lower)
				roc_train_df.to_csv(os.path.join(self.plot_dir,"roc_{}_mean_{}.csv".format("train",self.method)),index=False)

			macro_metrics["AUC"].append(mean_auc)
			macro_metrics["SD"].append(std_auc)
			macro_metrics["Youden\'s index"].append(youden_idx_train)
			macro_metrics["Sensitivity"].append(np.mean(sensitivity_train))
			macro_metrics["Specificity"].append(np.mean(specificity_train))
			macro_metrics["Precision"].append(np.mean(precision_train))
			macro_metrics["Accuracy"].append(np.mean(accuracy_train))
			macro_metrics["F1 score"].append(np.mean(f1_train))

			# test
			macro_metrics["Method"].append(self.method)
			macro_metrics["Train/Test"].append("Test")
			tprs = []
			for tpr, fpr in zip(tprs_test,fprs_test):
				interp_tpr = np.interp(mean_fpr, fpr, tpr)
				tprs.append(interp_tpr)
			mean_tpr = np.mean(tprs,axis=0)
			mean_tpr[-1] = 1.0

			mean_auc = metrics.auc(mean_fpr, mean_tpr)
			aucs = [auc for auc in aucs_test]
			std_auc = np.std(aucs)
			# youden's index
			_, optimal_point_test, youden_idx_test = Find_Optimal_Cutoff(TPR=mean_tpr,FPR=mean_fpr)
			axs[1].plot(mean_fpr, mean_tpr, color='b',
				label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
				lw=2, alpha=.8)
			axs[1].plot(optimal_point_test[0], optimal_point_test[1], marker='o', color='r')
			axs[1].text(optimal_point_test[0]+0.05, optimal_point_test[1]-0.05, 'J: {:.2f}'.format(youden_idx_test))

			std_tpr = np.std(tprs, axis=0)
			tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
			tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
			axs[1].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				label=r'$\pm$ 1 std. dev.')

			if self.save_roc:
				roc_cols = ["FPR_0","TPR_0","SD_Upper_0","SD_Lower_0"]
				roc_test_df = pd.DataFrame(columns=roc_cols)
				roc_test_df["FPR_0"] = np.array(mean_tpr)
				roc_test_df["TPR_0"] = np.array(mean_fpr)
				roc_test_df["SD_Upper_0"] = np.array(tprs_upper)
				roc_test_df["SD_Lower_0"] = np.array(tprs_lower)
				roc_test_df.to_csv(os.path.join(self.plot_dir,"roc_{}_mean_{}.csv".format("test",self.method)),index=False)

			macro_metrics["AUC"].append(mean_auc)
			macro_metrics["SD"].append(std_auc)
			macro_metrics["Youden\'s index"].append(youden_idx_test)
			macro_metrics["Sensitivity"].append(np.mean(sensitivity_test))
			macro_metrics["Specificity"].append(np.mean(specificity_test))
			macro_metrics["Precision"].append(np.mean(precision_test))
			macro_metrics["Accuracy"].append(np.mean(accuracy_test))
			macro_metrics["F1 score"].append(np.mean(f1_test))
		else:
			# train
			macro_metrics["Method"].append(self.method)
			macro_metrics["Train/Test"].append("Train")
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
				# youden's index
				_, optimal_point_train, youden_idx_train = Find_Optimal_Cutoff(TPR=mean_tpr,FPR=mean_fpr)

				axs[0,i].plot(mean_fpr, mean_tpr, color='b',
					label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
					lw=2, alpha=.8)
				axs[0,i].plot(optimal_point_train[0], optimal_point_train[1], marker='o', color='r')
				axs[0,i].text(optimal_point_train[0]+0.05, optimal_point_train[1]-0.05, 'J: {:.2f}'.format(youden_idx_train))

				std_tpr = np.std(tprs, axis=0)
				tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
				tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
				axs[0,i].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					label=r'$\pm$ 1 std. dev.')

				macro_metrics["AUC {}".format(str(i))].append(mean_auc)
				macro_metrics["SD {}".format(str(i))].append(std_auc)
				macro_metrics["Youden\'s index {}".format(str(i))].append(youden_idx_train)

			# test
			macro_metrics["Method"].append(self.method)
			macro_metrics["Train/Test"].append("Test")

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
				# youden's index
				_, optimal_point_test, youden_idx_test = Find_Optimal_Cutoff(TPR=mean_tpr,FPR=mean_fpr)
				axs[1,i].plot(mean_fpr, mean_tpr, color='b',
					label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
					lw=2, alpha=.8)
				axs[1,i].plot(optimal_point_test[0], optimal_point_test[1], marker='o', color='r')
				axs[1,i].text(optimal_point_teset[0]+0.05, optimal_point_test[1]-0.05, 'J: {:.2f}'.format(youden_idx_test))

				std_tpr = np.std(tprs, axis=0)
				tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
				tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
				axs[1,i].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					label=r'$\pm$ 1 std. dev.')

				macro_metrics["AUC {}".format(str(i))].append(mean_auc)
				macro_metrics["SD {}".format(str(i))].append(std_auc)
				macro_metrics["Youden\'s index {}".format(str(i))].append(youden_idx_test)

		for i, ax in enumerate(axs.flatten()):
			ax.legend(loc="lower right")

		if self.save_plot and (not self.plot_dir == ""):
			os.makedirs(self.plot_dir,exist_ok=True)
			tqdm.write(("Saving plot..."))
			plt.savefig(os.path.join(self.plot_dir,"roc_{}.png".format(self.method)))

		if self.show_plot:
			plt.show()

		prob_avg = np.zeros_like(self.prob["fold_0"])
		for key, value in self.prob.items():
			prob_avg += value
		self.prob["fold_avg"] = prob_avg/self.n_folds

		return fold_metrics, macro_metrics

def main():
	suffix = "dos"

	# result_csv = "Z:/data/intracranial/CFD_results/results_stenosis_no_neg_pressure.csv"
	# plot_output_dir = "Z:/data/intracranial/CFD_results/plots/results_{}".format(suffix)
	# model_output_dir = "Z:/data/intracranial/CFD_results/models/{}".format(suffix)
	# kfold_output_csv = "Z:/data/intracranial/CFD_results/metrics/metrics_{}_kfold.csv".format(suffix)
	# macro_output_csv = "Z:/data/intracranial/CFD_results/metrics/metrics_{}_macro.csv".format(suffix)
	# probability_output_csv = "Z:/data/intracranial/CFD_results/results_with_probability_{}.csv".format(suffix)

	result_csv = "Y:/data/intracranial/CFD_results/results_stenosis_no_neg_pressure.csv"
	plot_output_dir = "Y:/data/intracranial/CFD_results/plots/results_{}".format(suffix)
	model_output_dir = "Y:/data/intracranial/CFD_results/models/{}".format(suffix)
	kfold_output_csv = "Y:/data/intracranial/CFD_results/metrics/metrics_{}_kfold.csv".format(suffix)
	macro_output_csv = "Y:/data/intracranial/CFD_results/metrics/metrics_{}_macro.csv".format(suffix)
	probability_output_csv = "Y:/data/intracranial/CFD_results/results_with_probability_{}.csv".format(suffix)
	result = pd.read_csv(result_csv)

	show_plot = False
	save_plot = True
	save_models = True
	save_roc = True
	mlp_iter = 200

	methods = [
		"RandomForest",
		"LogisticRegression",
		# "SVM_Linear",
		"SVM_RBF",
		"MLP"
	]

	result_X = result[[
		# "radius mean(mm)",
		"degree of stenosis(%)",
		#"radius min(mm)",
		# "pressure mean(mmHg)",
		# "max radius gradient",
		#"max pressure gradient(mmHg)",
		#"in/out pressure gradient(mmHg)",
		# "velocity mean(ms^-1)",
		#"peak velocity(ms^-1)",
		#"max velocity gradient(ms^-1)",
		# "vorticity mean(s^-1)",	
		#"peak vorticity(s^-1)",
		#"translesion peak presssure(mmHg)",
		#"translesion presssure ratio",	
		#"translesion peak pressure gradient(mmHgmm^-1)",	
		##"translesion peak velocity(ms^-1)",	
		#"translesion velocity ratio",	
		##"translesion peak velocity gradient(ms^-1mm^-1)",
		##"translesion peak vorticity(ms^-1)",
		# "translesion vorticity ratio",	
		#"translesion peak vorticity gradient(Pamm^-1)",
		#"translesion peak wss(Pa)",	
		#"translesion peak wss gradient(Pamm^-1)"
		]]

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

	# # select by human
	# result_X = result[[
	# 	"degree of stenosis(%)",
	# 	"in/out pressure gradient(mmHg)",
	# 	"peak velocity(ms^-1)",
	# 	"peak vorticity(s^-1)"
	# 	]]

	# result_X = result[[
	# 	"degree of stenosis(%)",
	# 	]]

	result_Y = result[["stroke","type","stenosis"]]

	result_X_array = result_X.to_numpy()
	# severity
	# classes = [0,1,2]
	# classnames = ["normal","moderate","severe"]
	classes = [0,1]
	classnames = ["normal","stroke"]
	# classnames = ["normal","ICAD"]

	# stroke: [:,0], severity: [:,1], "icad": [:,2]
	result_Y_array = label_binarize(result_Y.to_numpy()[:,0],classes=classes)
	# result_Y_array = label_binarize(result_Y.to_numpy()[:,1],classes=classes)
	# result_Y_array = label_binarize(result_Y.to_numpy()[:,2],classes=classes)

	classify = Classify(result_X_array,result_Y_array,classnames=classnames)
	classify.mlp_iter = mlp_iter
	classify.plot_dir = plot_output_dir
	classify.show_plot = show_plot
	classify.save_models = save_models
	classify.save_plot = save_plot
	classify.save_roc = save_roc

	# output df
	columnNames = ["Method","Train/Test","Fold"]
	if len(classes) <3:
		columnNames.append("AUC") 
	else:
		for i in range(len(classes)):
			columnNames.append("AUC {}".format(str(i))) 

	fold_metrics_df = pd.DataFrame(columns=columnNames)

	columnNames = ["Method","Train/Test"]
	if len(classes) <3:
		columnNames.append("AUC")
		columnNames.append("SD")
	else:
		for i in range(len(classes)):
			columnNames.append("AUC {}".format(str(i)))
			columnNames.append("SD {}".format(str(i)))

	macro_metrics_df = pd.DataFrame(columns=columnNames)

	# perform classification on different methods
	pbar = tqdm(methods)
	for method in pbar:
		pbar.set_description(method)
		classify.method = method
		classify.model_dir = os.path.join(model_output_dir,method)
		fold_metrics, macro_metrics = classify.run()

		if len(classes) < 3:
			result['probability_{}'.format(method)] = classify.prob['fold_avg']
		else:
			exit("probability output not ready for class number > 3")

		fold_metrics_df = fold_metrics_df.append(pd.DataFrame.from_dict(fold_metrics), ignore_index=True)
		macro_metrics_df = macro_metrics_df.append(pd.DataFrame.from_dict(macro_metrics), ignore_index=True)

	# write csv
	tqdm.write("Writing output CSV...")
	if not os.path.exists(os.path.dirname(kfold_output_csv)):
		os.makedirs(os.path.dirname(kfold_output_csv))
	fold_metrics_df.to_csv(kfold_output_csv,index=False)

	if not os.path.exists(os.path.dirname(macro_output_csv)):
		os.makedirs(os.path.dirname(macro_output_csv))
	macro_metrics_df.to_csv(macro_output_csv,index=False)

	if not os.path.exists(os.path.dirname(probability_output_csv)):
		os.makedirs(os.path.dirname(probability_output_csv))
	result.to_csv(probability_output_csv,index=False)

	tqdm.write("Write output CSV complete")

if __name__ == "__main__":
	main()