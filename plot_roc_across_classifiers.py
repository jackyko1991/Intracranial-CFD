import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import colors

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

def main():
	data_dir = "Z:/data/intracranial/CFD_results/baseline/plots"

	inputs = {"dos": "DoS","cfd":"CFD","dos_cfd":"Combined"}
	phases = {"train":"Training","test":"Validation"}
	classifiers = {"LogisticRegression": "Logistic Regression","RandomForest": "Random Forest","SVM_RBF": "Support Vector Machine","MLP":"Multilayer Perceptron"}

	# create plot figure
	fig, axs = plt.subplots(2,3,figsize=(15, 8))

	for i in range(2):
		for j in range(3):
			axs[i,j].plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.6)
			axs[i,j].set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate')
			axs[i,j].plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.6)
			axs[i,j].set(xlim=[-0.00, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate')

	for i, (phase_key, phase_value) in enumerate(phases.items()):
		for j, (input_key, input_value) in enumerate(inputs.items()):
			for k, (classifier_key, classifier_value) in enumerate(classifiers.items()):
				# load the csv file
				df = pd.read_csv(os.path.join(data_dir,"results_{}".format(input_key),"roc_{}_mean_{}.csv".format(phase_key,classifier_key)))
				p = axs[i,j].plot(df["FPR_0"],df["TPR_0"],label=classifier_value)
				#axs[i,j].fill_between(df["FPR_0"], df["SD_Upper_0"], df["SD_Lower_0"], color=p[0].get_color(), alpha=.2,label=r'$\pm$ 1 std. dev.')
				axs[i,j].fill_between(df["FPR_0"], df["SD_Lower_0"], df["SD_Upper_0"], color=p[0].get_color(), alpha=.2)

			if i == 0:
				axs[i,j].set_title(input_value)

			if j == 0:
				t = axs[i,j].text(-0.25,0.5,phase_value, size=14,verticalalignment='center', rotation=90)

	for i, ax in enumerate(axs.flatten()):
		ax.legend(loc="lower right")
	#plt.show()
	plt.savefig(os.path.join(data_dir,"roc_overall.png"))

if __name__ == "__main__":
	main()