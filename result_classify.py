import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("TkAgg")

from sklearn import decomposition
from sklearn import datasets

def PCA_2D(X,Y):
	pca = decomposition.PCA(n_components=2)
	principalComponents = pca.fit_transform(X)
	principalDf = pd.DataFrame(data = principalComponents
		, columns = ['principal component 1', 'principal component 2'])

	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title('2 component PCA', fontsize = 20)
	targets = ['Normal', 'Stenosis']

	resultDf = pd.DataFrame(data = Y, columns = ['result'])

	finalDf = pd.concat([principalDf, resultDf], axis = 1)
	print(finalDf)

	for i in [0,1]:
		ax.scatter(
			principalComponents[:,0][Y==i],
			principalComponents[:,1][Y==i], 
			s = 20
			)
	ax.legend(targets)
	ax.grid()
	plt.show()

def PCA_3D(X,Y):
	pca = decomposition.PCA(n_components=3)
	pca.fit(X)

	# plot
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	plt.cla()
	X = pca.transform(X)

	iris = datasets.load_iris()
	X_iris = iris.data
	y_iris = iris.target

	for name, label in [('Normal', 0), ('Stenosis', 1)]:
		ax.text3D(
			X[Y == label, 0].mean(),
			X[Y == label, 1].mean() + 1.5, 
			X[Y == label, 2].mean(),
			name,
			horizontalalignment='center',
			bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
	# Reorder the labels to have colors matching the cluster results
	Y = np.choose(Y, [1, 2, 0]).astype(np.float)
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.jet,edgecolor='k')

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])

	plt.show()

def main():
	result_csv = "Z:/data/intracranial/data_30_30/pmv_result.csv"

	result = pd.read_csv(result_csv)
	result_X = result[[
		# "window",
		"Radius_average",
		# "U_average",
		# "p(mmHg)_average",
		# "vorticity_average",
		"Curvature_average",
		# "Torsion_average"
		]]

	result_Y = result[["dataset"]]

	result_X_array = result_X.to_numpy()
	result_Y_array = result_Y.to_numpy()[:,0]

	PCA_2D(result_X_array,result_Y_array)
	# PCA_3D(result_X_array,result_Y_array)

if __name__=="__main__":
	main()