import pandas as pd
from pylatex import Document, Section, Figure, NoEscape, Tabular, basic, Center
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def result_to_score(arr, doc, arr_name="", n_class=5, sort="asc", sigma=2.5, iqr_factor=1.5, outliers_method="z-score"):
	'''
	convert continuous value to ranged score 
	:param str arr: input result array
	:param sec: pylatex document object
	:param str arr_name: array name
	:param str n_class: number of score classes
	:param str sort: "asc" or "dsc", order of the class count with ascending or descending order
	:param float sigma: standard deviation of the data in z-score outliers removal
	:param float iqr_factor: factor for iqr outliers removal
	:param str outliers_method: "z-score" or "iqr"
	:type sigma: float
	:return: None
	:rtype: None
	'''

	# remove outliners
	if outliers_method == "z-score":
		z = np.abs(stats.zscore(arr))
		arr = arr[(z<sigma)]
	elif outliers_method == "iqr":
		Q1 = np.quantile(arr,0.25)
		Q3 = np.quantile(arr,0.75)
		IQR = Q3 - Q1
		arr = arr[~((arr < (Q1 - iqr_factor * IQR)) |(arr > (Q3 + iqr_factor * IQR)))]

	plt.clf()
	n, bins, patches = plt.hist(x=arr, bins=n_class, color='#607c8e',
		alpha=0.7, rwidth=0.9)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	# plt.title(arr_name)
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

	print(n, bins, patches)

	# convert result to score
	score = np.zeros_like(arr)
	if sort == "asc":
		for i in range(n_class):
			if i == 0:
				score[arr<bins[1]] = 1
			elif i == n_class -1:
				score[arr>bins[n_class-1]] = n_class
			else:
				score[~((arr < bins[i]) |(arr > bins[i+1]))] = i+1
	elif sort == "dsc":
		for i in reversed(range(n_class)):
			if i == 0:
				score[arr>bins[n_class-1]] = 1
			elif i == n_class -1:
				score[arr<bins[1]] = n_class
			else:
				score[~((arr < bins[::-1][i+1]) |(arr > bins[::-1][i]))] = i+1

	with doc.create(Figure(position='htbp')) as plot:
		plot.add_plot(dpi=300)
		plot.add_caption(arr_name)

	with doc.create(Center()) as centered:
		with centered.create(Tabular('c|ccccc')) as table:
			table.add_row(['score']+[i + 1 for i in range(n_class)])
			table.add_hline(1, n_class+1)
			table.add_row(['range']+["{:.2f}-{:.2f}".format(bins[i],bins[i+1]) for i in reversed(range(n_class))])
			table.add_row(['count']+[value for value in n])
			# table.add_hline(1, 2)
			# table.add_row((4, 5, 6, 7))

	doc.append(basic.NewPage())

def main():
	result_csv = "Z:/projects/intracranial/results.csv"
	# result_csv = "/Volumes/shared/projects/intracranial/results.csv"
	result = pd.read_csv(result_csv)

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

	# latex option
	geometry_options = {"right": "2cm", "left": "2cm"}
	doc = Document("cfd_result_score", geometry_options=geometry_options)

	# plot distribution
	with doc.create(Section('CFD results distribution plots')):
		for (columnName, columnData) in result_X.iteritems():
			result_to_score(columnData.values,doc,arr_name=columnName,iqr_factor=1.5,outliers_method="iqr",sort="dsc")
			break

	print("Generating pdf...")
	doc.generate_pdf(clean_tex=True)

if __name__ == "__main__":
	main()