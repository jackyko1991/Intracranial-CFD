import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
	csv_list = [
		"Z:/data/intracranial/CFD_results/results_with_probability_dos.csv",
		"Z:/data/intracranial/CFD_results/results_with_probability_selected.csv",
		"Z:/data/intracranial/CFD_results/results_with_probability_all.csv"
	]

	input_types = ['DoS', 'CFD param' , 'CFD param all']
	plot_fmt = ["-","--","-."]
	colors = ["r","g","b"]

	n_inputs = len(csv_list)

	# create plot
	fig, axs = plt.subplots(2,2,figsize=(5, 5))
	plt.suptitle('Probability against DoS')

	for i, _csv in enumerate(csv_list):
		# load csv
		result = pd.read_csv(_csv)

		result = result.sort_values(by=['degree of stenosis(%)'])

		result_X = result[[
			"degree of stenosis(%)",
		]]

		result_Y_columns = [col for col in result.columns if 'probability' in col]
		result_Y = result[result_Y_columns]

		# # colormap
		# cmap = matplotlib.cm.get_cmap('Paired',lut=8)

		# colors = [cmap.hsv(x) for x in np.linspace(0, 1, 10)]

		for j, col in enumerate(result_Y_columns):
			ax = axs.flatten()[j]

			result_Y[col] = result_Y[col].rolling(10).mean()
			label = input_types[i]
			# label = "{} {}".format(input_types[i] , col.split("_")[1])

			# axs.flatten()[j].plot(result_X.to_numpy(), result_Y[col].to_numpy(), plot_fmt[i], label=label, color=cmap(2*i+j))
			ax.set_title(col.split("_")[1])
			ax.plot(result_X.to_numpy(), result_Y[col].to_numpy(), plot_fmt[0], label=label, color=colors[i])

			ax.set_xlabel('degree of stenosis(%)')
			ax.set_ylabel('probability of stroke')
			ax.set_xlim(0,100)
			ax.set_ylim(0,1)
			ax.legend()

	plt.show()

if __name__ == "__main__":
	main()