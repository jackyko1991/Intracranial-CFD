import pandas as pd
import os
from tqdm import tqdm

def main():
	data_dir = "Z:/data/intracranial/data_30_30"
	output_filename = "pmv_result.csv"

	sub_data_dirs = [
		'stenosis/ESASIS_medical',
		'stenosis/ESASIS_stent',
		'surgery'
		]

	fields = [
		"case",
		"dataset",
		"stage",
		'window',
		'Radius_average',
		'U_average',
		'p(mmHg)_average',
		'vorticity_average',
		'Curvature_average',
		'Torsion_average',
		'Radius_average_dev',
		'U_average_dev',
		'p(mmHg)_average_dev',
		'vorticity_average_dev',
		'Curvature_average_dev',
		'Torsion_average_dev'
	]

	ignore_cases = [
		"ChanSiuYung",
		"ChuFongShu",
		"ChanMeiLing"
	]

	output_pd = pd.DataFrame(columns = fields)

	for sub_data_dir in sub_data_dirs:
		data_list = os.listdir(os.path.join(data_dir,sub_data_dir))
		pbar = tqdm(data_list)

		for case in pbar:
			if not (os.path.exists(os.path.join(data_dir,sub_data_dir,case,"mv_matrix.csv")) and 
				os.path.exists(os.path.join(data_dir,sub_data_dir,case,"mv_df_matrix.csv"))):
				continue

			if case in ignore_cases:
				continue

			input_pd = pd.read_csv(os.path.join(data_dir,sub_data_dir,case,"mv_matrix.csv"))
			input_df_pd = pd.read_csv(os.path.join(data_dir,sub_data_dir,case,"mv_df_matrix.csv"))

			input_pd["case"] = case
			if sub_data_dir == "surgery":
				input_pd["dataset"] = 0
			else:
				input_pd["dataset"] = 1
			# input_pd["dataset"] = sub_data_dir
			input_pd["stage"] = "baseline"

			input_df_pd = input_df_pd.drop(columns=["window"])
			case_pd = pd.concat([input_pd,input_df_pd],axis=1)
			output_pd = pd.concat([output_pd,case_pd])

	output_pd.to_csv(os.path.join(data_dir,output_filename),index=False)


if __name__=="__main__":
	main()