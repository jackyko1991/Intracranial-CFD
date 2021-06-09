import os
import argparse
from VascularSim import *

def ArgParse():
	parser = argparse.ArgumentParser(description="Run OpenFOAM case with user defined domain file")

	parser.add_argument(
		"dir",
		metavar='DIR',
		type=str,
		help="Case directory"
		)
	parser.add_argument(
		"-v",
		"--vtk",
		action="store_true",
		help="Options to output VTK format files (default = false)"
		)
	parser.add_argument(
		"-p",
		"--parallel",
		action="store_true",
		help="Options to use parallel run (default = false)"
		)
	parser.add_argument(
		"-c",
		"--cores",
		metavar="2,4,8,12",
		type=int,
		default=4,
		choices=[2,4,8,12],
		help="Number of cores used for parallel run (defualt = 4)"
		)
	parser.add_argument(
		"-n",
		"--cell_num",
		metavar="INT",
		type=int,
		default=40,
		help="Number of cell blocks in each dimension during blockMesh"
		)

	# args = [
	# 	"/mnt/DIIR-JK-NAS/data/intracranial/data_surgery/217/baseline",
	# 	"-v","-p",
	# 	"-c 8",
	# 	"-n 20"
	# ]

	# return parser.parse_args(args)

	return parser.parse_args()

def main(args):
	run_case(args.dir, output_vtk=args.vtk, parallel=args.parallel, cores=args.cores, cellNumber=args.cell_num)

if __name__ == "__main__":
	args = ArgParse()
	main(args)