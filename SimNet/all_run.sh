export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/mnt/DIIR-JK-NAS/software/Linux/SimNet_v20.12/SimNet/external/pysdf/build:/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-10.0/bin:$PATH

python all_run.py