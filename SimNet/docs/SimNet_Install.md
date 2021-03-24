# SimNet Install (Bare Metal)

1. Create conda virtual environment with Python 3.6
	```bash
	conda create -n simnet python=3.6
	```
2. Switch conda virtual environment
	```bash
	conda activate simnet
	```
3. Tensorflow installation. 
	### CUDA 10
	```bash
	pip install tensorflow-gpu==1.15
	```
	Test for successful installation
	```bash
	python -c "import tensorflow as tf;sess=tf.Session();print(sess.run(tf.reduce_sum(tf.random.normal([1000, 1000]))))"
	```

	For successful Tensorflow install with GPU, you will see message similar to the following:

	```
	...
	2021-03-24 14:58:38.134714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 1 with properties:
	name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
	pciBusID: 0000:0c:00.0
	...
	2021-03-24 14:58:38.134752: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
	2021-03-24 14:58:38.134769: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
	2021-03-24 14:58:38.134785: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
	2021-03-24 14:58:38.134800: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
	2021-03-24 14:58:38.134815: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
	2021-03-24 14:58:38.134830: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
	2021-03-24 14:58:38.134845: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
	...
	-1021.13855

	```

	### CUDA version 11 or above

	Note that if you are using CUDA 11 or above, you should [Nvidia-Tensorflow](https://github.com/NVIDIA/tensorflow). This version is supported by Nvidia but not Google  as official release only up to 1.15 with CUDA 10. You may stick with CUDA 10 version if you are not using Ampere architecture GPUs.

	NVIDIA wheels are not hosted on PyPI.org. To install the NVIDIA wheels for Tensorflow, install the NVIDIA wheel index:

	```bash
	pip install --user nvidia-pyindex
	```
	To install the current NVIDIA Tensorflow release:

	```bash
	pip install --user nvidia-tensorflow[horovod]
	```
	The nvidia-tensorflow package includes CPU and GPU support for Linux.

4. SimNet Download
		
	You should first register the SimNet early access program to login member zone. Download the Tar Archive for bare metal installation

	![SimNet Download](./docs/simnet_tar.jpg)

	Other dependency for SimNet:
	```bash
	pip install matplotlib transforms3d future typing numpy numpy-stl h5py sympy==1.5.1 termcolor psutil symengine numba Cython horovod scipy
	```
	```bash
	pip install -U https://github.com/paulo-herrera/PyEVTK/archive/v1.1.2.tar.gz
	```

	SimNet can be installed from the SimNet source tar ball using python setup
	```bash
	tar -xvzf ./SimNetv0.2_source.tar.gz
	cd ./SimNet/
	python setup.py install
	```

5. STL support with pySDF
	To run examples using the STL point cloud generation you will need to put `libsdf.so` in your library path and install the accompanying PySDF library. This can be done using
	```bash
	export LD_LIBRARY_PATH=<BASE_PATH>/SimNet/external/lib:${LD_LIBRARY_PATH}
	```

	For convenient we export the `$LB_LIBRARY_PATH` by appending follow line to the end of `~/.bashrc`:
	```bashrc
	export LD_LIBRARY_PATH=<BASE_PATH>/SimNet/external/lib:$LD_LIBRARY_PATH
	```
	Then refresh environment variable with 
	```bash
	source ~/.bashrc
	```

	Note that you should change `<BASE_PATH>` to the exact location of SimNet.

	Install pysdf
	```bash
	cd ./SimNet/external/pysdf/
	python setup.py install
	```
6. Test with SimNet examples
	### Helmholtz
	To verify the installation has been done correctly, you can run the following commands:
	```bash
	tar -xvzf ./SimNet_examples.tar.gz
	cd examples/helmholtz/
	python helmholtz.py
	```
	SimNet starts to run with following loss decreasing output:
	```bash
	total_loss: 9999.548
	time: 0.05145785331726074
	total_loss: 934.9323
	time: 0.011391396522521973
	total_loss: 32.646095
	time: 0.011517229080200196
	...
	```

	If you see `./network_checkpoint_hemholtz/` directory created after the execution of t0he command (~5 min), the installation is successful.
	
	### STL Surface
	To verify the installation of SDF library and the STL geometry support, you have to export `libsdf.so` in environment. 
	```bash
	cd examples/aneurysm/
	touch aneurysm.sh
	vim aneurysm.sh
	```
	Edit `aneurysm.sh`
	```bash
	export LD_LIBRARY_PATH=	export LD_LIBRARY_PATH=<BASE_PATH>/SimNet/external/pysdf/build:${LD_LIBRARY_PATH}

	python aneurysm.py
	```

	Execute the bash file:
	```bash
	bash aneurysm.sh
	```
	
	The process take quite a long time to start, by the mean time you may see the following messages:
	```bash
	--------------------------------------------------------------------------
	[[26139,1],0]: A high-performance Open MPI point-to-point messaging module
	was unable to find any relevant network interfaces:

	Module: OpenFabrics (openib)
	Host: diir-2080ti

	Another transport will be used instead, although this may result in
	lower performance.

	NOTE: You can disable this warning by setting the MCA parameter
	btl_base_warn_component_unused to 0.
	--------------------------------------------------------------------------
	```
    If everything goes properly, you will see a gradually decreasing training loss:
    ```bash
    total_loss: 3.4091697
    time: 0.2046236515045166
    total_loss: 2.0797606
    time: 0.20521608591079712
    total_loss: 1.8883789
    time: 0.205885591506958
    ...
    ```