# GMClustering
Uses a pipeline of unsupervised methods to partition global magnetospheric data from MMS and THEMIS measurements.

To install, it is strongly recommended that you make a dedicated virtual environment for this package. Strict package requirements are made so that they're the same versions as when the models were first created. If you're using anaconda, then you can make and activate that virtual environment using:

conda create --name gmclustering_env python=3.11

conda activate gmclustering_env

After making the virtual environment, the package can be pip-installed using:

pip install git+https://github.com/jae1018/GMClustering.git

Package functionality, including the bare-minimum needed to get started, is outlined in the gmclustering_example.ipynb file.
