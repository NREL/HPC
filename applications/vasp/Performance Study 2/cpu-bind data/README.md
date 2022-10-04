The cpu-bind_VASP.ipynb looks at how tasks bind to cores using various cpu-bind settings on Swift and Eagle. It is independent of the cpu-bind recommendations in the Performance Study 2 documentation, and instead serves only to better understand the direct effect of the cpu-bind flag on the way tasks are processed when running VASP on Swift and Eagle. It uses the data in the eagle_onnodes_data and swift_onnodes_data folders. 

To run the notebook, use the VASP_performance.yml file to create a conda environment with the necessary python dependencies, then open cpu-bind_VASP.ipynb in Jupyter Notebook

```
conda env create -f VASP_performance.yml
conda activate VASP_performance

Jupyter Notebook
```
