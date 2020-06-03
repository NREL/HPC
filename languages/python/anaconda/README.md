# Anaconda

1. The notebook used to create the slides on creating and maintaining Conda environments can be found [here](conda_tutorial.ipynb).
2. The slides themselves don't render on GitHub, so if you'd like to interact with the content that way, download the .html file and open it in your browser.  The .html file generated from the notebook can be found [here](conda_tutorial.slides.html)

## Managing Python 
Conda enables creation of virtual environments with specific versions of Python and isolation of packages installed via pip.  

Create a virtual environment with Python 3.7 on Eagle:

```
conda create --name py37 python=3.7
```

Once the environment has been created you need to activated it with Conda: 

```
conda activate py37
```

**Note**: You will need to either activate the environment each time you login to Eagle or add it to your Bash profile. 

Once active you can install packages as necessary
```
pip install numpy
```
