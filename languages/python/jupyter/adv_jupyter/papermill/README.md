# Papermill

[Papermill](https://papermill.readthedocs.io/en/latest/) is a tool for running parameterized Jupyter notebooks. It can inject parameter values into a notebook, and create a new notebook with these values. This can be useful for generating notebooks for different conditions without having to modify, run, and save the notebooks manually. 

## Installing
Papermill can be installed using PIP

```
pip install papermill
```

## Usage

### Notebook setup
To modify a notebook to work with papermill the cell which has a parameter in it must be tagged. The tag toolbar can be enabled by View -> Cell Toolbar -> Tags, and then adding `parameters` as a tag to the desired cell. 

The `papermill.ipynb` file is a notebook with the following two cells:


```python
x = 5
y = 4
```


```python
result = x + y
print (result)
```
Either the API or CLI can be used to create new notebooks changing the `x` variable. 

### API
Papermill can be run using another Python script and the API. The below script demonstrates iterating over a range of numbers, and generating a new notebook replacing the specified parameter. 
```python
import papermill as pm

for i in range(0,5):
   pm.execute_notebook(
      'papermill.ipynb',
      f'output_{i}.ipynb',
      parameters=dict(x=i)
   )

```

Papermill shows a progress bar as each of the notebooks is generated and executed. 
```
Executing: 100%|██████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.60cell/s]
Executing: 100%|██████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.10cell/s]
Executing: 100%|██████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.24cell/s]
Executing: 100%|██████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.93cell/s]
Executing: 100%|██████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.62cell/s]
```
### CLI 
The same can be achieved using the Papermill CLI. In the example below the notebook to parameretize, `papermill.ipnyb`, is specified along with the parameter `x` to replace. The output notebook, `cliout.ipynb`, is also specified. 
```
papermill papermill.ipynb cliout.ipynb -p x 110
```

## Limitations
Papermill works best for serially generating notebooks. It does not directly provide a way to parallelize the notebook generation, though this can be achieved with some extra effort. 