import papermill as pm

for i in range(0,5):
   pm.execute_notebook(
      'papermill.ipynb',
      f'output_{i}.ipynb',
      parameters=dict(x=i)
   )
