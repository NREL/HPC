import papermill as pm

digits = [50, 100, 650]
for i in digits:
   pm.execute_notebook(
      'mnist.ipynb',
      f'output_{i}.ipynb',
      parameters=dict(image_index=i)
   )
