import pyomo.environ as pyo
from p_median import model

gams = pyo.SolverFactory('gams')
result = gams.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median with GAMS")
else:
    print("Something went wrong; see message above")
