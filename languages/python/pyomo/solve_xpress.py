import pyomo.environ as pyo
from p_median import model

xpress = pyo.SolverFactory('xpress_direct')
result = xpress.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median with Xpress")
else:
    print("Something went wrong; see message above")
