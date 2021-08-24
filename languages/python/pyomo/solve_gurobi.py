import pyomo.environ as pyo
from p_median import model

gurobi = pyo.SolverFactory('gurobi_direct')
result = gurobi.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median with Gurobi")
else:
    print("Something went wrong; see message above")
