import pyomo.environ as pyo
from p_median import model

ipopt = pyo.SolverFactory('ipopt')
result = ipopt.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median *relaxation* with IPOPT")
else:
    print("Something went wrong; see message above")
