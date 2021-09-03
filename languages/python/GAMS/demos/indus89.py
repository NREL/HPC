from __future__ import print_function
from gams import *
import os
import sys
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ws = GamsWorkspace(system_directory = sys.argv[1])
    else:
        ws = GamsWorkspace()
 
    ws.gamslib("indus89")    
    t1 = ws.add_job_from_file("indus89.gms")
    
    opt = ws.add_options()
    opt.all_model_types = "gurobi"
       
    with open("indus89.log", "w") as log:
        t1.run(opt, output=log)