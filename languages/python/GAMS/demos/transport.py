from __future__ import print_function
from gams import *
import os
import sys

current_dir = os.getcwd()
gams_dir = current_dir + '/gams_files'
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ws = GamsWorkspace(system_directory = sys.argv[1])
    else:
        ws = GamsWorkspace(working_directory = gams_dir)
     
    model = ws.add_job_from_file("transport.gms")
    
    opt = ws.add_options()
    opt.all_model_types = "xpress"
       
    with open("transport.log", "w") as log:
        model.run(opt, output=log)
