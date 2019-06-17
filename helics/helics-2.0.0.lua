help([[****helics cosimulation software]])

whatis("Name: helics")
whatis("Version: 2.0.0")

local datestamp = "2019-04-30"
local base = "/projects/aces/helics_2.0.0_mpi"

load("boost/1.69.0/gcc-7.3.0")
load("gcc/7.3.0")
load("openmpi/3.1.3/gcc-7.3.0")

prepend_path("CPATH", pathJoin(base, "include"))
prepend_path("CMAKE_PREFIX_PATH", base)
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib64"))
prepend_path("LIBRARY_PATH", pathJoin(base, "lib64"))
prepend_path("LD_LIBRARY_PATH",pathJoin(base,"lib"))
prepend_path("LIBRARY_PATH",pathJoin(base,"lib"))
prepend_path("PATH",pathJoin(base,"bin"))
prepend_path("PYTHONPATH",pathJoin(base,"python"))