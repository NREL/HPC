#!/bin/bash

# Example input to run the models listed in .txt in the .xml input file
# ./submit_multiple.sh 5_bus_system_v2.xml models.txt

############################ USER MODIFIED SECTION ############################
export runtime="00:10:00"
export alloc="hpcapps" #allcoation to use
export runscript="submit_plexos.sh"
###############################################################################

cd /scratch/${USER}/HPC/applications/plexos/RunFiles/

xml_name = "${1%.*}"
model_list_file=$2

echo $xml_name
working_dir=output_${xml_name}

if [ -d "$working_dir" ]; then rm -Rf ${working_dir}; fi
mkdir -p ${working_dir}

while read line
do
    # This line exports environment variables "filename" and "model" to the runscript. 
    # Think of this as passing function arguments.
    sbatch -A ${alloc} -t ${runtime} --export=filename="${xml_name}",model="${line}" ${runscript} --qos=medium   
done
