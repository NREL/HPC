# AlphaFold

[AlphaFold](https://github.com/google-deepmind/alphafold3/tree/main) is an open-source inference pipeline developed by Google DeepMind to predict three-dimensional protein structures from input biological sequence data. The default model from the third major release of AlphaFold (AlphaFold3) is currently only supported on Kestrel.

## Using the AlphaFold module

AlphaFold runs as a [containerized](../Development/Containers/index.md) module on Kestrel and can leverage GPU resources. As such, **you must be on a GPU node to load the module**: `ml alphafold`. Once the module is loaded, run `run_alphafold --help` for the pipeline's help page.

!!! Note
    You need to be part of the `esp-alphafold3` group on Kestrel to access the model and its reference databases in `/kfs2/shared-projects/alphafold3`. To be added to the group, contact [HPC-Help@nrel.gov](mailto:HPC-Help@nrel.gov).

## Input data

AlphaFold requires input data to be formatted in a [special JSON style](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md). Please see the link for the full breadth of available JSON elements, noting that anything specific to "AlphaFold Server" does not apply to the module on Kestrel. A simple input JSON file (`fold_input.json`) is provided below as an example. Refer to the next section for instructions on how to run AlphaFold on JSON files like this one.

??? example "Input JSON format example: `fold_input.json`"
    This is a simple example input JSON file for an AlphaFold run:
    ```json
    {
    "name": "2PV7",
    "sequences": [
        {
        "protein": {
            "id": ["A", "B"],
            "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
        }
        }
    ],
    "modelSeeds": [1],
    "dialect": "alphafold3",
    "version": 1
    }
    ```

## Running AlphaFold on Kestrel

As of February 2025, AlphaFold3 is generally limited to running its inference pipeline on a single JSON on a single GPU device at a time. In other words, multi-GPU and/or multi-node jobs are not inherently supported by the software. As such, running the model sequentially on multiple JSONs requires the least amount of effort on the user's part. However, depending on the number of input JSONs, this limited throughput can lead to prohibitively long jobs in terms of research productivity. For cases in which you need to run AlphaFold on many JSON files representing as many individual sequences, we can instead leverage the "embarrassing parallelism" that Slurm provides with [job arrays](../Slurm/job_arrays.md). 

Example submission scripts representing both strategies are given in the next sections.

!!! Note
    As for every GPU job, please ensure that you submit your AlphaFold jobs from one of the [GPU login nodes](../Systems/Kestrel/index.md).

### Low throughput: Sequentially loop through JSON files

If throughput is not a concern, running AlphaFold on several JSON files is straightforward. Simply load the AlphaFold module, save your JSON files into a folder that is found by the `--input_dir` option, and define an output directory with `--output_dir` in the `run_alphafold` command: 

??? example "Sequentially loop through input JSON files in a folder: `af3_sequential.sh`"
    ```slurm
    #!/bin/bash
    #SBATCH -A hpcapps # Replace with your HPC account
    #SBATCH -t 00:30:00
    #SBATCH -n 32 # note that genetic searches run on the CPUs
    #SBATCH -N 1
    #SBATCH --job-name=af3
    #SBATCH --gres=gpu:1 # Alphafold inference cannot use more than one GPU
    #SBATCH --mem=80G # Note this is for system (i.e., CPU) RAM
    #SBATCH -o %x-%j.log
    
    # Load Alphafold3 module
    ml alphafold
    
    # Run Alphafold3 inference pipeline to predict 3d structure from sequence
    # Note: the "af_input" folder contains "fold_input.json"
    run_alphafold --input_dir af_input --output_dir af_output
    ```

### High throughput: "Embarrassing parallelism" using job arrays

If you need to run AlphaFold on many (e.g., dozens to hundreds) of JSON files, it is worth setting up a job array to run the model on each input in separate jobs from the same submission script. This is a form of "embarrassing parallelism," i.e., a form of parallel computing in which each task is independent of any other and therefore does not require communication between nodes. Thinking of each input JSON as its own "array task" allows us to submit separate jobs that can run simultaneously, which can significantly increase throughput when compared to the `af3_sequential.sh` example above:

??? example "Submit separate jobs for each input JSON file: `af3_job_array.sh`"
    ```slurm
    #!/bin/bash
    #SBATCH -A hpcapps # Replace with your HPC account
    #SBATCH -t 00:30:00
    #SBATCH -n 32 # note that genetic searches run on the CPUs
    #SBATCH -N 1
    #SBATCH --job-name=af3
    #SBATCH --array=0-9 # length of the array corresponds to how many inputs you have (10 in this example)
    #SBATCH --gres=gpu:1 # Alphafold inference cannot use more than one GPU
    #SBATCH --mem=80G # Note this is for system (i.e., CPU) RAM
    #SBATCH -o %A_%a-%x.log
    
    # Load Alphafold3 module
    ml alphafold
    
    # input_json_list.txt is a list of 10 individual JSON files for the run
    IFS=$'\n' read -d '' -r -a input_jsons < input_json_list.txt
    # JSON_INPUT indexes the $input_jsons bash array with the special SLURM_ARRAY_TASK_ID variable
    JSON_INPUT=${input_jsons[$SLURM_ARRAY_TASK_ID]}
    
    # Run Alphafold3 inference pipeline to predict 3d structure from sequence
    # Note the use of --json_path instead of --input_dir to run on a specific JSON
    run_alphafold --json_path $JSON_INPUT --output_dir af_output
    ```

Let's break down some parts of the script to better understand its function:

* `#SBATCH --array=0-9`: This submits an array of 10 individual jobs. The length of this array should match the number of inputs listed in `input_json_list.txt`.
* `#SBATCH -o %A_%a-%x.log`: Send stdout and stderr to a file named after the parent Slurm array job ID (`%A`), the given array task ID (`%a`), and the name of the job (`%x`). For example, a 2-job array with a parent ID of `1000` and a job name of `af3` would create two log files named `1000_0-af3.log` and `1000_1-af3.log`.
* `input_json_list.txt`: A file containing the paths to 10 input JSON files, each on a new line.
* `IFS=$'\n' read -d '' -r -a input_jsons < input_json_list.txt`: Read in `input_json_list.txt` as a bash array named `input_jsons`, using a new line character (`\n`) as the "internal field separator" (`IFS`).
* `JSON_INPUT=${input_jsons[$SLURM_ARRAY_TASK_ID]}`: Index the `input_jsons` bash array with the special `$SLURM_ARRAY_TASK_ID` variable, which will point to a single input file for the given job based on what is specified in `input_json_list.txt`.
* `run_alphafold --json_path $JSON_INPUT --output_dir af_output`: Note that unlike `af3_sequential.sh`, we use the `--json_path` option to point to the specific input JSON that was indexed via `${input_jsons[$SLURM_ARRAY_TASK_ID]}`.

Submitting this script results in 10 individual jobs submitted to Slurm that run independently of each other, rather than sequentially. Since AlphaFold can only run on 1 GPU at a time, you can theoretically fit four simultaneous AlphaFold jobs on a single GPU node.

As with the `af3_sequential.sh` example, each output will be found as a subfolder in the defined `--output_dir` passed to `run_alphafold`.

### Other considerations when running AlphaFold

#### Genetic search stage

AlphaFold3 queries genetic/peptide databases found in `/kfs2/shared-projects/alphafold3` before it runs the structure prediction stage. Sequeunce querying with `jackhmmer` in the AlphaFold3 pipeline can currently only run on CPUs. As such, it benefits you to request up to 32 tasks on the requested node to speed up `jackhmmer` and other CPU-only packages. Since requesting 32 CPUs is 1/4th of the CPUs available on a GPU node, this does not affect how [AUs are charged](../Systems/Kestrel/Running/index.md#allocation-unit-au-charges).

Sequence querying can also consume a significant amount of system RAM. The AlphaFold [documentation](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) states that longer queries can use up to 64G of system (i.e., CPU) memory. As such, you should start with requesting 80G of system RAM via `#SBATCH --mem=80G`. This is below 1/4th of a GPU node's available system RAM and should be plenty for most queries. If you receive an "out of memory" error from Slurm, try increasing `--mem`, but keep in mind that if you exceed 1/4th of the system RAM on the node, you will be charged more AUs for that fraction of the node. Alternatively, you could reduce the number of CPU tasks with `#SBATCH -n` while keeping `#SBATCH --mem=80G` constant to increase the amount of effective RAM per CPU. However, keep in mind that with this approach, sequence querying will likely take longer, which will also increase the number of AUs charged for the job.

#### Visualizing the outputs

Each output subfolder will contain `<job_name>_model.cif`, which is the best predicted structure saved in mmCIF format. Note that you will need to use an external tool such as [PyMol](https://www.pymol.org) to visualize the predicted protein structure.

Refer to the AlphaFold [page](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md) for details regarding all files in the output directory.