# Julia

*Julia is a dynamic programming language that offers high performance while being easy to learn and develop code in.*

## Installing Julia on Eagle

Julia modules exist on Eagle and provide an easy way to use Julia on the HPC system. Access simply with

```bash
module load julia
```

To see all available Julia modules, use the command

```bash
module spider julia
```

If you need a version of Julia for which a module does not exist or want your own personal Julia build, there are several options described in the rest of this document. Below is a general guide for what approach to use:

* fast and easy - Anaconda
* perfomance and ease - Spack
* performance or need to customize Julia build - do it yourself (i.e. build from source)

### Anaconda

Older versions of Julia are available from conda-forge channel

```bash
conda create -n julia-env
source activate julia-env
conda install -c conda-forge julia
```

### Spack Build

#### Pre-requites

A working version of Spack. For detailed instructions on getting spack setup see the github repository. Briefly, this can be done with the following

```bash
git clone https://github.com/spack/spack.git
cd spack
git checkout releases/v0.15 # Change to desired release
. share/spack/setup-env.sh # Activate spack shell support
```

#### Instructions

**NOTE:** Steps 1 and 2 may be skipped when using the develop branch or any release branch after v0.15.

1. In the spack repository, open the file var/spack/repos/builtin/packages/julia/package.py in your favorite editor.
2. There is an if-else statement under the if statement
    ```python
    if spec.target.family == 'x86_64'  or spec.target.family == 'x86':
    ```
    Change the else clause to read
    ```python
    else:
        target_str = str(spec.target).replace('_','-')
        options += [
            'MARCH={0}'.format(target_str),
            'JULIA_CPU_TARGET={0}'.format(target_str)
        ]
    ```
3. Now install julia with spack
    ```bash
    spack install julia
    ```

### Do It Yourself Build (v 1.2 or later)

#### Pre-requites

All the required build tools and libraries are available on Eagle either by default or through modules. The needed modules are covered in the instructions.
