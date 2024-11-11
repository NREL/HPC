# Compiler Information

*This document describes some of the important command line options for various compilers.  This includes gcc, gfortran, g++, Intel, Fortran, C, C++, as well as the Cray compilers. The infomation contained herein is not complete but only a small subset of what is available in man pages and full documentation.  For example, the man page for gcc is over 21,000 lines long.*


## Topics

The topics covered include:

* Normal invocation
* Default optimization level
* Compiling for performance
* Compiling for debugging and related purposes
* Runtime checks
* Some File extensions
* Language standard settings (Dialect)
* Generating listing, if available
* Preprocessing
* OpenMP support
* OpenACC support
* UPC support (C++)
* Coarray support (Fortran)
* Important compiler specific options
* Notes

## Compilers covered

* gcc
* gfortran
* Intel icc (Classic)
	* Moving to Intel's new icx compiler
* Intel ifort (Fortran)
	* Moving to Intel's new ifx compiler
* Cray C (Clang based)
* Cray Fortran (ftn)



## gcc/g++

This discussion is for version 12.x. Most options are supported for recent versions of the compilers.  Also, most command line options for gcc and g++ are supported for each compiler.  It is recommended that C++ programs be compiled with g++ and C programs with gcc.

#### Normal invocation
```
# Compile and link a program with the executable sent to the indicated
  file
gcc mycode.c -o myexec
g++ mycode.C -o myexec

# Compile a file but don't link 
gcc -c mycode.c 
g++ -c mycode.C 

```
#### Default optimization
The default optimization level is -O0 on most systems.  It is possible that a compiler might be configured to have a different default.  One easy way to determine the default is to build a simple application without specifying an optimization level and compare its size to a version compiled with optimization on.  

#### Compiling for performance

```
-O1 Optimize.  Optimizing compilation takes somewhat more time, and a
	lot more memory for a large function.

-O2 Optimize even more.  GCC performs nearly all supported
	optimizations that do not involve a space-speed tradeoff.

-O3 Optimize yet more.

-Ofast Disregard strict standards compliance.  -Ofast enables all -O3
	optimizations.  It also enables optimizations that are not valid
	for all standard-compliant programs.  

```           
                   
You can discover which optimizations are at various levels of optimization as shown below.  The last command will show all potential optimization flags, over 250.

```
gcc -c -Q -O3 --help=optimizers > /tmp/O3-opts
gcc -c -Q -O2 --help=optimizers > /tmp/O2-opts
diff /tmp/O2-opts /tmp/O3-opts | grep enabled

gcc -Q  --help=optimizers 
```

#### Compiling for debugging and related purposes
```
-Og Optimize debugging experience. Use instead of -O0. Does sopme
	optimization but maintains debug information

-g  Produce debugging information

gcc -Og -g myprog.c

-p,-pg Generate extra code to write profile information suitable for
	the analysis program prof (for -p) or gprof
```
There are many potential options  for profiling.  See the man page and search for -pg. 

#### Some file extensions

```
file.c
   C source code that must be preprocessed.

file.i
   C source code that should not be preprocessed.

file.ii
   C++ source code that should not be preprocessed.

file.cc
file.cp
file.cxx
file.cpp
file.CPP
file.c++
file.C
   C++ source code that must be preprocessed.  
```

You can specify explicitly the language for file indepenent of the extension using the -x option.  For example **gcc -x c file.cc** will complie the program as C instead of C++.

#### Language standard settings (Dialect)

```
-ansi This is equivalent to -std=c90. In C++ mode, it is equivalent to -std=c++98.


-std=

c90
   Support all ISO C90 programs 

iso9899:199409
   ISO C90 as modified in amendment 1.

c99
   ISO C99.  

c11
   ISO C11, the 2011 revision of the ISO C standard.  

c18
   ISO C17, the 2017 revision of the ISO C standard
   (published in 2018).  

c2x The next version of the ISO C standard, still under
    development.  The support for this version is
    experimental and incomplete.


c++98 The 1998 ISO C++ standard plus the 2003 technical
      corrigendum and some additional defect reports. Same as
      -ansi for C++ code.

c++11
   The 2011 ISO C++ standard plus amendments.  
   
c++14
   The 2014 ISO C++ standard plus amendments.  
   
c++17
   The 2017 ISO C++ standard plus amendments. 
 
```
    
This is a subset of all of the options.  There are "gnu" specific versions of many of these which give slight variations.  Also, some fo these can be specified  in various deprecated flags. The dialects available for the compilers are highly version dependent.  Older versions of compiler will not support newer dialects.

#### Preprocessing

Unless explicitly disabled by the file extension as described above files are preprocessed.  If you pass the **-E** option the file will be preprocessed only and will not be compiled.  The output is sent to the standard output 

#### OpenMP support
```
-fopenmp 
          Enable handling of OpenMP directives
-fopenmp-simd
          Enable handling of OpenMP's SIMD directives   
-mgomp          
          Generate code for use in OpenMP offloading 
```

Offlading will not work on all platforms and may require additional options.
      
#### OpenACC support

```
 -fopenacc
           Enable handling of OpenACC directives
           
 -fopenacc-dim=geom
           Specify default compute dimensions for parallel offload
           regions that do not explicitly specify
 ```

Offlading will not work on all platforms and may require additional options.          
           
```

#### Important compiler specific options


-Wall
     This enables all the warnings about constructions that some
     users consider questionable, and that are easy to avoid (or
     modify to prevent the warning)
           
-Wextra 
     This enables some extra warning flags that are not enabled by
     -Wall.
     


## gfortran

This discussion is for version 12.x. Most options are supported for recent versions of the compilers.  Also, most command line options for gcc and g++ are supported for gfortran.

#### Normal invocation
```
# Compile and link a program with the executable sent to the indicated
  file
gfortran mycode.f90  -o myexec

# Compile a file but don't link 
gfortran -c mycode.f90

```
#### Default optimization
The default optimization level is -O0 on most systems.  It is possible that a compiler might be configured to have a different default.  One easy way to determine the default is to build a simple application without specifying an optimization level and compare its size to a version compiled with optimization on.  

#### Compiling for performance

```
-O1 Optimize.  Optimizing compilation takes somewhat more time, and a
	lot more memory for a large function.

-O2 Optimize even more.  GCC performs nearly all supported
	optimizations that do not involve a space-speed tradeoff.

-O3 Optimize yet more.

-Ofast Disregard strict standards compliance.  -Ofast enables all -O3
	optimizations.  It also enables optimizations that are not valid
	for all standard-compliant programs.  

```           
                   
You can discover which optimizations are at various levels of optimization as shown below.  The last command will show all potential optimization flags, over 250.

```
gfortran -c -Q -O3 --help=optimizers > /tmp/O3-opts
gfortran -c -Q -O2 --help=optimizers > /tmp/O2-opts
diff /tmp/O2-opts /tmp/O3-opts | grep enabled

gfortran -Q  --help=optimizers 
```

#### Compiling for debugging and related purposes
```
-Og Optimize debugging experience. Use instead of -O0. Does sopme
	optimization but maintains debug information

-g  Produce debugging information

-fbacktrace Try to print a back trace on error

-fcheck=<all|array-temps|bits|bounds|do|mem|pointer|recursion>.
	Perform various runtime checks.  This will slow your program
	down.

gfortran -Og -g -fbacktrace -fcheck=all myprog.c

-fcheck=<all|array-temps|bits|bounds|do|mem|pointer|recursion>
	Perform various runtime checks

-p,-pg Generate extra code to write profile information suitable for
	the analysis program prof (for -p) or gprof


```
There are many potential options  for profiling.  See the man page and search for -pg. 

#### Some file extensions

```
.F, .FOR, .FTN, .fpp, .FPP, .F90, .F95, .F03
    preprocessor is run automatically   

.f, .for, .ftn, .f90, .f95, .f03
    preprocessor is not run automatically   
   
```



#### Language standard settings (Dialect)

```

f95, f2003, f2008, f2018 Specify strict conformance to the various
	standards

gnu 2018 with gnu extensions

legacy Older codes

-ffree-form / -ffixed-form The source is in Free / Fixed form
           
```

#### Language standard settings (Save)

The Fortran 90 standard does not indicate the status of variables that leave scope.  That is in general, a variable defined in a subroutine may or may not be defined when the subroutine is reentered.  There are exceptions for variables in common blocks and those defined in modules.

For Fortran 95 and later local allocatable variables are automatically deallocated upon exit from a subroutine.

The flags -fautomatic  and -fno-automatic change this behavior.

```

-fautomatic Automatically deallocate variables on exit independent of
	standard setting

-fno-automatic Do not automatically deallocate variables on exit
	independent of standard setting

-fmax-stack-var-size With this value set to some small value, say 1
	it appears that variables are not deallocated.  A program
	compiled with this option would in general be nonconformnet.
           
```

The above applies to allocatable arrays.  It is not clean what happens to scalers.

#### Language standard settings (argument mismatch)

Some code contains calls to external procedures with mismatches
between the calls and the procedure definition, or with
mismatches between different calls. Such code is non-conforming,
and will usually be flagged with an error.  This options degrades
the error to a warning, which can only be disabled by disabling
all warnings via -w.  Only a single occurrence per argument is
flagged by this warning. -fallow-argument-mismatch is implied by
-std=legacy.



It is recomended that source code be modified to have interfaces for routines that are called iwth various types of arguments.  Fortran 2018 allows for a generic type for such interfaces.  For example here is an interface for MPI_Bcast 

```
module bcast
interface
 subroutine MPI_BCAST(BUF, COUNT, DATATYPE, DEST, COMM, IERROR)
 type(*),intent(inout) :: BUF
 !type(*), dimension(..), intent(in) :: BUF
 integer, intent(in) ::  COUNT, DATATYPE, DEST,  COMM
 integer, intent(out) :: IERROR
 end subroutine
end interface
end module

```

       
    
#### Generating listing

Gfortran does not produce listings.


#### Preprocessing

Automatic preprocessing is determined by the file name extension as discussed above. You can manually turn it on/off via the options 
 
```
-cpp - Preprocess
-nocpp - Don't preprocess
-cpp -E - Preprocess and send output to standard out. Don't compile
```

#### OpenMP support
```
-fopenmp        Enable handling of OpenMP directives
-fopenmp-simd   Enable handling of OpenMP's SIMD directives   
-mgomp          Generate code for use in OpenMP offloading 
```

Offlading will not work on all platforms and may require additional options.
      
#### OpenACC support

```
 -fopenacc Enable handling of OpenACC directives
           
 -fopenacc-dim=geom Specify default compute dimensions for parallel offload
     regions that do not explicitly specify


```
Offlading will not work on all platforms and may require additional options.          
           
#### Important compiler specific options

```

-fimplicit-none 
            Produce and error message if there are explicitly typed variables.  

-fdefault-real-8
            Set the default real type to an 8 byte wide type.  This option also affects the kind of non-double real constants like 1.0. 

-pedantic 
            Issue warnings for uses of extensions to Fortran.

       -fall-intrinsics
           This option causes all intrinsic procedures (including the GNU-specific extensions) to be accepted.  This can
           be useful with -std= to force standard-compliance but get access to the full range of intrinsics available
           with gfortran.  
      

```           



## icc/icpc

This discussion is for version 2021.6.0.  Icc and icpc will be replaced with clang based alternatives in the near future, icx and icpx.  In the Cray environment if PrgEnv-intel is loaded the "cc" maps to icc.


#### Normal invocation

```

Compile and link a program with the executable sent to the indicated
  file
icc mycode.c -o myexec
icpc mycode.C -o myexec

Compile a file but don't link 
icc -c mycode.c 
icpc -c mycode.C 

```

NOTE: The icpc command uses the same compiler options as the icc command. Invoking the compiler using icpc compiles .c and .i files as C++. Invoking the compiler using icc compiles .c and .i files as C. Using icpc always  links in C++ libraries. Using icc only links in C++ libraries if C++ source is provided on the command line.
       
       
#### Default optimization
The default optimization level is -O2.  

#### Compiling for performance

```
-O0  Disables all optimizations.

-O1  Enables optimizations for speed.

-O2 Optimize even more. 

-O  Same ans -O2

-O3 Optimize yet more.

-Ofast -O3, -no-prec-div, and -fp-model

-no-prec-div  enables optimizations that give slightly less precise
	results than full IEEE division

-fp-model slight decrease in the accuracy of math library functions

-opt_report  Generate and optimization report
```           
                   
You can learn more about optimizations are at various levels of optimization as shown below.  

```
icc -V -help opt
```

#### Compiling for debugging and related purposes
```
-g[n] 
    0 Disables generation of symbolic debug information.
    1 Produces minimal debug information for performing stack traces.
    2 Produces complete debug information. This is the same as specifying -g with no n.
    3 Produces extra information that may be useful for some tools.

-Os Generate extra code to write profile information suitable for
    the analysis program gprof
```

#### Some file extensions

```
file.c
   C source code that must be preprocessed.

file.i
   C source code that should not be preprocessed.

file.ii
   C++ source code that should not be preprocessed.

file.cc
file.cp
file.cxx
file.cpp
file.CPP
file.c++
file.C
   C++ source code that must be preprocessed.  
```

You can specify explicitly the language for file indepenent of the extension using the -x option.  For example **icc -x c file.cc** will complie the program as C instead of C++.
```

#### Language standard settings (Dialect)

```
-std=<std>  enable language support for <std>, as described below

c99
    conforms to ISO/IEC 9899:1999 standard for C programs

c11
    conforms to ISO/IEC 9899:2011 standard for C programs

c17
    conforms to ISO/IEC 9899:2017 standard for C programs

c18 
    conforms to ISO/IEC 9899:2018 standard for C programs

c++11
    enables C++11 support for C++ programs

c++14
    enables C++14 support for C++ programs

c++17
    enables C++17 support for C++ programs

c++20
    enables C++20 support for C++ programs

c89
    conforms to ISO/IEC 9899:1990 standard for C programs

gnu89
    conforms to ISO C90 plus GNU extensions

gnu99 
    conforms to ISO C99 plus GNU extensions

gnu++98 
    conforms to 1998 ISO C++ standard plus GNU extensions

gnu++11
    conforms to 2011 ISO C++ standard plus GNU extensions

gnu++14
    conforms to 2014 ISO C++ standard plus GNU extensions

gnu++17 
    conforms to 2017 ISO C++ standard plus GNU extensions

gnu++20 c
    onforms to 2020 ISO C++ standard plus GNU extensions

            
-strict-ansi 
    Implement a strict ANSI conformance dialect

    ```
    

#### Preprocessing

Unless explicitly disabled by the file extension as described above files are preprocessed.  If you pass the **-E** option the file will be preprocessed only and will not be compiled.  The output is sent to the standard output 

#### OpenMP support
```
-fopenmp
    Enable handling of OpenMP directives
-qopenmp-stubs
    Compile OpenMP programs in sequential mode 
-parallel          
    Auto parallelize
```

      
#### OpenACC support

```
Not supported
```

Offlading will not work on all platforms and may require additional options.          
           
* Important compiler specific options

```

-Wall
     This enables all the warnings about constructions that some
     users consider questionable, and that are easy to avoid (or
     modify to prevent the warning)
           
-Wextra 
     This enables some extra warning flags that are not enabled by
     -Wall.
     
-help [category]   print full or category help message

Valid categories include
       advanced        - Advanced Optimizations
       codegen         - Code Generation
       compatibility   - Compatibility
       component       - Component Control
       data            - Data
       deprecated      - Deprecated Options
       diagnostics     - Compiler Diagnostics
       float           - Floating Point
       help            - Help
       inline          - Inlining
       ipo             - Interprocedural Optimization (IPO)
       language        - Language
       link            - Linking/Linker
       misc            - Miscellaneous
       opt             - Optimization
       output          - Output
       pgo             - Profile Guided Optimization (PGO)
       preproc         - Preprocessor
       reports         - Optimization Reports

       openmp          - OpenMP and Parallel Processing
```
           
## Moving to Intel's new compiler icx
The Intel compilers icc and icpc are being retired and being replaced with icx and icpx.  
Other than the name change many people will not notice significant differences.  

The document [https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-icc-users-to-dpcpp-or-icx.html](https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-icc-users-to-dpcpp-or-icx.html) 
has details.  Here are some important blurbs from that page.



ICX and ICC Classic use different compiler drivers. The Intel® C++ Compiler Classic 
compiler drivers are icc, icpc, and icl.  The Intel® oneAPI DPC++/C++ Compiler drivers 
are icx and icpx. Use icx to compile and link C programs, and icpx for C++ programs.

Unlike the icc driver, icx does not use the file extension to determine whether to 
compile as C or C+. Users must invoke icpx to compile C+ files. . In addition to 
providing a core C++ Compiler, ICX/ICPX is also used to compile SYCL/DPC++ codes for the 
Intel® oneAPI Data Parallel C++ Compiler when we pass an additional flag “-fsycl”. 

The major changes in compiler defaults are listed below:

* The Intel® oneAPI DPC++/C++ Compiler drivers are icx and icpx.
* Intel® C++ Compiler Classic uses icc, icpc or icl drivers but this compiler will be deprecated in the upcoming release.
* DPC++/SYCL users can use the icx/icpx driver along with the -fsycl flag which invokes ICX with SYCL extensions. 
* Unlike Clang, the ICX Default floating point model was chosen to match ICC behavior and by default it is -fp-model=fast .
* MACRO naming is changing. Please be sure to check release notes for future macros to be included in ICX.
* No diagnostics numbers are listed for remarks, warnings, or notes. Every diagnostic is emitted with the corresponding compiler option to disable it. 
* Compiler intrinsics cannot be automatically recognized without processor targeting options, unlike the behavior in Intel® C++ Compiler Classic. If you use intrinsics, read more on the documentation about intrinsic behavior changes. 





## ifort

This discussion is for version 2021.6.0.  Ifort  will be replaced with a clang backend based alternative in the near future, ifx.  Ifx will have most of the same options as ifort with some clang additions.  In the Cray environment if PrgEnv-intel is loaded the "cc" maps to icc.

#### Normal invocation
```
# Compile and link a program with the executable sent to the indicated
  file
ifort mycode.f90 -o myexec

# Compile a file but don't link 
ifort -c mycode.c 

```

       
       
#### Default optimization
The default optimization level is -O2.  

#### Compiling for performance

```
-O1 optimize for maximum speed, but disable some optimizations which
	increase code size for a small speed benefit

-O2 optimize for maximum speed (DEFAULT)

-O3 optimize for maximum speed and enable more aggressive
	optimizations that may not improve performance on some programs

-O  same as -O2

-Os  enable speed optimizations, but disable some optimizations which
	increase code size for small speed benefit

-O0  disable optimizations

-Ofast  enable -O3 -no-prec-div -fp-model fast=2 optimizations

-fno-alias  assume no aliasing in program

-fno-fnalias  assume no aliasing within functions, but assume
	aliasing across calls

-fast  enable -xHOST -ipo -no-prec-div -O3 -static -fp-model=fast=2
	optimizations

-opt_report Generate and optimization report

```           
                   
You can learn more about optimizations are at various levels of optimization as shown below.  

```
ifort -V -help opt
```

#### Compiling for debugging and related purposes
```
 -g[n] 
       0 Disables generation of symbolic debug information.
       1 Produces minimal debug information for performing stack traces.
       2 Produces complete debug information. This is the same as specifying -g with no n.
       3 Produces extra information that may be useful for some tools.

none    Disables all check options.

arg_temp_created    Determines whether checking occurs for actual
	arguments copied into temporary storage before routine calls.

assume    Determines whether checking occurs to test that the
	scalar-Boolean-expression in the ASSUME directive is true, or
	that the addresses in the ASSUME_ALIGNED directive  are  aligned 
	on  the specified byte boundaries.

bounds    Determines whether checking occurs for array subscript and
	character s ubstring expressions.

contiguous    Determines whether the compiler checks pointer
	contiguity at pointer-assignment time.

format    Determines whether checking occurs for the data type of an
	item being formatted for output.

output_conversion    Determines whether checking occurs for the fit
	of data items within a designated format descriptor field.

pointers    Determines whether checking occurs for certain
	disassociated or uninitialized pointers or unallocated
	allocatable objects.

shape    Determines whether array conformance checking is performed.

stack    Determines whether checking occurs on the stack frame.

teams    Determines whether the run-time system diagnoses
	non-standard coarray team usage.

udio_iostat    Determines whether conformance checking occurs when
	user-defined derived type input/output routines are executed.

uninit     Determines whether checking occurs for uninitialized
	variables.

	all    Enables all check options.

-Os Generate extra code to write profile information suitable for
           the analysis program gprof
```

#### Some file extensions

```

Filenames with the suffix .f90 are interpreted as free-form Fortran
	95/90 source files.

Filenames with the suffix .f, .for, or .ftn are interpreted as
	fixed-form Fortran source files.

Filenames with the suffix .fpp, .F, .FOR, .FTN, or .FPP are
	interpreted as fixed-form Fortran source files, which must be
	preprocessed by the fpp preprocessor before being compiled.

Filenames with the suffix .F90 are interpreted as free-form Fortran
	source files, which must be pre-processed by the fpp preprocessor
	before being compiled.


```

You can specify explicitly the language for file indepenent of the extension using the -x option.  For example **icc -x c file.cc** will complie the program as C instead of C++.

#### Language standard settings (Dialect)

```
-stand 

none    Tells the compiler to issue no messages for nonstandard
	language elements. This is the same as specifying nostand.

f90    Tells the compiler to issue messages for language elements
	that are not standard in Fortran 90.

f95    Tells the compiler to issue messages for language elements
	that are not standard in Fortran 95.

f03    Tells the compiler to issue messages for language elements
	that are not standard in Fortran 2003.

f08    Tells the compiler to issue messages for language elements
	that are not standard in Fortran 2008.

f18    Tells the compiler to issue messages for language elements
	that are not standard in Fortran 2018. This option is set if you
	specify warn stderrors.
	
```    

### Generate Listing

```
-list
```

#### Preprocessing

Unless explicitly enabled by the file extension as described above files are not preprocessed.  If you pass the **-E** option the file will be preprocessed only and will not be compiled.  The output is sent to the standard output.  The option  **-fpp ** will force running the preprocessor.

#### OpenMP support
```
-fopenmp
    Enable handling of OpenMP directives
-qopenmp-stubs
    Compile OpenMP programs in sequential mode 
-parallel          
    Auto parallelize
```

      
#### OpenACC support

```
Not supported
```

#### Coarray Fortran

```
-coarray[=keyword] Enables the coarray feature where keyword
	Specifies the memory system where the coarrays will be
	implemented. Possible values are:

shared    Indicates a shared memory system. This is the default.

distributed    Indicates a distributed memory system.

single     Indicates a configuration where the image does not
	contain self-replication code. This results in an executable with
	a single running image. This configuration can be useful for
	debugging purposes, even though there are no inter-image
	interactions.

```

           
* Important compiler specific options

```

-save    Causes variables to be placed in static memory.


Default:    This option saves all variables in static allocation
	except local variables within a recursive routine and variables
	declared as AUTOMATIC.

-auto-scalar    Scalar variables of intrinsic types INTEGER, REAL,
	COMPLEX, and LOGICAL are allocated  to the run-time stack unless
	the routine is recursive of OpenMP For Fortran 95 and later
	variables are not saved by default and allocatable arrays are
	deallocated.  This appears to be true ifort even if the standard
	is set to f90.  However, it is poor practice to rely on this
	behavior.


-Wall.   This enables all the warnings about constructions that some
	users consider questionable, and that are easy to avoid (or
	modify to prevent the warning)

-warn declarations    Generate warnings for variables that are not
	explicitly typed.

-Wextra     This enables some extra warning flags that are not
	enabled by -Wall.
-save    Causes variables to be placed in static memory.


Default:    This option saves all variables in static allocation
	except local variables within a recursive routine and variables
	declared as AUTOMATIC.

-auto-scalar    Scalar variables of intrinsic types INTEGER, REAL,
	COMPLEX, and LOGICAL are allocated  to the run-time stack unless
	the routine is recursive of OpenMP For Fortran 95 and later
	variables are not saved by default and allocatable arrays are
	deallocated.  This appears to be true ifort even if the standard
	is set to f90.  However, it is poor practice to rely on this
	behavior.


-Wall.   This enables all the warnings about constructions that some
	users consider questionable, and that are easy to avoid (or
	modify to prevent the warning)

-warn declarations    Generate warnings for variables that are not
	explicitly typed.

-Wextra     This enables some extra warning flags that are not
	enabled by -Wall.

     
-help [category]    print full or category help message

Valid categories include
       advanced        - Advanced Optimizations
       codegen         - Code Generation
       compatibility   - Compatibility
       component       - Component Control
       data            - Data
       deprecated      - Deprecated Options
       diagnostics     - Compiler Diagnostics
       float           - Floating Point
       help            - Help
       inline          - Inlining
       ipo             - Interprocedural Optimization (IPO)
       language        - Language
       link            - Linking/Linker
       misc            - Miscellaneous
       opt             - Optimization
       output          - Output
       pgo             - Profile Guided Optimization (PGO)
       preproc         - Preprocessor
       reports         - Optimization Reports

       openmp          - OpenMP and Parallel Processing
```  

## Moving to Intel's new compiler ifx

Intel® Fortran Compiler Classic (ifort) is now deprecated and will be discontinued in late 2024. 
Intel recommends that customers transition now to using the LLVM-based Intel® Fortran Compiler (ifx).
Other  than the name change some people will not notice significant differences.  The new compiler
supports offloading to Intel GPU. Kestrel and Swift do not have Intel GPUs so this is not at NREL.  

One notable deletion from the new compiler is dropping of auto-parilization.  With ifort the 
-parallel compiler option auto-parallelization is enabled. That is not true for ifx; there
 is no auto-parallelization feature with ifx.
 
 
For complete details please see: [https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-ifort-to-ifx.html](https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-ifort-to-ifx.html)

         
## Cray CC

In the Cray environment cc is a generic call for several different compilers.  The compile actually called is determined by the modules loaded.  Here we discuss Cray C : Version 14.0.4.  cc will detect if the program being compiled calls MPI routines.  If so, it will call the program as MPI.  Cray C : Version 14.0.4 is clang based with Cray enhancements


#### Normal invocation
```
# Compile and link a program with the executable sent to the indicated
  file
cc mycode.c  -o myexec

# Compile a file but don't link 
cc -c mycode.c 



```
#### Default optimization
The default optimization level is -O0.

#### Compiling for performance

```

-O0, -O1, -O2, -O3, -Ofast, -Os, -Oz, -Og, -O, -O4 Specify which
	optimization level to use: 
	
-O0    Means "no optimization": this
	level compiles the fastest and generates the most debuggable
	code.

-O1    Somewhere between -O0 and -O2.

-O2    Moderate level of optimization which enables most
	optimizations.

-O3     Like -O2, except that it enables optimizations that take
	longer to perform or that may generate larger code (in an attempt
	to make the program run faster).

-Ofast     Enables all the optimizations from -O3 along with other
	aggressive optimizations that may violate strict compliance with
	language standards.

-Os     Like -O2 with extra optimizations to reduce code size.

-Oz    Like -Os (and thus -O2), but reduces code size further.

-Og    Like -O1. In future versions, this option might disable
	different optimizations in order to improve debuggability.

-O    Equivalent to -O1.

-O4    and higher Currently equivalent to -O3

	 
```
For best performance, -Ofast with -flto is recommended where -flot = Generate  output  files  in  LLVM  formats,  suitable for link time optimization.  The performance improvement with high levels of optimmization.  Here are the run times for a simple finite difference code at various levels of optimization.

```
Option       Run Time (sec)
-O0            10.30
-O1             3.19
-O2             2.99
-O3             2.04
-Ofast          1.88
-Ofast -flto    1.49
-Os             3.19
-Oz             3.31
-Og             3.19
-O              3.20
```
                   


#### Compiling for debugging and related purposes
```
-fstandalone-debug 
	  Turn off the stripping of some debug information that might be useful to some debuggers

-feliminate-unused-debug-types
	  By default, Clang does not emit type information for types that are defined but not 
	  used in a program. To retain the debug info for these unused types, the negation 
	  -fno-eliminate-unused-debug-types can be used.

-fexceptions
	  Enable generation of unwind information. This allows exceptions to be thrown through 
	  Clang compiled stack frames.  This is on by default in x86-64.

-ftrapv
	  Generate code to catch integer overflow errors.  Signed integer overflow is undefined 
	  in C. With this flag, extra code is generated to detect this and abort when it happens.


```

#### Some file extensions

```
file.c
   C source code that must be preprocessed.

file.i
   C source code that should not be preprocessed.

file.ii
   C++ source code that should not be preprocessed.

file.cc
file.cp
file.cxx
file.cpp
file.CPP
file.c++
file.C
   C++ source code that must be preprocessed.  
file.upc
   UPC
```   



#### Language standard settings (Dialect)

Standards are determined by the file extension as given above.  Some addttional checks can be performed.

      -std=<standard>
Specify the language standard to compile for.

##### Supported values for the C language are:



* ISO C 1999 with GNU extensions
	* c89
	* c90

* 	iso9899:1990

* ISO C 2011
	* 	c11
	* 	iso9899:2011


* ISO C 2011 with GNU extensions
	* 	gnu11
* ISO C 2017
	* 	iso9899:2017
	* 	c17

* ISO C 2017 with GNU extensions
	* 	gnu17

The default C language standard is gnu17

##### Supported values for the C++ language are:
* ISO C++ 1998 with amendments
	* c++98
	* c++03

* ISO C++ 1998 with amendments and GNU extensions
	* gnu++98
	* gnu++03

* ISO C++ 2011 with amendments
	* c++11

* ISO C++ 2011 with amendments and GNU extensions
	* gnu++11

* ISO C++ 2014 with amendments
	* c++14

* ISO C++ 2014 with amendments and GNU extensions
	* gnu++14

* ISO C++ 2017 with amendments
	* c++17

* ISO C++ 2017 with amendments and GNU extensions
	* gnu++17

* Working draft for ISO C++ 2020
	* c++2a

* Working draft for ISO C++ 2020 with GNU extensions
	* gnu++2a


* The default OpenCL language standard is cl1.0.
	* OpenCL


##### Supported values for the CUDA language are:
* cuda


       
    
#### Generating listing

```
-fsave-loopmark    Generate a loopmark listing file (.lst) that shows which optimizations 
    were applied to which parts of the source code.

-floopmark-style=<style>    Specifies the style of the loopmark listing file.

    Valid values for <style> are:
	    ''grouped''         Places all messages at the end of the listing.
		''interspersed''    Places each message after the relevant source code line.

```

#### Preprocessing

Automatic preprocessing is determined by the file name extension as discussed above. You can manually turn it on/off via the options
 
```
 -E    with output going to standard out
```

The compiler predefines the macro __cray__ in addition to all of the usual Clang predefined macros.


#### OpenMP support
```
-fopenmp    Enables OpenMP and links in OpenMP libraries
```
      
#### OpenACC support

```
Not suported    
 
```
           
#### Important compiler specific options

```
   Unified Parallel C (UPC) Options
-hupc, -hdefault -hupc    Configures the compiler driver to expect
	UPC source code.  Source files with a .upc extension are
	automatically treated as UPC code, but this option permits a file
	with any other extension (typically .c) to be understood as UPC
	code.  -hdefault cancels this behavior; if both -hupc and
	-hdefault appear in a command line, whichever appears last takes
	precedence and applies to all source files in the command line.

-fupc-auto-amo, -fno-upc-auto-amo    Automatically use network
	atomics for remote updates to reduce latency.  For example, x +=
	1 can be performed as a remote atomic add.  If an update is
	recognized as local to the current  thread,  then  no  atomic  is
	used.  These atomics are intended as a performance optimization
	only and shall not be relied upon to prevent race conditions. 
	Enabled at -O1 and above.

-fupc-buffered-async, -fno-upc-buffered-async    Set aside memory in
	the UPC runtime library for aggregating random remote accesses
	designated with "#pragma pgas buffered_async".  Disabled by
	default.

-fupc-pattern, -fno-upc-pattern    Identify simple communication
	loops and aggregate the remote accesses into a single function
	call which replaces the loop.  Enabled at -O1 and above.

-fupc-threads=<N>    Set  the number of threads for a static THREADS
	translation.  This option causes __UPC_STATIC_THREADS__ to be
	defined instead of __UPC_DYNAMIC_THREADS__ and replaces all uses
	of the UPC keyword THREADS with the value N.
      

```          

## Cray ftn

In the Cray environment ftn is a generic call for several different compilers.  The compile actually called is determined by the modules loaded.  Here we discuss Cray Fortran : Version 14.0.4.  Ftn will detect if the program being compiled calls MPI routines.  If so, it will call the program as MPI.

#### Normal invocation
```
# Compile and link a program with the executable sent to the indicated
  file
ftn mycode.f90  -o myexec

# Compile a file but don't link 
ftn -c mycode.f90

```
#### Default optimization
The default optimization level is -O 2.

#### Compiling for performance

```

-O

0      Specifies  no  automatic  cache  management;  all memory
	references are allocated to cache.  Both automatic cache blocking
	and manual cache blocking (by use of the BLOCKABLE directive) are
	shut off. Characteristics include low compile time.  This option
	is compatible with all optimization levels.

1      Specifies conservative automatic cache management.
	Characteristics include moderate compile time.  Symbols are
	placed in the cache when the possibility of cache reuse exists
	and the predicted cache footprint of the symbol in isolation is
	small enough to experience reuse.

2      Specifies  moderately  aggressive automatic cache management. 
	Characteristics include moderate compile time.  Symbols are
	placed in the cache when the possibility of cache reuse exists
	and the preâ€ dicted state of the cache model is such that the
	symbol will be reused. (Default)

3      Specifies aggressive automatic cache management.
	Characteristics include potentially high compile time.  Symbols
	are placed in the cache when the possibility of cache reuse
	exists and the  allocation of the symbol to the cache is
	predicted to increase the number of cache hits.

fast    Same as 3.

	 
```           
                   


#### Compiling for debugging and related purposes
```
-G (level)

	0      Full   information is available for debugging, but at the cost
		of a slower and larger executable.  Breakpoints can be set at
		each line.  Most optimizations are disabled.

	1      Most  information is available with partial optimization. Some
		optimizations make tracebacks and limited breakpoints available
		in the debugger.  Some scalar optimizations and  all  loop  nest
		reâ€ structuring  is  disabled,  but  the source code will be
		visible and most symbols will be available.

	2      Partial information.  Most optimizations, tracebacks and very
		limited breakpoints are available in the debugger.  The source
		code will be visible and some symbols will be  available.


-R runchk Specifies any of a group of runtime checks for your
	program.  To specify more than one type of checking, specify
	consecutive runchk arguments, as follows: -R bs.


	b      Enables checking of array bounds.  Bounds checking is not
		performed on arrays dimensioned as (1).  Enables -Ooverindex.

	c      Enables conformance checking of array operands in array
		expressions.

	d      Enables a run time check for the !dir$ collapse directive and
		checks the validity of the loop_info count information.

	p      Generates run time code to check the association or allocation
		status of referenced POINTER variables, ALLOCATABLE arrays, or
		assumed-shape arrays.

	s      Enables checking of character substring bounds.


```

#### Some file extensions

The default is fixed for source files that have .f, .F, .for, or .FOR

The default is free for source files that have .f90, .F90, .f95, .F95, .f03, .F03, .f08, .F08, .f18, .F18,  .ftn,  or  .FTN

The upper-case file extensions, .F, .FOR, .F90, .F95, .F03, .F08, .F18, or .FTN, will enable source preprocessing by default.




#### Language standard settings (Dialect)

Standards are determined by the file extension as given above.  Some addttional checks can be performed.

```

-e enable

	  b      If enabled, issue a warning message rather than an error
	  	message when the compiler detects a call to a procedure
	  	with one or more dummy arguments having the TARGET,
	  	VOLATILE or ASYNCHRONOUS attribute and there is not an
	  	explicit interface definition.


	  c      Interface checking: use Cray system modules to check
	  	library calls in a compilation.  If you have a procedure
	  	with the same name as one in the library, you will get
	  	errors, as the compiler does not skip  user- specified
	  	procedures when performing checks.


	  C      Enable/disable some types of standard call site
	  	checking.  The current Fortran standard requires that the
	  	number and types of arguments must agree between the caller
	  	and callee.  These constraints are enforced in cases where
	  	the compiler can detect them, however, specifying -dC
	  	disables some of this error-checking, which may be
	  	necessary in order to get some older Fortran codes to
	  	compile.

-f source_form free or fixed

           
```

#### Language standard settings (Save)
```

	-e v    Allocate  variables to static storage.  These variables
			are treated as if they had appeared in a SAVE statement.  Variables
			that are explicitly or implicitly defined as automatic variables are
			not allocated to static storage. The following types of variables are
			not allocated to static storage: automatic variables (explicitly or
			implicitly stated), variables declared with the AUTOMATIC attribute,
			variables allocated in  an  ALLOCATE statement, and local
			variables in explicit recursive procedures.  Variables with the
			ALLOCATABLE attribute remain allocated upon procedure exit, unless
			explicitly deallocated, but they are not allocated in static memory. 
			Variables in explicit recursive procedures consist of those in
			functions, in subroutines, and in internal procedures within
			functions and subroutines that have been defined with the RECURSIVE 
			attribute.  The STACK compiler directive overrides this option.



```

       
    
#### Generating listing

 -h list=a


#### Preprocessing

Automatic preprocessing is determined by the file name extension as discussed above. You can manually turn it on/off via the options
 
```
 -E    Preprocess and compile
 -eZ   Preprocess and compile
 -eP   Preprocess don' compile
```
The Cray Fortran preprocessor has limited functionality.  In particular it does not remove C style comments which can cause compile errors.  You might want to use the gnu preprocessor instead.  

```
gfortran -cpp -E file.F90 > file.f90
ftn file.f80
```


#### OpenMP support
```
-homp    Enables OpenMP and links in OpenMP libraries when possible
	using CCE-Classic.

-hnoomp    Disables OpenMP and links in non-OpenMP libraries when
	using CCE-classic.

THE FOLLOWING APPLIE IF THE BACKEND COMPILER IS NOT CRAY FORTRAN.

-fopenmp   Enables OpenMP and links in OpenMP libraries when possible
	using CCE, AOCC, and GNU.

-openmp	   Enables OpenMP and links in OpenMP libraries when
	possible.

-noopenmp		Disables OpenMP.

-mp		   Enables OpenMP and links in OpenMP libraries when
	possible using PGI.

-Mnoopenmp	Disables OpenMP and links in non-OpenMP libraries when
	using PGI.

-qopenmp     Enables OpenMP and links in OpenMP libraries when
	possible when using Intel.

-qno-openmp	 Disables OpenMP and links in non-OpenMP libraries
	when possible when using Intel.

```
      
#### OpenACC support

```
 -h acc         
 
```

#### Coarray
The -h pgas_runtime option directs the compiler driver to link with the runtime libraries required when linking programs that use UPC or coarrays.  In general, a resource manager job launcher such as aprun  or
                     srun must be used to launch the resulting executable.
           
#### Important compiler specific options

```

-e I      Treat all variables as if an IMPLICIT NONE statement had been specified. 
      

```           

