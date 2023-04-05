---
title: Fortran 90 for Fortran 77 programmers
postdate: October 13, 2020
layout: default
author: Tim Kaiser
description: Introduce Fortran 90 concepts to Fortran 77 programers
parent: Fortran
grand_parent: Programming Languages
---


# Advanced Fortran 90

This document is derived from an HTML page written at the [San Diego Supercomper Center](https://www.sdsc.edu) many years ago. Its purpose is to Introduce Fortran 90 concepts to Fortran 77 programers.  It does this by presenting an example program and introducing concepts as various routines of the program are presented.  The original web page has been used over the years and has been translated into several languages. 

**Note:** See our [Fortran Overview](f90.md) page for basic getting started instructions and compiler/toolchain information.

- - -
- - -


<!-- 
### Timothy H. Kaiser,Ph.D.
#### Written:  Summer 1997

- - -
- - -

## Introduction
- Personal Introduction
- The mind of the language writers
- Justification for topics covered
- Classification of topics covered
- Listing of covered topics
- Format of our presentation
- Meat

- - -
- - -

## Who am I?
- Wide experience background
	- Physics
    -  **Electrical Engineering**            
    -  Computer Science
- .gt. 25 years programming        
	- Defense Industry           
	- Academia
-  Languages
    -  Fortran
    -  C
    -  C++
    -  Pascal
    -  Lisp
    -  Java
    -  Python
    -  Others

-  Beta tester of several Fortran compilers
-  I have had the chance to make just about every mistake in the book and
    some that ain't

- - -
- - -
 -->
 
## Format for our presentation
- We will "develop" an application
    - Incorporate f90 features
    - Show source code
    - Explain what and why as we do it
- Application is a genetic algorithm
    - Easy to understand and program
    - Offers rich opportunities for enhancement
- We also provide an summary of F90 syntax, key words, operators, constants, and functions

- - -
- - -


## What was in mind of the language writers? What were they thinking?
- Enable portable codes
    - Same precision
    - Include many common extensions
- More reliable programs
- Getting away from underlying hardware
- Move toward parallel programming
- Run old programs
- Ease of programming
    - Writing
    - Maintaining
    - Understanding
    - Reading
- Recover C and C++ users

- - -
- - -

## Why Fortran?
 Famous Quote: *"I don't know what the technical characteristics of
 the standard language for scientific and engineering
 computation in the year 2000 will be... but I know it
 will be called Fortran." John Backus.*

Note: He claimed that he never said this.
 
- Language of choice for Scientific programming
- Large installed user base.
- Fortran 90 has most of the features of C . . . and then some
- The compilers produce better programs

- - -
- - -

## Justification of topics
- Enhance performance
- Enhance portability
- Enhance reliability
- Enhance maintainability

- - -
- - -

## Classification of topics
- New useful features
- Old tricks
- Power features
- Overview of F90

## What is a Genetic Algorithm
- A "suboptimization" system
    - Find good, but maybe not optimal, solutions to difficult problems
    -  Often used on NP-Hard or combinatorial optimization problems
- Requirements
    -  Solution(s) to the problem represented as a string
    -  A fitness function
		- Takes as input the solution string
		- Output the desirability of the solution
    -  A method of combining solution strings to generate new solutions
- Find solutions to problems by Darwinian evolution
	- Potential solutions ar though of as living entities in a population
    - The strings are the genetic codes for the individuals
    - Fittest individuals are allowed to survive to reproduce

- - -
- - -

## Simple algorithm for a GA
- Generate a initial population, a collection of strings
- do for some time
    -  evaluate each individual (string) of the population using the fitness function
    -  sort the population with fittest coming to the top
    -  allow the fittest individuals to "sexually" reproduce replacing the old
        population
    -  allow for mutation
- end do

- - -
- - -

## Our example problem
- Instance:Given a map of the N  states or countries and a fixed number of colors
- Find a coloring of the map, if it exists, such that no two states that share a boarder have the same color
- Notes
        - In general, for a fixed number of colors and an arbitrary map the only
        		known way to find if there is a valid coloring is a brute force search
        		with the number of combinations = (NUMBER_OF_COLORS)**(NSTATES)
    - The strings of our population are integer vectors represent the coloring
    - Our fitness function returns the number of boarder violations
    - The GA searches for a mapping with few, hopefully 0 violations
    - This problem is related to several important NP_HARD problems in computer science
        - Processor scheduling
        - Communication and grid allocation for parallel computing
        - Routing
                    
- - -
- - -

**Start of real Fortran 90 discussion**

- - -
- - -

## Comparing a FORTRAN 77 routine to a Fortran 90 routine
- The routine is one of the random number generators from:  *Numerical Recipes, The Art of Scientific Computing. Press, Teukolsky, Vetterling and Flannery.  Cambridge University Press 1986.*
- Changes
    -  correct bugs
    -  increase functionality
    -  aid portability

### Original

```fortran
    function ran1(idum)
        real ran1
        integer idum
        real r(97)
        parameter ( m1=259200,ia1=7141,ic1=54773)
        parameter ( m2=134456,ia2=8121,ic2=28411)
        parameter ( m3=243000,ia3=4561,ic3=51349)
        integer j
        integer iff,ix1,ix2,ix3
        data iff /0/
        if (idum.lt.0.or.iff.eq.0)then
            rm1=1.0/m1
            rm2=1.0/m2
            iff=1
            ix1=mod(ic1-idum,m1)
            ix1=mod(ia1*ix1+ic1,m1)
            ix2=mod(ix1,m2)
            ix1=mod(ia1*ix1+ic1,m1)
            ix3=mod(ix1,m3)
            do 11 j=1,97
                ix1=mod(ia1*ix1+ic1,m1)
                ix2=mod(ia2*ix2+ic2,m2)
                r(j)=(real(ix1)+real(ix2)*rm2)*rm1
 11           continue
            idum=1
        endif
        ix1=mod(ia1*ix1+ic1,m1)
        ix2=mod(ia2*ix2+ic2,m2)
        ix3=mod(ia3*ix3+ic3,m3)
        j=1+(97*ix3)/m3
        if(j.gt.97.or.j.lt.1)then
            write(*,*)' error in ran1 j=',j
            stop
        endif
        ran1=r(j)
        r(j)=(real(ix1)+real(ix2)*rm2)*rm1
        return
     end 
```

### Fortran 90
```fortran
module ran_mod
contains
     function ran1(idum)
        use numz
        implicit none  !note after use statement
        real (b8) ran1
        integer , intent(inout), optional ::  idum
        real (b8) r(97),rm1,rm2
        integer , parameter :: m1=259200,ia1=7141,ic1=54773
        integer , parameter :: m2=134456,ia2=8121,ic2=28411
        integer , parameter :: m3=243000,ia3=4561,ic3=51349
        integer j
        integer iff,ix1,ix2,ix3
        data iff /0/
        save ! corrects a bug in the original routine
        if(present(idum))then
          if (idum.lt.0.or.iff.eq.0)then
            rm1=1.0_b8 m1
            rm2=1.0_b8 m2
            iff=1
            ix1=mod(ic1-idum,m1)
            ix1=mod(ia1*ix1+ic1,m1)
            ix2=mod(ix1,m2)
            ix1=mod(ia1*ix1+ic1,m1)
            ix3=mod(ix1,m3)
            do j=1,97
                ix1=mod(ia1*ix1+ic1,m1)
                ix2=mod(ia2*ix2+ic2,m2)
                r(j)=(real(ix1,b8)+real(ix2,b8)*rm2)*rm1
            enddo
            idum=1
          endif
        endif
        ix1=mod(ia1*ix1+ic1,m1)
        ix2=mod(ia2*ix2+ic2,m2)
        ix3=mod(ia3*ix3+ic3,m3)
        j=1+(97*ix3)/m3
        if(j.gt.97.or.j.lt.1)then
            write(*,*)' error in ran1 j=',j
            stop
        endif
        ran1=r(j)
        r(j)=(real(ix1,b8)+real(ix2,b8)*rm2)*rm1
        return
     end function ran1
```

## Comments
1. Modules are a way of encapsulating functions an data.  More below.
1. The **use numz** line is similar to an include file.  In this case it defines our real data type.
1. **real (b8)**  is a new way to specify percision for data types in a portable way.
1. **integer , intent(inout), optional ::  idum** we are saying idum is an optional input parameter
1. **integer , parameter ::** just a different syntax
1. The **save** statement is needed for program correctness
1. **present(idum)** is a function to determine if ran1 was called with the optional parameter



- - -
- - -

## Obsolescent features
 The following are available in Fortran 90. On the other hand, the concept of "obsolescence" is introduced. This means that some constructs may be removed in the future.

- Arithmetic IF-statement
- Control variables in a DO-loop which are floating point or double-precision floating-point
- Terminating several DO-loops on the same statement
- Terminating the DO-loop in some other way than with CONTINUE or END DO
- Alternate return
- Jump to END IF from an outer block
- PAUSE
- ASSIGN and assigned GOTO and assigned FORMAT , that is the whole "statement number variable" concept.
- Hollerith editing in FORMAT.

- - -
- - -

## New source form and related things
### Summary
- ! now indicates the start of a comment
- & indicates the next line is a continuation
- Lines can be longer than 72 characters
- Statements can start in any column
- Use ; to put multiple statements on one line
- New forms for the do loop
- Many functions are generic
- 32 character names
- Many new array assignment techniques


### Features
- Flexibility can aid in program readability
- Readability decreases errors
- Got ya!
	- Can no longer use C to start a comment
	- Character in column 5 no longer is continue
	- Tab is not a valid character (may produce a warning)
	- Characters past 72 now count



```fortran
program darwin
     real a(10), b(10), c(10), d(10), e(10), x, y
     integer odd(5),even(5)
! this line is continued by using "&"
     write(*,*)"starting ",&  
                "darwin" ! this line in a continued from above
! multiple statement per line --rarely a good idea
     x=1; y=2; write(*,*)x,y  
     do i=1,10    ! statement lable is not required for do
        e(i)=i
     enddo
     odd= (/ 1,3,5,7,9 /)  ! array assignment
     even=(/ 2,4,6,8,10 /) ! array assignment
     a=1          ! array assignment, every element of a = 1
     b=2
     c=a+b+e      ! element by element assignment
     c(odd)=c(even)-1  ! can use arrays of indices on both sides
     d=sin(c)     ! element by element application of intrinsics
     write(*,*)d
     write(*,*)abs(d)  ! many intrinsic functions are generic
 a_do_loop : do i=1,10
               write(*,*)i,c(i),d(i)
             enddo a_do_loop
     do
        if(c(10) .lt. 0.0 ) exit
        c(10)=c(10)-1
     enddo
     write(*,*)c(10)
     do while (c(9) .gt. 0)
        c(9)=c(9)-1
     enddo
     write(*,*)c(9)
end program
```

- - -
- - -

## New data declaration method
- Motivation
    -  Variables can now have attributes such as
            - Parameter
                - Save
                - Dimension
    -  Attributes are assigned in the variable declaration statement

- One variable can have several attributes
- Requires Fortran 90 to have a new statement form

```fortran
integer,parameter :: in2 = 14
    real, parameter :: pi = 3.141592653589793239
    real, save, dimension(10) :: cpu_times,wall_times
!****    the old way of doing the same    ****!
!****    real cpu_times(10),wall_times(10) ****!
!****    save cpu_times, wall_times        ****!
```
- Other Attributes
    -  allocatable
    -  public
    -  private
    -  target
    -  pointer
    -  intent
    -  optional

- - -
- - -

## Kind facility
- Motivation
    -  Assume we have a program that we want to run on two different machines
    -  We want the same representation of reals on both machines (same number
        of significant digits)
    -  Problem: different machines have different representations for reals

### Digits of precision for some (old) machines and data type

	
|Machine|Real|Double Precision|
|:--|:--|:--|
|IBM (SP)|6| 15|
|Cray (T90) |15|33|
|Cray (T3E)|15|15|


### ********* or *********
- We may want to run with at least 6 digits today and at least 14 digits tomorrow
- Use the Select_Real_Kind(P) function to create a data type with P digits of precision

```fortran
program darwin
! e has at least 4 significant digits
  real(selected_real_kind(4))e
! b8 will be used to define reals with 14 digits
  integer, parameter:: b8 = selected_real_kind(14)
  real(b8), parameter :: pi = 3.141592653589793239_b8 ! note usage of _b8
! with  a constant
! to force precision
 e= 2.71828182845904523536
  write(*,*)"starting ",&  ! this line is continued by using "&"
            "darwin"       ! this line in a continued from above
  write(*,*)"pi has ",precision(pi)," digits precision ",pi
  write(*,*)"e has   ",precision(e)," digits precision ",e
end program
```
### Example output
```
  sp001  % darwin
 starting darwin
 pi has  15  digits precision  3.14159265358979312
 e has    6  digits precision  2.718281746
sp001 %
```

- Can convert to/from given precision for all variables created using "b8" by changing definition of "b8"
- Use the Select_Real_Kind(P,R) function to create a data type with P digits of precision and exponent range of R

- - -
- - -

## Modules
- Motivation:
    -  Common block usage is prone to error
    -  Provide most of capability of common blocks but safer
    -  Provide capabilities beyond common blocks

- Modules can contain:
    -  Data definitions
    -  Data to be shared much like using a labeled common
    -  Functions and subroutines
    -  Interfaces (more on this later)

- You "include" a module with a "use" statement

```fortran
module numz
  integer,parameter:: b8 = selected_real_kind(14)
  real(b8),parameter :: pi = 3.141592653589793239_b8
  integergene_size
end module
 program darwin
    use numz
    implicit none    ! now part of the standard, put it after the use statements
   write(*,*)"pi has ",precision(pi),"
digits precision ",pi
   call set_size()
   write(*,*)"gene_size=",gene_size
 end program
subroutine set_size
  use numz
  gene_size=10
end subroutine
```
### An example run
```
  pi has  15  digits precision  3.14159265358979312
  gene_size=10
```

- - -
- - -

## Module functions and subroutines
- Motivation:
    -  Encapsulate related functions and subroutines
    -  Can "USE" these functions in a program or subroutine
    -  Can be provided as a library
    -  Only routines that contain the use statement can see the routines


- Example is a random number package:
```fortran
module ran_mod
! module contains three functions
! ran1 returns a uniform random number between 0-1
! spread returns random number between min - max
! normal returns a normal distribution
contains
    function ran1()  !returns random number between 0 - 1
        use numz
        implicit none
        real(b8) ran1,x
        call random_number(x) ! built in fortran 90 random number function
        ran1=x
    end function ran1
    function spread(min,max)  !returns random # between min/max
        use numz
        implicit none
        real(b8) spread
        real(b8) min,max
        spread=(max - min) * ran1() + min
    end function spread
    function normal(mean,sigma) !returns a normal distribution
        use numz
        implicit none
        real(b8) normal,tmp
        real(b8) mean,sigma
        integer flag
        real(b8) fac,gsave,rsq,r1,r2
        save flag,gsave
        data flag /0/
        if (flag.eq.0) then
        rsq=2.0_b8
            do while(rsq.ge.1.0_b8.or.rsq.eq.0.0_b8) ! new from for do
                r1=2.0_b8*ran1()-1.0_b8
                r2=2.0_b8*ran1()-1.0_b8
                rsq=r1*r1+r2*r2
            enddo
            fac=sqrt(-2.0_b8*log(rsq)/rsq)
            gsave=r1*fac
            tmp=r2*fac
            flag=1
        else
            tmp=gsave
            flag=0
        endif
        normal=tmp*sigma+mean
        return
    end function normal end module ran_mod
```

- - -
- - -

 Exersize 1:  Write a program that returns 10 uniform random numbers.
- - -
- - -

## Allocatable arrays (the basics)
- Motivation:
    -  At compile time we may not know the size an array needs to be
    -  We may want to change problem size without recompiling

- Allocatable arrays allow us to set the size at run time
- We set the size of the array using the allocate statement
- We may want to change the lower bound for an array
- A simple example:

```fortran
module numz
  integer, parameter:: b8 = selected_real_kind(14)
  integer gene_size,num_genes
  integer,allocatable :: a_gene(:),many_genes(:,:)
end module
program darwin
    use numz
    implicit none
    integer ierr
    call set_size()
    allocate(a_gene(gene_size),stat=ierr) !stat= allows for an error code return
    if(ierr /= 0)write(*,*)"allocation error"  ! /= is .ne.
    allocate(many_genes(gene_size,num_genes),stat=ierr)  !2d array
    if(ierr /= 0)write(*,*)"allocation error"
    write(*,*)lbound(a_gene),ubound(a_gene) ! get lower and upper bound
                                            ! for the array
    write(*,*)size(many_genes),size(many_genes,1) !get total size and size
                                                  !along 1st dimension
    deallocate(many_genes) ! free the space for the array and matrix
    deallocate(a_gene)
    allocate(a_gene(0:gene_size)) ! now allocate starting at 0 instead of 1
    write(*,*)allocated(many_genes),allocated(a_gene) ! shows if allocated
    write(*,*)lbound(a_gene),ubound(a_gene)
end program
  subroutine set_size
    use numz
    write(*,*)'enter gene size:'
    read(*,*)gene_size
    write(*,*)'enter number of genes:'
    read(*,*)num_genes
end subroutine set_size
```
## Example run
```
    enter gene size:
10
 enter number of genes:
20
           1          10
         200          10
 F T
           0          10
```

## Passing arrays to subroutines

- There are several ways to specify arrays for subroutines
	- Explicit shape
		- integer, dimension(8,8)::an_explicit_shape_array
	- Assumed size
		- integer, dimension(i,*)::an_assumed_size_array
	- Assumed Shape
		- integer, dimension(:,:)::an_assumed_shape_array

### Example

```fortran
subroutine arrays(an_explicit_shape_array,&
                  i                      ,& !note we pass all bounds except the last
                  an_assumed_size_array  ,&
                  an_assumed_shape_array)
! Explicit shape
    integer, dimension(8,8)::an_explicit_shape_array
! Assumed size
    integer, dimension(i,*)::an_assumed_size_array
! Assumed Shape
    integer, dimension(:,:)::an_assumed_shape_array
    write(*,*)sum(an_explicit_shape_array)
    write(*,*)lbound(an_assumed_size_array) ! why does sum not work here?
    write(*,*)sum(an_assumed_shape_array)
end subroutine
```

- - -
- - -

## Interface for passing arrays
- **!!!!Warning!!!!  When passing assumed shape arrays as arguments you must provide an interface**
- Similar to C prototypes but much more versatile
- The interface is a copy of the invocation line and the argument definitions
- Modules are a good place for interfaces
- If a procedure is part of a "contains" section in a module an interface
    is not required
- **!!!!Warning!!!! The compiler may not tell you that you need an interface**
```fortran
module numz
    integer, parameter:: b8 = selected_real_kind(14)
    integer,allocatable :: a_gene(:),many_genes(:,:)
end module module face
    interface fitness
        function fitness(vector)
        use numz
        implicit none
        real(b8) fitness
        integer, dimension(:) ::  vector
        end function fitness
    end interface
end module program darwin
    use numz
    use face
    implicit none
    integer i
    integer vect(10) ! just a regular array
    allocate(a_gene(10));allocate(many_genes(3,10))
    a_gene=1  !sets every element of a_gene to 1
    write(*,*)fitness(a_gene)
    vect=8
    write(*,*)fitness(vect) ! also works with regular arrays
    many_genes=3  !sets every element to 3
    many_genes(1,:)=a_gene  !sets column 1 to a_gene
    many_genes(2,:)=2*many_genes(1,:)
    do i=1,3
        write(*,*)fitness(many_genes(i,:))
    enddo
    write(*,*)fitness(many_genes(:,1))  !go along other dimension
!!!!write(*,*)fitness(many_genes)!!!!does not work
end program
function fitness(vector)
    use numz
    implicit none
    real(b8) fitness
    integer, dimension(:)::  vector ! must match interface
    fitness=sum(vector)
end function
```
- - -
- - -

Exersize 2:  Run this program using the "does not work line".
Why?  Using intrinsic functions make it work?

Exersize 3:  Prove that f90 does not "pass by address".
- - -
- - -

## Optional arguments and intent
- Motivation:
    -  We may have a function or subroutine that we may not want to always pass
        all arguments
    -  Initialization
- Two examples
    - Seeding the intrinsic random number generator requires keyword arguments
    -  To define an optional argument in our own function we use the optional
        attribute

```
integer :: my_seed
```
### becomes
```
integer, optional :: my_seed
```

Used like this:

```fortran
! ran1 returns a uniform random number between 0-1
! the seed is optional and used to reset the generator
contains
   function ran1(my_seed)
      use numz
      implicit none
      real(b8) ran1,r
      integer, optional ,intent(in) :: my_seed  ! optional argument not changed in the routine
      integer,allocatable :: seed(:)
      integer the_size,j
      if(present(my_seed))then            ! use the seed if present
          call random_seed(size=the_size) ! how big is the intrisic seed?
          allocate(seed(the_size))        ! allocate space for seed
          do j=1,the_size                 ! create the seed
             seed(j)=abs(my_seed)+(j-1)   ! abs is generic
          enddo
          call random_seed(put=seed)      ! assign the seed
          deallocate(seed)                ! deallocate space
      endif
      call random_number(r)
      ran1=r
  end function ran1
end module program darwin
    use numz
    use ran_mod          ! interface required if we have
                         ! optional or intent arguments
    real(b8) x,y
    x=ran1(my_seed=12345) ! we can specify the name of the argument
    y=ran1()
    write(*,*)x,y
    x=ran1(12345)         ! with only one optional argument we don't need to
    y=ran1()
    write(*,*)x,y
end program
```


- Intent is a hint to the compiler to enable optimization
	- intent(in)
		- We will not change this value in our subroutine
	- intent(out)
		- We will define this value in our routine
	- intent(inout)
		- The normal situation

- - -
- - -

## Derived data types
- Motivation:
    -  Derived data types can be used to group different types of data together
        (integers, reals, character, complex)
    -  Can not be done in F77 although people have "faked" it

- Example
    -  In our GA we define a collection of genes as a 2d array
    -  We call the fitness function for every member of the collection
    -  We want to sort the collection of genes based on result of fitness function
    -  Define a data type that holds the fitness value and an index into the 2d
        array
    -  Create an array of this data type, 1 for each member of the collection
    -  Call fitness function with the result being placed into the new data type
        along with a pointer into the array
- Again modules are a good place for data type definitions


```fortran
module galapagos
    use numz
    type thefit !the name of the type
      sequence  ! sequence forces the data elements
                ! to be next to each other in memory
                ! where might this be useful?
      real(b8) val   ! our result from the fitness function
      integer index  ! the index into our collection of genes
    end type thefit
end module
```

- - -
- - -

## Using defined types
- Use the % to reference various components of the derived data type
```fortran
program darwin
    use numz
    use galapagos ! the module that contains the type definition
    use face      ! contains various interfaces
 implicit none
! define an allocatable array of the data type
! than contains an index and a real value
    type (thefit),allocatable ,target  :: results(:)
! create a single instance of the data type
    type (thefit) best
    integer,allocatable :: genes(:,:) ! our genes for the genetic algorithm
    integer j
    integer num_genes,gene_size
    num_genes=10
    gene_size=10
    allocate(results(num_genes))         ! allocate the data type
                                         ! to hold fitness and index
    allocate(genes(num_genes,gene_size)) ! allocate our collection of genes
    call init_genes(genes)               ! starting data
    write(*,'("input")' ) ! we can put format in write statement
    do j=1,num_genes
       results(j)%index =j
       results(j)%val =fitness(genes(j,:)) ! just a dummy routine for now
       write(*,"(f10.8,i4)")results(j)%val,results(j)%index
    enddo
end program
```

- - -
- - -

## User defined operators
- Motivation
    -  With derived data types we may want (need) to define operations
    -  (Assignment is predefined)

- Example:
    -  .lt. .gt. ==  not defined for our data types
            -  We want to find the minimum of our fitness values so we need &lt; operator
            -  In our sort routine we want to do &lt;, &gt;, ==
            -  In C++ terms the operators are overloaded
    -  We are free to define new operators

- Two step process to define operators
    -  Define a special interface
    -  Define the function that performs the operation
```fortran
module sort_mod
!defining the interfaces
  interface operator (.lt.)  ! overloads standard .lt.
    module procedure theless ! the function that does it
  end interface   interface operator (.gt.)   ! overloads standard .gt.
    module procedure thegreat ! the function that does it
  end interface   interface operator (.ge.)  ! overloads standard .ge.
    module procedure thetest ! the function that does it
  end interface   interface operator (.converged.)  ! new operator
    module procedure index_test     ! the function that does it
  end interface
  contains      ! our module will contain
              ! the required functions
    function theless(a,b) ! overloads .lt. for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical theless           ! what we return
    if(a%val .lt. b%val)then     ! this is where we do the test
        theless=.true.
    else
        theless=.false.
    endif
    return
  end function theless   function thegreat(a,b) ! overloads .gt. for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical thegreat
    if(a%val .gt. b%val)then
        thegreat=.true.
    else
        thegreat=.false.
    endif
    return
  end function thegreat
  function thetest(a,b)   ! overloads .gt.= for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical thetest
    if(a%val >= b%val)then
        thetest=.true.
    else
        thetest=.false.
    endif
    return
end function thetest
  function index_test(a,b) ! defines a new operation for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical index_test
    if(a%index .gt. b%index)then   ! check the index value for a difference
        index_test=.true.
    else
        index_test=.false.
    endif
    return
end function index_test
```



- - -
- - -

## Recursive functions introduction
- Notes

    -  Recursive function is one that calls itself
    -  Anything that can be done with a do loop can be done using a recursive
        function

- Motivation
    -  Sometimes it is easier to think recursively
    -  Divide an conquer algorithms are recursive by nature
            -  Fast FFTs
                -  Searching
                -  Sorting


### Algorithm of searching for minimum of an array

```fortran
    function findmin(array)
        is size of array 1?
           min in the array is first element
        else
           find minimum in left half of array using findmin function
           find minimum in right half of array using findmin function
           global minimum is min of left and right half
    end function
```

- - -
- - -

## Fortran 90 recursive functions
- Recursive functions should have an interface
- The result and recursive keywords are required as part of the function definition
- Example is a function finds the minimum value for an array

```fortran
recursive function realmin(ain) result (themin)
! recursive and result are required for recursive functions
    use numz
    implicit none
    real(b8) themin,t1,t2
    integer n,right
    real(b8) ,dimension(:) :: ain
    n=size(ain)
    if(n == 1)then
       themin=ain(1) ! if the size is 1 return value
    return
    else
      right=n/2
      t1=realmin(ain(1:right))   ! find min in left half
      t2=realmin(ain(right+1:n)) ! find min in right half
      themin=min(t1,t2)          ! find min of the two sides
     endif
end function
```

- Example 2 is the same except the input data is our derived data type

```fortran
!this routine works with the data structure thefit not reals
recursive function typemin(ain) result (themin)
    use numz
 use sort_mod
 use galapagos
 implicit none
 real(b8) themin,t1,t2
 integer n,right
    type (thefit) ,dimension(:) :: ain ! this line is different
 n=size(ain)
 if(n == 1)then
     themin=ain(1)%val  ! this line is different
  return
 else
  right=n/2
  t1=typemin(ain(1:right))
  t2=typemin(ain(right+1:n))
  themin=min(t1,t2)
 endif
end function
```

- - -
- - -

## Pointers

- Motivation
    -  Can increase performance
    -  Can improve readability
    -  Required for some derived data types (linked lists and trees)
    -  Useful for allocating "arrays" within subroutines
    -  Useful for referencing sections of arrays

- Notes
    -  Pointers can be thought of as an alias to another variable
    -  In some cases can be used in place of an array
    -  To assign a pointer use => instead of just =
    -  Unlike C and C++, pointer arithmetic is not allowed


- First pointer example
    -  Similar to the last findmin routine
    -  Return a pointer to the minimum

```fortran
recursive function pntmin(ain) result (themin) ! return a pointer
 use numz
 use galapagos
 use sort_mod ! contains the .lt. operator for thefit type
 implicit none
 type (thefit),pointer:: themin,t1,t2
 integer n,right
    type (thefit) ,dimension(:),target :: ain
 n=size(ain)
 if(n == 1)then
     themin=>ain(1) !this is how we do pointer assignment
  return
 else
  right=n/2
  t1=>pntmin(ain(1:right))
  t2=>pntmin(ain(right+1:n))
  if(t1 .lt. t2)then; themin=>t1; else; themin=>t2; endif
 endif
end function
```
- - -
- - -

Exercise 4:  Carefully write a recursive N! program.

- - -
- - -

## Function and subroutine overloading
- Motivation
    -  Allows us to call functions or subroutine with the same name with different
        argument types
    -  Increases readability

- Notes:
    -  Similar in concept to operator overloading
    -  Requires an interface
    -  Syntax for subroutines is same as for functions
    -  Many intrinsic functions have this capability
            -  abs (reals,complex,integer)
                -  sin,cos,tan,exp(reals, complex)
                -  array functions(reals, complex,integer)
    -  Example
            -  Recall we had two functions that did the same thing but with different argument types

```
         recursive function realmin(ain) result (themin)
         real(b8) ,dimension(:) :: ain         recursive function typemin(ain) result (themin)
         type (thefit) ,dimension(:) :: ain
```
- We can define a generic interface for these two functions and call
    them using the same name
    
```fortran
! note we have two functions within the same interface
! this is how we indicate function overloading
! both functions are called "findmin" in the main program
interface findmin
! the first is called with an array of reals as input
        recursive function realmin(ain) result (themin)
          use numz
       real(b8) themin
          real(b8) ,dimension(:) :: ain
        end function ! the second is called with a array of data structures as input
     recursive function typemin(ain) result (themin)
          use numz
    use galapagos
       real(b8) themin
          type (thefit) ,dimension(:) :: ain
     end function
    end interface
```
### Example usage

```fortran
program darwin
    use numz
    use ran_mod
    use galapagos ! the module that contains the type definition
    use face      ! contains various interfaces
    use sort_mod  ! more about this later it
                  ! contains our sorting routine
      ! and a few other tricks
    implicit none
! create an allocatable array of the data type
! than contains an index and a real value
    type (thefit),allocatable ,target :: results(:)
! create a single instance of the data type
    type (thefit) best
! pointers to our type
    type (thefit) ,pointer :: worst,tmp
    integer,allocatable :: genes(:,:) ! our genes for the ga
    integer j
    integer num_genes,gene_size
    real(b8) x
    real(b8),allocatable :: z(:)
    real(b8),pointer :: xyz(:) ! we'll talk about this next
    num_genes=10
    gene_size=10
    allocate(results(num_genes))         ! allocate the data type to
    allocate(genes(num_genes,gene_size)) ! hold our collection of genes
    call init_genes(genes)               ! starting data
    write(*,'("input")')
    do j=1,num_genes
       results(j)%index=j
       results(j)%val=fitness(genes(j,:)) ! just a dummy routine
       write(*,"(f10.8,i4)")results(j)%val,results(j)%index
    enddo     allocate(z(size(results)))
    z=results(:)%val ! copy our results to a real array ! use a recursive subroutine operating on the real array
    write(*,*)"the lowest fitness: ",findmin(z)
! use a recursive subroutine operating on the data structure
    write(*,*)"the lowest fitness: ",findmin(results)
end program
```

- - -
- - -

## Fortran Minval and Minloc routines
- Fortran has routines for finding minimum and maximum values in arrays and
    the locations
    -  minval
    -  maxval
    -  minloc (returns an array)
    -  maxloc (returns an array)

```fortran
! we show two other methods of getting the minimum fitness
! use the built in f90 routines  on a real array
    write(*,*)"the lowest fitness: ",minval(z),minloc(z)
```

- - -
- - -

## Pointer assignment
- This is how we use the pointer function defined above
- worst is a pointer to our data type
- note the use of =>
```
! use a recursive subroutine operating on the data
! structure and returning a pointer to the result
    worst=>pntmin(results) ! note pointer assignment
! what will this line write?
 write(*,*)"the lowest fitness: ",worst
```

- - -
- - -

## More pointer usage, association and nullify
- Motivation
    -  Need to find if pointers point to anything
    -  Need to find if two pointers point to the same thing
    -  Need to deallocate and nullify when they are no longer used

- Usage
    -  We can use associated() to tell if a pointer has been set
    -  We can use associated() to compare pointers
    -  We use nullify to zero a pointer


```fortran
! This code will print "true" when we find a match,
! that is the pointers point to the same object
    do j=1,num_genes
     tmp=>results(j)
        write(*,"(f10.8,i4,l3)")results(j)%val,   &
                                results(j)%index, &
           associated(tmp,worst)
    enddo
    nullify(tmp)
```

- Notes:
    -  If a pointer is nullified the object to which it points is not deallocated.
    -  In general, pointers as well as allocatable arrays become undefined on leaving a subroutine
    -  This can cause a memory leak


- - -
- - -

## Pointer usage to reference an array without copying
- Motivation
    -  Our sort routine calls a recursive sorting routine
    -  It is messy and inefficient to pass the array to the recursive routine
- Solution
    -  We define a "global" pointer in a module
    -  We point the pointer to our input array

```fortran
module Merge_mod_types
    use galapagos
    type(thefit),allocatable :: work(:) ! a "global" work array
    type(thefit), pointer:: a_pntr(:)   ! this will be the pointer to our input array
end module Merge_mod_types
  subroutine Sort(ain, n)
    use Merge_mod_types
    implicit none
    integer n
    type(thefit), target:: ain(n)
    allocate(work(n))
    nullify(a_pntr)
    a_pntr=>ain  ! we assign the pointer to our array
                 ! in RecMergeSort we reference it just like an array
    call RecMergeSort(1,n) ! very similar to the findmin functions
    deallocate(work)
    return
end subroutine Sort
```

- In our main program sort is called like this:
```fortran
! our sort routine is also recursive but
! also shows a new usage for pointers
    call sort(results,num_genes)
    do j=1,num_genes
       write(*,"(f10.8,i4)")results(j)%val,   &
                            results(j)%index
    enddo
```

- - -
- - -

## Data assignment with structures

```fortran
! we can copy a whole structure
! with a single assignment
    best=results(1)
    write(*,*)"best result ",best
```
- - -
- - -

## Using the user defined operator

```fortran
! using the user defined operator to see if best is worst
! recall that the operator .converged. checks to see if %index matches
    worst=>pntmin(results)
    write(*,*)"worst result ",worst
    write(*,*)"converged=",(best .converged. worst)
```

- - -
- - -

## Passing arrays with a given arbitrary lower bounds
- Motivation
    - Default lower bound within a subroutine is 1

    - May want to use a different lower bound

```fortran
    if(allocated(z))deallocate(z)
    allocate(z(-10:10)) ! a 21 element array
    do j=-10,10
       z(j)=j
    enddo ! pass z and its lower bound
! in this routine we give the array a specific lower
! bound and show how to use a pointer to reference
! different parts of an array using different indices
  call boink1(z,lbound(z,1)) ! why not just lbound(z) instead of lbound(z,1)?
                             ! lbound(z) returns a rank 1 array
     subroutine boink1(a,n)
     use numz
     implicit none
     integer,intent(in) :: n
     real(b8),dimension(n:):: a ! this is how we set lower bounds in a subroutine
     write(*,*)lbound(a),ubound(a)
   end subroutine
```


### Warning:  because we are using an assumed shape array we need an interface
## Using pointers to access sections of arrays
- Motivation
    - Can increase efficiency
    - Can increase readability

```fortran
call boink2(z,lbound(z,1))

subroutine boink2(a,n)
use numz
implicit none
integer,intent(in) :: n
real(b8),dimension(n:),target:: a
real(b8),dimension(:),pointer::b
b=>a(n:) ! b(1) "points" to a(-10)
write(*,*)"a(-10) =",a(-10),"b(1) =",b(1)
b=>a(0:) ! b(1) "points" to a(0)
write(*,*)"a(-6) =",a(-6),"b(-5) =",b(-5)
end subroutine
```

- - -
- - -

## Allocating an array inside a subroutine and passing it back
- Motivation
    - Size of arrays are calculated in the subroutine

```fortran
module numz
    integer, parameter:: b8 = selected_real_kind(14)
end module
program bla
   use numz
   real(b8), dimension(:) ,pointer :: xyz
   interface boink
     subroutine boink(a)
     use numz
     implicit none
     real(b8), dimension(:), pointer :: a
     end subroutine
   end interface
   nullify(xyz) ! nullify sets a pointer to null
   write(*,'(l5)')associated(xyz) ! is a pointer null, should be
   call boink(xyz)
   write(*,'(l5)',advance="no")associated(xyz)
   if(associated(xyz))write(*,'(i5)')size(xyz)
end program
subroutine boink(a)
    use numz
    implicit none
    real(b8),dimension(:),pointer:: a
    if(associated(a))deallocate(a)
    allocate(a(10))
end subroutine
```
### An example run
```

     F
     T
10
```
- - -
- - -

## Our fitness function
Given a fixed number of colors, M, and a description of a map of a collection
of  N states.

Find a coloring of the map such that no two states that share a boarder
have the same coloring.
### Example input is a sorted list of 22 western states
```
22
ar ok tx la mo xx
az ca nm ut nv xx
ca az nv or xx
co nm ut wy ne ks xx
ia mo ne sd mn xx
id wa or nv ut wy mt xx
ks ne co ok mo xx
la tx ar xx
mn ia sd nd xx
mo ar ok ks ne ia xx
mt wy id nd xx
nd mt sd wy xx
ne sd wy co ks mo ia xx
nm az co ok tx mn xx
nv ca or id ut az xx
ok ks nm tx ar mo xx
or ca wa id xx
sd nd wy ne ia mn xx
tx ok nm la ar xx
ut nv az co wy id xx
wa id or mt xx
wy co mt id ut nd sd ne xx
```

Our fitness function takes a potential coloring, that is, an integer
vector of length N and a returns the number of boarders that have states
of the same coloring

- How do we represent the map in memory?
    -  One way would be to use an array but it would be very sparse
    -  Linked lists are often a better way

- - -
- - -

## Linked lists
- Motivation
	- We have a collection of states and for each state a list of adjoining states. (Do not count a boarder twice.)
    - Problem is that you do not know the length of the list until runtime.
    - List of adjoining states will be different lengths for different states

    - Solution
    	    - 	Linked list are a good way to handle such situations
    - 	Linked lists use a derived data type with at least two components
        - Data	
        - Pointer to next element
                			
```fortran
module list_stuff
type llist
integer index ! data
type(llist),pointer::next ! pointer to the
! next element
end type llist
end module
```

- - -
- - -

## Linked list usage
One way to fill a linked list is to use a recursive function
``fortran`
recursive subroutine insert (item, root)
use list_stuff
implicit none
type(llist), pointer :: root
integer item
if (.not. associated(root)) then
allocate(root)
nullify(root%next)
root%index = item
else
call insert(item,root%next)
endif
end subroutine
```

- - -
- - -

## Our map representation
- An array of the derived data type states
    	    - 	State is name of a state
    - 	Linked list containing boarders

```fortran
    type states
        character(len=2)name
        type(llist),pointer:: list
    end type states
```
- Notes:
    -  We have an array of linked lists
            -  This data structure is often used to represent sparse arrays
                -  We could have a linked list of linked lists
    -  State name is not really required

- - -
- - -


## Date and time functions
- Motivation
    -  May want to know the date and time of your program

    -  Two functions

```fortran
! all arguments are optional
call date_and_time(date=c_date, &  ! character(len=8) ccyymmdd
                   time=c_time, &  ! character(len=10) hhmmss.sss
                   zone=c_zone, &  ! character(len=10) +/-hhmm (time zone)
                   values=ivalues) ! integer ivalues(8) all of the above
           call system_clock(count=ic,           & ! count of system clock (clicks)
                  count_rate=icr,     & ! clicks / second
                  count_max=max_c)      ! max value for count
```

- - -
- - -

## Non advancing and character IO
- Motivation
    -  We read the states using the two character identification

    -  One line per state and do not know how many boarder states per line

- Note: Our list of states is presorted
```fortran
character(len=2) a ! we have a character variable of length 2
read(12,*)nstates ! read the number of states
allocate(map(nstates)) ! and allocate our map
do i=1,nstates
    read(12,"(a2)",advance="no")map(i)%name ! read the name
    !write(*,*)"state:",map(i)%name
    nullify(map(i)%list) ! "zero out" our list
    do
        read(12,"(1x,a2)",advance="no")a ! read list of states
        ! without going to the
        ! next line
        if(lge(a,"xx") .and. lle(a,"xx"))then ! if state == xx
        backspace(12) ! go to the next line
        read(12,"(1x,a2)",end=1)a ! go to the next line
        exit
        endif
        1 continue
        if(llt(a,map(i)%name))then ! we only add a state to
        ! our list if its name
        ! is before ours thus we
        ! only count boarders 1 time
        ! what we want put into our linked list is an index
        ! into our map where we find the bordering state
        ! thus we do the search here
        ! any ideas on a better way of doing this search?
        found=-1
        do j=1,i-1
            if(lge(a,map(j)%name) .and. lle(a,map(j)%name))then
            !write(*,*)a
            found=j
            exit
            endif
        enddo
        if(found == -1)then
            write(*,*)"error"
            stop
        endif
        ! found the index of the boarding state insert it into our list
        ! note we do the insert into the linked list for a particular state
        call insert(found,map(i)%list)
        endif
    enddo
enddo
```

- - -
- - -


## Internal IO
- Motivation
    -  May need to create strings on the fly

    -  May need to convert from strings to reals and integers

    -  Similar to sprintf and sscanf

- How it works
    -  Create a string

    -  Do a normal write except write to the string instead of file number

- Example 1: creating a date and time stamped file name

```fortran
character (len=12)tmpstr

write(tmpstr,"(a12)")(c_date(5:8)//c_time(1:4)//".dat") ! // does string concatination
write(*,*)"name of file= ",tmpstr
open(14,file=tmpstr)
name of file= 03271114.dat
```

- Example 2: Creating a format statement at run time (array of integers and a real)

```fortran
! test_vect is an array that we do not know its length until run time
nstate=9 ! the size of the array
write(fstr,'("(",i4,"i1,1x,f10.5)")')nstates
write(*,*)"format= ",fstr
write(*,fstr)test_vect,fstr
format= ( 9i1,1x,f10.5)
```
Any other ideas for writing an array when you do not know its length?

- Example 3: Reading from a string
```fortran
integer ht,minut,sec
read(c_time,"(3i2)")hr,minut,sec
```

- - -
- - -

## Inquire function
- Motivation
    - Need to get information about I/O
- Inquire statement has two forms
    -  Information about files (23 different requests can be done)
    -  Information about space required for binary output of a value

- Example: find the size of your real relative to the "standard" real
    - Useful for inter language programming
    - Useful for determining data types in MPI (MPI_REAL or MPI_DOUBLE_PRECISION)

```fortran
inquire(iolength=len_real)1.0
inquire(iolength=len_b8)1.0_b8
write(*,*)"len_b8 ",len_b8
write(*,*)"len_real",len_real
iratio=len_b8/len_real
select case (iratio)
    case (1)
      my_mpi_type=mpi_real
    case(2)
      my_mpi_type=mpi_double_precision
    case default
      write(*,*)"type undefined"
      my_mpi_type=0
end select
```

### An example run

```fortran
len_b8 2
len_real 1
```

- - -
- - -

## Namelist
- Now part of the standard
- Motivation
    -  A convenient method of doing I/O
    -  Good for cases where you have similar runs but change one or two variables
    -  Good for formatted output
- Notes:
    -  A little flaky
    -  No options for overloading format

- Example:
```fortran
integer ncolor
logical force
namelist /the_input/ncolor,force
ncolor=4
force=.true.
read(13,the_input)
write(*,the_input)
```
On input:
```
& THE_INPUT NCOLOR=4,FORCE = F /
```
Output is
```
&THE_INPUT
NCOLOR = 4,
FORCE = F
/
```

- - -
- - -

## Vector valued functions
- Motivation
    -  May want a function that returns a vector
- Notes
    -  Again requires an interface
    -  Use explicit or assumed size array
    -  Do not return a pointer to a vector unless you really want a pointer

- Example:
    -  Take an integer input vector which represents an integer in some base and
        add 1
    -  Could be used in our program to find a "brute force" solution

```fortran
  function add1(vector,max) result (rtn)
  integer, dimension(:),intent(in) ::  vector
  integer,dimension(size(vector)) :: rtn
  integer max
  integer len
  logical carry
  len=size(vector)
  rtn=vector
  i=0
  carry=.true.
  do while(carry)         ! just continue until we do not do a carry
      i=i+1
   rtn(i)=rtn(i)+1
   if(rtn(i) .gt. max)then
       if(i == len)then   ! role over set everything back to 0
        rtn=0
    else
        rtn(i)=0
       endif
   else
       carry=.false.
   endif
  enddo
end function
```
### Usage
```fortran
test_vect=0
        do
           test_vect=add1(test_vect,3)
           result=fitness(test_vect)
           if(result .lt. 1.0_b8)then
               write(*,*)test_vect
               stop
           endif
        enddo
```

- - -
- - -

## Complete source for recent discussions
- [recent.f90](source/recent.f90)
- [fort.13](source/fort.13)

- - -
- - -

Exersize 5  Modify the program to use the random
number generator given earlier.

- - -
- - -

## Some array specific intrinsic functions

- ALL True if all values are true (LOGICAL)
- ANY True if any value is true (LOGICAL)
- COUNT Number of true elements in an array (LOGICAL)
- DOT_PRODUCT Dot product of two rank one arrays
- MATMUL Matrix multiplication
- MAXLOC Location of a maximum value in an array
- MAXVAL Maximum value in an array
- MINLOC Location of a minimum value in an array
- MINVAL Minimum value in an array
- PACK Pack an array into an array of rank one
- PRODUCT Product of array elements
- RESHAPE  Reshape an array
- SPREAD Replicates array by adding a dimension
- SUM Sum of array elements
- TRANSPOSE Transpose an array of rank two
- UNPACK Unpack an array of rank one into an array under a mask



- Examples

```fortran
program matrix
    real w(10),x(10),mat(10,10)
    call random_number(w)
    call random_number(mat)
    x=matmul(w,mat)   ! regular matrix multiply  USE IT
    write(*,'("dot(x,x)=",f10.5)'),dot_product(x,x)
end program
program allit
     character(len=10):: f1="(3l1)"
     character(len=10):: f2="(3i2)"
     integer b(2,3),c(2,3),one_d(6)
     logical l(2,3)
     one_d=(/ 1,3,5 , 2,4,6 /)
     b=transpose(reshape((/ 1,3,5 , 2,4,6 /),shape=(/3,2/)))
     C=transpose(reshape((/ 0,3,5 , 7,4,8 /),shape=(/3,2/)))
     l=(b.ne.c)
     write(*,f2)((b(i,j),j=1,3),i=1,2)
     write(*,*)
     write(*,f2)((c(i,j),j=1,3),i=1,2)
     write(*,*)
     write(*,f1)((l(i,j),j=1,3),i=1,2)
     write(*,*)
     write(*,f1)all ( b .ne. C ) !is .false.
     write(*,f1)all ( b .ne. C, DIM=1) !is [.true., .false., .false.]
     write(*,f1)all ( b .ne. C, DIM=2) !is [.false., .false.]
end
```

- The output is:

```
 1 3 5
 2 4 6
 0 3 5
 7 4 8
 TFF
 TFT
 F
 TFF
 FF

```

- - -
- - -

## The rest of our GA
- Includes
- Reproduction
- Mutation
- Nothing new in either of these files
- [Source and makefile "git"](source)
- [Source and makefile "*tgz"](https://github.com/timkphd/examples/raw/master/fort/90/source/archive.tgz)

- - -
- - -

## Compiler Information
### gfortran
* .f, .for, .ftn .f77
	* fixed-format Fortran source; compile
* .f90, .f95
	* free-format Fortran source; compile
* -fbacktrace
	*          Add debug information for runtime traceback
* -ffree-form -ffixed-form
	* source form
* -O0, -O1, -O2, -O3
	* optimization level
* .fpp, .FPP,  .F, .FOR, .FTN, .F90, .F95, .F03 or .F08
	* Fortran source file with preprocessor directives
* -fopenmp
	* turn on OpenMP
### Intel
* .f, .for, .ftn
	* fixed-format Fortran source; compile
* .f90, .f95
	* free-format Fortran source; compile
* -O0, -O1, -O2, -O3, -O4
	* optimization level
* .fpp, .F, .FOR, .FTN, .FPP, .F90
	* Fortran source file with preprocessor directives
* -g
	* compile for debug     * -traceback -notraceback (default)
	* Add debug information for runtime traceback
* -nofree, -free
	* Source is fixed or free format
* -fopenmp
	* turn on OpenMP
### Portland Group (x86)
*  .f, .for, .ftn
	*   fixed-format Fortran source; compile
*   .f90, .f95, .f03
	*   free-format Fortran source; compile
*    .cuf
	*    free-format CUDA Fortran source; compile
*   .CUF
	*   free-format CUDA Fortran source; preprocess, compile

*   -O0, -O1, -O2, -O3, -O4
	*   optimization level

*   -g
	*   compile for debug     	* 	 -traceback (default) -notraceback
	*          Add debug information for runtime traceback
*   -Mfixed, -Mfree
	*   Source is fixed or free format
*   -qmp
	*   turn on OpenMP
### IBM xlf
* xlf, xlf_r, f77, fort77
	* Compile FORTRAN 77 source files.  _r = thread safe
* xlf90, xlf90_r, f90
	* Compile Fortran 90 source files.  _r = thread safe
* xlf95, xlf95_r, f95
	* Compile Fortran 95 source files.  _r = thread safe
* xlf2003, xlf2003_r,f2003 	* Compile Fortran 2003 source files. _r = thread safe
* xlf2008, xlf2008_r, f2008  	* Compile Fortran 2008 source files.
* .f, .f77, .f90, .f95, .f03, .f08
	* Fortran source file
* .F, .F77, .F90, .F95,  .F03, .F08
	* Fortran source file with preprocessor directives
* -qtbtable=full
	* Add debug information for runtime traceback
* -qsmp=omp
	* turn on OpenMP
* -O0, -O1, -O2, -O3, -O4, O5
	* optimization level
* -g , g0, g1,...g9
	* compile for debug     
- - -
- - -


## Summary
- Fortran 90 has features to:
    - Enhance performance
    - Enhance portability
    - Enhance reliability
    - Enhance maintainability

- Fortran 90 has new language elements
    - Source form
    - Derived data types
    - Dynamic memory allocation functions
    - Kind facility for portability and easy modification
    - Many new intrinsic function
    - Array assignments

- Examples
    - Help show how things work
    - Reference for future use

### Introduction to Fortran Language
```
  Brought to you by ANSI committee X3J3 and ISO-IEC/JTC1/SC22/WG5 (Fortran)
  This is neither complete nor precisely accurate, but hopefully, after
  a small investment of time it is easy to read and very useful.

  This is the free form version of Fortran, no statement numbers,
  no C in column 1, start in column 1 (not column 7),
  typically indent 2, 3, or 4 spaces per each structure.
  The typical extension is  .f90  .

  Continue a statement on the next line by ending the previous line with
  an ampersand  &amp; .  Start the continuation with  &amp;  for strings.

  The rest of any line is a comment starting with an exclamation mark  ! .

  Put more than one statement per line by separating statements with a
  semicolon  ; . Null statements are OK, so lines can end with semicolons.

  Separate words with space or any form of "white space" or punctuation.
```

### Meta language used in this compact summary 

```
  <xxx> means fill in something appropriate for xxx and do not type
        the  "<"  or  ">" .

  ...  ellipsis means the usual, fill in something, one or more lines

  [stuff] means supply nothing or at most one copy of "stuff"
          [stuff1 [stuff2]] means if "stuff1" is included, supply nothing
          or at most one copy of stuff2.

  "old" means it is in the language, like almost every feature of past
  Fortran standards, but should not be used to write new programs.

```

### Structure of files that can be compiled 
```

  program <name>                  usually file name is  <name>.f90
    use <module_name>             bring in any needed modules
    implicit none                 good for error detection
    <declarations>
    <executable statements>       order is important, no more declarations
  end program <name>


  block data <name>               old
    <declarations>                common, dimension, equivalence now obsolete
  end block data <name>


  module <name>                   bring back in with   use <name>
    implicit none                 good for error detection
    <declarations>                can have private and public and interface
  end module <name>

  subroutine <name>               use:  call <name>   to execute
    implicit none                 good for error detection
    <declarations>
    <executable statements>
  end subroutine <name>


  subroutine <name>(par1, par2, ...) 
                                  use:  call <name>(arg1, arg2,... ) to execute
    implicit none                 optional, good for error detection
    <declarations>                par1, par2, ... are defined in declarations 
                                  and can be specified in, inout, pointer, etc.
    <executable statements>
    return                        optional, end causes automatic return
    entry <name> (par...)         old, optional other entries
  end subroutine <name>


  function <name>(par1, par2, ...) result(<rslt>)
                                  use: <name>(arg1, arg2, ... argn) as variable
    implicit none                 optional, good for error detection
    <declarations>                rslt, par1, ... are defined in declarations
    <executable statements>
    <rslt> = <expression>         required somewhere in execution
    [return]                      optional, end causes automatic return
  end function <name>

                                  old
  <type> function(...) <name>     use: <name>(arg1, arg2, ... argn) as variable
    <declarations>
    <executable statements>
    <name> = <expression>         required somewhere in execution
    [return]                      optional, end causes automatic return
  end function <name>

```

### Executable Statements and Constructs 
```

  <statement> will mean exactly one statement in this section

  a construct is multiple lines

  <label> : <statement>      any statement can have a label (a name)

  <variable> = <expression>  assignment statement

  <pointer>  >= <variable>   the pointer is now an alias for the variable
  <pointer1> >= <pointer2>    pointer1 now points same place as pointer2

  stop                       can be in any executable statement group,
  stop <integer>             terminates execution of the program,
  stop <string>              can have optional integer or string

  return                     exit from subroutine or function

  do <variable>=<from>,<to> [,<increment&gt]   optional:  <label> : do ...
     <statements>

     exit                                   \_optional   or exit <label&gt
     if (<boolean expression>) exit         /
                                            exit the loop
     cycle                                  \_optional   or cycle <label>
     if (<boolean expression>) cycle        /
                                            continue with next loop iteration
  end do                                    optional:    end do <name>


  do while (<boolean expression>)
     ...                                   optional exit and cycle allowed
  end do


  do
     ...                                   exit required to end the loop
                                           optional  cycle  can be used
  end do



  if ( <boolean expression> ) <statement>  execute the statement if the
                                           boolean expression is true

  if ( <boolean expression1> ) then
    ...                                    execute if expression1 is true
  else if ( <boolean expression2> ) then
    ...                                    execute if expression2 is true
  else if ( <boolean expression3> ) then
    ...                                    execute if expression3 is true
  else
    ...                                    execute if none above are true
  end if


  select case (<expression>)            optional <name> : select case ...
     case (<value>)
        <statements>                    execute if expression == value
     case (<value1>:<value2>)           
        <statements>                    execute if value1 &le; expression &le; value2
     ...
     case default
        <statements>                    execute if no values above match
  end select                            optional  end select <name>


  real, dimension(10,12) :: A, R     a sample declaration for use with "where"
    ...
  where (A /= 0.0)                   conditional assignment, only assignment allowed
     R = 1.0/A
  elsewhere
     R = 1.0                         elements of R set to 1.0 where A == 0.0
  end where

    go to <statement number>          old

    go to (<statement number list>), <expression>   old

    for I/O statements, see:  section 10.0  Input/Output Statements

    many old forms of statements are not listed
```

###  Declarations 
```

  There are five (5) basic types: integer, real, complex, character and logical.
  There may be any number of user derived types.  A modern (not old) declaration
  starts with a type, has attributes, then ::, then variable(s) names

  integer i, pivot, query                             old

  integer, intent (inout) :: arg1

  integer (selected_int_kind (5)) :: i1, i2

  integer, parameter :: m = 7

  integer, dimension(0:4, -5:5, 10:100) :: A3D

  double precision x                                 old

  real  (selected_real_kind(15,300) :: x

  complex :: z

  logical, parameter :: what_if = .true.

  character, parameter :: me = "Jon Squire"

  type <name>       a new user type, derived type
    declarations
  end type <name>

  type (<name>) :: stuff    declaring stuff to be of derived type <name>

  real, dimension(:,:), allocatable, target :: A

  real, dimension(:,:), pointer :: P

  Attributes may be:

    allocatable  no memory used here, allocate later
    dimension    vector or multi dimensional array
    external     will be defined outside this compilation
    intent       argument may be  in, inout or out
    intrinsic    declaring function to be an intrinsic
    optional     argument is optional
    parameter    declaring a constant, can not be changed later
    pointer      declaring a pointer
    private      in a module, a private declaration
    public       in a module, a public declaration
    save         keep value from one call to the next, static
    target       can be pointed to by a pointer
    Note:        not all combinations of attributes are legal
```

### Key words (other than I/O) 
```

  note: "statement" means key word that starts a statement, one line
                    unless there is a continuation "&amp;"
        "construct" means multiple lines, usually ending with "end ..."
        "attribute" means it is used in a statement to further define
        "old"       means it should not be used in new code

  allocatable          attribute, no space allocated here, later allocate
  allocate             statement, allocate memory space now for variable
  assign               statement, old, assigned go to
  assignment           attribute, means subroutine is assignment (=)
  block data           construct, old, compilation unit, replaced by module
  call                 statement, call a subroutine
  case                 statement, used in  select case structure
  character            statement, basic type, intrinsic data type
  common               statement, old, allowed overlaying of storage
  complex              statement, basic type, intrinsic data type
  contains             statement, internal subroutines and functions follow
  continue             statement, old, a place to put a statement number
  cycle                statement, continue the next iteration of a do loop
  data                 statement, old, initialized variables and arrays
  deallocate           statement, free up storage used by specified variable
  default              statement, in a select case structure, all others
  do                   construct, start a do loop
  double precision     statement, old, replaced by selected_real_kind(15,300)
  else                 construct, part of if   else if   else   end if
  else if              construct, part of if   else if   else   end if
  elsewhere            construct, part of where  elsewhere  end where
  end block data       construct, old, ends block data
  end do               construct, ends do
  end function         construct, ends function
  end if               construct, ends if
  end interface        construct, ends interface
  end module           construct, ends module
  end program          construct, ends program
  end select           construct, ends select case
  end subroutine       construct, ends subroutine
  end type             construct, ends type
  end where            construct, ends where
  entry                statement, old, another entry point in a procedure
  equivalence          statement, old, overlaid storage
  exit                 statement, continue execution outside of a do loop
  external             attribute, old statement, means defines else where
  function             construct, starts the definition of a function
  go to                statement, old, requires fixed form statement number
  if                   statement and construct, if(...) statement
  implicit             statement, "none" is preferred to help find errors
  in                   a keyword for intent, the argument is read only
  inout                a keyword for intent, the argument is read/write
  integer              statement, basic type, intrinsic data type
  intent               attribute, intent(in) or intent(out) or intent(inout)
  interface            construct, begins an interface definition
  intrinsic            statement, says that following names are intrinsic
  kind                 attribute, sets the kind of the following variables
  len                  attribute, sets the length of a character string
  logical              statement, basic type, intrinsic data type
  module               construct, beginning of a module definition
  namelist             statement, defines a namelist of input/output
  nullify              statement, nullify(some_pointer) now points nowhere
  only                 attribute, restrict what comes from a module
  operator             attribute, indicates function is an operator, like +
  optional             attribute, a parameter or argument is optional
  out                  a keyword for intent, the argument will be written
  parameter            attribute, old statement, makes variable real only
  pause                old, replaced by stop
  pointer              attribute, defined the variable as a pointer alias
  private              statement and attribute, in a module, visible inside
  program              construct, start of a main program
  public               statement and attribute, in a module, visible outside
  real                 statement, basic type, intrinsic data type
  recursive            attribute, allows functions and derived type recursion
  result               attribute, allows naming of function result  result(Y)
  return               statement, returns from, exits, subroutine or function
  save                 attribute, old statement, keep value between calls
  select case          construct, start of a case construct
  stop                 statement, terminate execution of the main procedure
  subroutine           construct, start of a subroutine definition
  target               attribute, allows a variable to take a pointer alias
  then                 part of if construct
  type                 construct, start of user defined type
  type ( )             statement, declaration of a variable for a users type
  use                  statement, brings in a module
  where                construct, conditional assignment
  while                construct, a while form of a do loop
```

### Key words related to I/O 
```

  backspace            statement, back up one record
  close                statement, close a file
  endfile              statement, mark the end of a file
  format               statement, old, defines a format
  inquire              statement, get the status of a unit
  open                 statement, open or create a file
  print                statement, performs output to screen
  read                 statement, performs input
  rewind               statement, move read or write position to beginning
  write                statement, performs output

```

### Operators 
```

  **    exponentiation
  *     multiplication
  /     division
  +     addition
  -     subtraction
  //    concatenation
  ==    .eq.  equality
  /=    .ne.  not equal
  <     .lt.  less than
  >     .gt.  greater than
  <=    .le.  less than or equal
  >=    .ge.  greater than or equal
  .not.       complement, negation
  .and.       logical and
  .or.        logical or
  .eqv.       logical equivalence
  .neqv.      logical not equivalence, exclusive or

  .eq.  ==    equality, old
  .ne.  /=    not equal. old
  .lt.  <     less than, old
  .gt.  >     greater than, old
  .le.  <=    less than or equal, old
  .ge.  >=    greater than or equal, old


  Other punctuation:

   /  ...  /  used in data, common, namelist and other statements
   (/ ... /)  array constructor, data is separated by commas
   6*1.0      in some contexts, 6 copies of 1.0
   (i:j:k)    in some contexts, a list  i, i+k, i+2k, i+3k, ... i+nk&le;j
   (:j)       j and all below
   (i:)       i and all above
   (:)        undefined or all in range

```

### Constants 
```

  Logical constants:

    .true.      True
    .false.     False

  Integer constants:

     0    1     -1     123456789

  Real constants:

     0.0   1.0   -1.0    123.456   7.1E+10   -52.715E-30

  Complex constants:

     (0.0, 0.0)    (-123.456E+30, 987.654E-29)

  Character constants:

      "ABC"   "a"  "123'abc$%#@!"    " a quote "" "
      'ABC'   'a'  '123"abc$%#@!'    ' a apostrophe '' '

  Derived type values:

      type name
        character (len=30) :: last
        character (len=30) :: first
        character (len=30) :: middle
      end type name

      type address
        character (len=40) :: street
        character (len=40) :: more
        character (len=20) :: city
        character (len=2)  :: state
        integer (selected_int_kind(5)) :: zip_code
        integer (selected_int_kind(4)) :: route_code
      end type address

      type person
        type (name) lfm
        type (address) snail_mail
      end type person

      type (person) :: a_person = person( name("Squire","Jon","S."), &amp;
          address("106 Regency Circle", "", "Linthicum", "MD", 21090, 1936))

      a_person%snail_mail%route_code == 1936

```

### Input/Output Statements 
```

    open (<unit number>)
    open (unit=<unit number>, file=<file name>, iostat=<variable>)
    open (unit=<unit number>, ... many more, see below )

    close (<unit number>)
    close (unit=<unit number>, iostat=<variable>,
           err=<statement number>, status="KEEP")

    read (<unit number>) <input list>
    read (unit=<unit number>, fmt=<format>, iostat=<variable>,
          end=<statement number>, err=<statement number>) <input list>
    read (unit=<unit number>, rec=<record number>) <input list>

    write (<unit number>) <output list>
    write (unit=<unit number>, fmt=<format>, iostat=<variable>,
           err=<statement number>) <output list>
    write (unit=<unit number>, rec=<record number>) <output list>

    print *, <output list>

    print "(<your format here, use apostrophe, not quote>)", <output list>

    rewind <unit number>
    rewind (<unit number>, err=<statement number>)

    backspace <unit number>
    backspace (<unit number>, iostat=<variable>)

    endfile <unit number>
    endfile (<unit number>, err=<statement number>, iostat=<variable>)

    inquire ( <unit number>, exists = <variable>)
    inquire ( file=<"name">, opened = <variable1>, access = <variable2> )
    inquire ( iolength = <variable> ) x, y, A   ! gives "recl" for "open"

    namelist /<name>/ <variable list>      defines a name list
    read(*,nml=<name>)                     reads some/all variables in namelist
    write(*,nml=<name>)                    writes all variables in namelist
    &amp;<name> <variable>=<value> ... <variable=value> /  data for namelist read

  Input / Output specifiers

    access   one of  "sequential"  "direct"  "undefined"
    action   one of  "read"  "write"  "readwrite"
    advance  one of  "yes"  "no"  
    blank    one of  "null"  "zero"
    delim    one of  "apostrophe"  "quote"  "none"
    end      =       <integer statement number>  old
    eor      =       <integer statement number>  old
    err      =       <integer statement number>  old
    exist    =       <logical variable>
    file     =       <"file name">
    fmt      =       <"(format)"> or <character variable> format
    form     one of  "formatted"  "unformatted"  "undefined"
    iolength =       <integer variable, size of unformatted record>
    iostat   =       <integer variable> 0==good, negative==eof, positive==bad
    name     =       <character variable for file name>
    named    =       <logical variable>
    nml      =       <namelist name>
    nextrec  =       <integer variable>    one greater than written
    number   =       <integer variable unit number>
    opened   =       <logical variable>
    pad      one of  "yes"  "no"
    position one of  "asis"  "rewind"  "append"
    rec      =       <integer record number>
    recl     =       <integer unformatted record size>
    size     =       <integer variable>  number of characters read before eor
    status   one of  "old"  "new"  "unknown"  "replace"  "scratch"  "keep"
    unit     =       <integer unit number>

  Individual questions
    direct      =    <character variable>  "yes"  "no"  "unknown"
    formatted   =    <character variable>  "yes"  "no"  "unknown"
    read        =    <character variable>  "yes"  "no"  "unknown"
    readwrite   =    <character variable>  "yes"  "no"  "unknown"
    sequential  =    <character variable>  "yes"  "no"  "unknown"
    unformatted =    <character variable>  "yes"  "no"  "unknown"
    write       =    <character variable>  "yes"  "no"  "unknown"

```

### Formats 
```

    format                    an explicit format can replace * in any
                              I/O statement. Include the format in
                              apostrophes or quotes and keep the parenthesis.

    examples:
         print "(3I5,/(2X,3F7.2/))", <output list>
         write(6, '(a,E15.6E3/a,G15.2)' ) <output list>
         read(unit=11, fmt="(i4, 4(f3.0,TR1))" ) <input list>
                             
    A format includes the opening and closing parenthesis.
    A format consists of format items and format control items separated by comma.
    A format may contain grouping parenthesis with an optional repeat count.

  Format Items, data edit descriptors:

    key:  w  is the total width of the field   (filled with *** if overflow)
          m  is the least number of digits in the (sub)field (optional)
          d  is the number of decimal digits in the field
          e  is the number of decimal digits in the exponent subfield
          c  is the repeat count for the format item
          n  is number of columns

    cAw     data of type character (w is optional)
    cBw.m   data of type integer with binary base
    cDw.d   data of type real -- same as E,  old double precision
    cEw.d   or Ew.dEe  data of type real
    cENw.d  or ENw.dEe  data of type real  -- exponent a multiple of 3
    cESw.d  or ESw.dEe  data of type real  -- first digit non zero
    cFw.d   data of type real  -- no exponent printed
    cGw.d   or Gw.dEe  data of type real  -- auto format to F or E
    nH      n characters follow the H,  no list item
    cIw.m   data of type integer
    cLw     data of type logical  --  .true.  or  .false.
    cOw.m   data of type integer with octal base
    cZw.m   data of type integer with hexadecimal base
    "<string>"  literal characters to output, no list item
    '<string>'  literal characters to output, no list item

  Format Control Items, control edit descriptors:

    BN      ignore non leading blanks in numeric fields
    BZ      treat nonleading blanks in numeric fields as zeros
    nP      apply scale factor to real format items   old
    S       printing of optional plus signs is processor dependent
    SP      print optional plus signs
    SS      do not print optional plus signs
    Tn      tab to specified column
    TLn     tab left n columns
    TRn     tab right n columns
    nX      tab right n columns
    /       end of record (implied / at end of all format statements)
    :       stop format processing if no more list items

  <input list> can be:
    a variable
    an array name
    an implied do   ((A(i,j),j=1,n) ,i=1,m)    parenthesis and commas as shown

    note: when there are more items in the input list than format items, the
          repeat rules for formats applies.

  <output list> can be:
    a constant
    a variable
    an expression
    an array name
    an implied do   ((A(i,j),j=1,n) ,i=1,m)    parenthesis and commas as shown

    note: when there are more items in the output list than format items, the
          repeat rules for formats applies.

  Repeat Rules for Formats:

    Each format item is used with a list item.  They are used in order.
    When there are more list items than format items, then the following
    rule applies:  There is an implied end of record, /, at the closing
    parenthesis of the format, this is processed.  Scan the format backwards
    to the first left parenthesis.  Use the repeat count, if any, in front
    of this parenthesis, continue to process format items and list items.

    Note: an infinite loop is possible
          print "(3I5/(1X/))", I, J, K, L    may never stop

```

### Intrinsic Functions 

```

  Intrinsic Functions are presented in alphabetical order and then grouped
  by topic.  The function name appears first. The argument(s) and result
  give an indication of the type(s) of argument(s) and results.
  [,dim=] indicates an optional argument  "dim".
  "mask" must be logical and usually conformable.
  "character" and "string" are used interchangeably.
  A brief description or additional information may appear.
```


####  Intrinsic Functions (alphabetical):

```
    abs(integer_real_complex) result(integer_real_complex)
    achar(integer) result(character)  integer to character
    acos(real) result(real)  arccosine  |real| &le; 1.0   0&le;result&le;Pi
    adjustl(character)  result(character) left adjust, blanks go to back
    adjustr(character)  result(character) right adjust, blanks to front
    aimag(complex) result(real)  imaginary part
    aint(real [,kind=]) result(real)  truncate to integer toward zero
    all(mask [,dim]) result(logical)  true if all elements of mask are true
    allocated(array) result(logical)  true if array is allocated in memory
    anint(real [,kind=]) result(real)  round to nearest integer
    any(mask [,dim=}) result(logical)  true if any elements of mask are true
    asin(real) result(real)  arcsine  |real| &le; 1.0   -Pi/2&le;result&le;Pi/2
    associated(pointer [,target=]) result(logical)  true if pointing
    atan(real) result(real)  arctangent  -Pi/2&le;result&le;Pi/2 
    atan2(y=real,x=real) result(real)  arctangent  -Pi&le;result&le;Pi
    bit_size(integer) result(integer)  size in bits in model of argument
    btest(i=integer,pos=integer) result(logical)  true if pos has a 1, pos=0..
    ceiling(real) result(real)  truncate to integer toward infinity
    char(integer [,kind=]) result(character)  integer to character [of kind]
    cmplx(x=real [,y=real] [kind=]) result(complex)  x+iy
    conjg(complex) result(complex)  reverse the sign of the imaginary part
    cos(real_complex) result(real_complex)  cosine
    cosh(real) result(real)  hyperbolic cosine
    count(mask [,dim=]) result(integer)  count of true entries in mask
    cshift(array,shift [,dim=]) circular shift elements of array, + is right
    date_and_time([date=] [,time=] [,zone=] [,values=])  y,m,d,utc,h,m,s,milli
    dble(integer_real_complex) result(real_kind_double)  convert to double
    digits(integer_real) result(integer)  number of bits to represent model
    dim(x=integer_real,y=integer_real) result(integer_real) proper subtraction
    dot_product(vector_a,vector_b) result(integer_real_complex) inner product
    dprod(x=real,y=real) result(x_times_y_double)  double precision product
    eoshift(array,shift [,boundary=] [,dim=])  end-off shift using boundary
    epsilon(real) result(real)  smallest positive number added to 1.0 /= 1.0
    exp(real_complex) result(real_complex)  e raised to a power
    exponent(real) result(integer)  the model exponent of the argument
    floor(real) result(real)  truncate to integer towards negative infinity
    fraction(real) result(real)  the model fractional part of the argument
    huge(integer_real) result(integer_real)  the largest model number
    iachar(character) result(integer)  position of character in ASCII sequence
    iand(integer,integer) result(integer)  bit by bit logical and
    ibclr(integer,pos) result(integer)  argument with pos bit cleared to zero
    ibits(integer,pos,len) result(integer)  extract len bits starting at pos
    ibset(integer,pos) result(integer)  argument with pos bit set to one
    ichar(character) result(integer)  pos in collating sequence of character
    ieor(integer,integer) result(integer)  bit by bit logical exclusive or
    index(string,substring [,back=])  result(integer)  pos of substring
    int(integer_real_complex) result(integer)  convert to integer
    ior(integer,integer) result(integer)  bit by bit logical or
    ishft(integer,shift) result(integer)  shift bits in argument by shift
    ishftc(integer, shift) result(integer)  shift circular bits in argument
    kind(any_intrinsic_type) result(integer)  value of the kind
    lbound(array,dim) result(integer)  smallest subscript of dim in array
    len(character) result(integer)  number of characters that can be in argument
    len_trim(character) result(integer)  length without trailing blanks
    lge(string_a,string_b) result(logical)  string_a &ge; string_b
    lgt(string_a,string_b) result(logical)  string_a > string_b
    lle(string_a,string_b) result(logical)  string_a &le; string_b
    llt(string_a,string_b) result(logical)  string_a < string_b
    log(real_complex) result(real_complex)  natural logarithm
    log10(real) result(real)  logarithm base 10
    logical(logical [,kind=])  convert to logical
    matmul(matrix,matrix) result(vector_matrix)  on integer_real_complex_logical
    max(a1,a2,a3,...) result(integer_real)  maximum of list of values
    maxexponent(real) result(integer)  maximum exponent of model type
    maxloc(array [,mask=]) result(integer_vector)  indices in array of maximum
    maxval(array [,dim=] [,mask=])  result(array_element)  maximum value
    merge(true_source,false_source,mask) result(source_type)  choose by mask
    min(a1,a2,a3,...) result(integer-real)  minimum of list of values
    minexponent(real) result(integer)  minimum(negative) exponent of model type
    minloc(array [,mask=]) result(integer_vector)  indices in array of minimum
    minval(array [,dim=] [,mask=])  result(array_element)  minimum value
    mod(a=integer_real,p) result(integer_real)  a modulo p
    modulo(a=integer_real,p) result(integer_real)  a modulo p
    mvbits(from,frompos,len,to,topos) result(integer)  move bits
    nearest(real,direction) result(real)  nearest value toward direction
    nint(real [,kind=]) result(real)  round to nearest integer value
    not(integer) result(integer)  bit by bit logical complement
    pack(array,mask [,vector=]) result(vector)  vector of elements from array
    present(argument) result(logical)  true if optional argument is supplied
    product(array [,dim=] [,mask=]) result(integer_real_complex)  product
    radix(integer_real) result(integer)  radix of integer or real model, 2
    random_number(harvest=real_out)  subroutine, uniform random number 0 to 1
    random_seed([size=] [,put=] [,get=])  subroutine to set random number seed
    range(integer_real_complex) result(integer_real)  decimal exponent of model
    real(integer_real_complex [,kind=]) result(real)  convert to real
    repeat(string,ncopies) result(string)  concatenate n copies of string
    reshape(source,shape,pad,order) result(array)  reshape source to array
    rrspacing(real) result(real)  reciprocal of relative spacing of model
    scale(real,integer) result(real)  multiply by  2**integer
    scan(string,set [,back]) result(integer)  position of first of set in string
    selected_int_kind(integer) result(integer)  kind number to represent digits
    selected_real_kind(integer,integer) result(integer)  kind of digits, exp
    set_exponent(real,integer) result(real)  put integer as exponent of real
    shape(array) result(integer_vector)  vector of dimension sizes
    sign(integer_real,integer_real) result(integer_real) sign of second on first
    sin(real_complex) result(real_complex)  sine of angle in radians
    sinh(real) result(real)  hyperbolic sine of argument
    size(array [,dim=]) result(integer)  number of elements in dimension
    spacing(real) result(real)  spacing of model numbers near argument
    spread(source,dim,ncopies) result(array)  expand dimension of source by 1
    sqrt(real_complex) result(real_complex)  square root of argument
    sum(array [,dim=] [,mask=]) result(integer_real_complex)  sum of elements
    system_clock([count=] [,count_rate=] [,count_max=])  subroutine, all out
    tan(real) result(real)  tangent of angle in radians
    tanh(real) result(real)  hyperbolic tangent of angle in radians
    tiny(real) result(real)  smallest positive model representation
    transfer(source,mold [,size]) result(mold_type)  same bits, new type
    transpose(matrix) result(matrix)  the transpose of a matrix
    trim(string) result(string)  trailing blanks are removed
    ubound(array,dim) result(integer)  largest subscript of dim in array
    unpack(vector,mask,field) result(v_type,mask_shape)  field when not mask
    verify(string,set [,back]) result(integer)  pos in string not in set
```



####  Intrinsic Functions (grouped by topic):

#####  Intrinsic Functions (Numeric)
```
    abs(integer_real_complex) result(integer_real_complex)
    acos(real) result(real)  arccosine  |real| &le; 1.0   0&le;result&le;Pi
    aimag(complex) result(real)  imaginary part
    aint(real [,kind=]) result(real)  truncate to integer toward zero
    anint(real [,kind=]) result(real)  round to nearest integer
    asin(real) result(real)  arcsine  |real| &le; 1.0   -Pi/2&le;result&le;Pi/2
    atan(real) result(real)  arctangent  -Pi/2&le;result&le;Pi/2 
    atan2(y=real,x=real) result(real)  arctangent  -Pi&le;result&le;Pi
    ceiling(real) result(real)  truncate to integer toward infinity
    cmplx(x=real [,y=real] [kind=]) result(complex)  x+iy
    conjg(complex) result(complex)  reverse the sign of the imaginary part
    cos(real_complex) result(real_complex)  cosine
    cosh(real) result(real)  hyperbolic cosine
    dble(integer_real_complex) result(real_kind_double)  convert to double
    digits(integer_real) result(integer)  number of bits to represent model
    dim(x=integer_real,y=integer_real) result(integer_real) proper subtraction
    dot_product(vector_a,vector_b) result(integer_real_complex) inner product
    dprod(x=real,y=real) result(x_times_y_double)  double precision product
    epsilon(real) result(real)  smallest positive number added to 1.0 /= 1.0
    exp(real_complex) result(real_complex)  e raised to a power
    exponent(real) result(integer)  the model exponent of the argument
    floor(real) result(real)  truncate to integer towards negative infinity
    fraction(real) result(real)  the model fractional part of the argument
    huge(integer_real) result(integer_real)  the largest model number
    int(integer_real_complex) result(integer)  convert to integer
    log(real_complex) result(real_complex)  natural logarithm
    log10(real) result(real)  logarithm base 10
    matmul(matrix,matrix) result(vector_matrix)  on integer_real_complex_logical
    max(a1,a2,a3,...) result(integer_real)  maximum of list of values
    maxexponent(real) result(integer)  maximum exponent of model type
    maxloc(array [,mask=]) result(integer_vector)  indices in array of maximum
    maxval(array [,dim=] [,mask=])  result(array_element)  maximum value
    min(a1,a2,a3,...) result(integer-real)  minimum of list of values
    minexponent(real) result(integer)  minimum(negative) exponent of model type
    minloc(array [,mask=]) result(integer_vector)  indices in array of minimum
    minval(array [,dim=] [,mask=])  result(array_element)  minimum value
    mod(a=integer_real,p) result(integer_real)  a modulo p
    modulo(a=integer_real,p) result(integer_real)  a modulo p
    nearest(real,direction) result(real)  nearest value toward direction
    nint(real [,kind=]) result(real)  round to nearest integer value
    product(array [,dim=] [,mask=]) result(integer_real_complex)  product
    radix(integer_real) result(integer)  radix of integer or real model, 2
    random_number(harvest=real_out)  subroutine, uniform random number 0 to 1
    random_seed([size=] [,put=] [,get=])  subroutine to set random number seed
    range(integer_real_complex) result(integer_real)  decimal exponent of model
    real(integer_real_complex [,kind=]) result(real)  convert to real
    rrspacing(real) result(real)  reciprocal of relative spacing of model
    scale(real,integer) result(real)  multiply by  2**integer
    set_exponent(real,integer) result(real)  put integer as exponent of real
    sign(integer_real,integer_real) result(integer_real) sign of second on first
    sin(real_complex) result(real_complex)  sine of angle in radians
    sinh(real) result(real)  hyperbolic sine of argument
    spacing(real) result(real)  spacing of model numbers near argument
    sqrt(real_complex) result(real_complex)  square root of argument
    sum(array [,dim=] [,mask=]) result(integer_real_complex)  sum of elements
    tan(real) result(real)  tangent of angle in radians
    tanh(real) result(real)  hyperbolic tangent of angle in radians
    tiny(real) result(real)  smallest positive model representation
    transpose(matrix) result(matrix)  the transpose of a matrix
```


#####  Intrinsic Functions (Logical and bit)

```
    all(mask [,dim]) result(logical)  true if all elements of mask are true
    any(mask [,dim=}) result(logical)  true if any elements of mask are true
    bit_size(integer) result(integer)  size in bits in model of argument
    btest(i=integer,pos=integer) result(logical)  true if pos has a 1, pos=0..
    count(mask [,dim=]) result(integer)  count of true entries in mask
    iand(integer,integer) result(integer)  bit by bit logical and
    ibclr(integer,pos) result(integer)  argument with pos bit cleared to zero
    ibits(integer,pos,len) result(integer)  extract len bits starting at pos
    ibset(integer,pos) result(integer)  argument with pos bit set to one
    ieor(integer,integer) result(integer)  bit by bit logical exclusive or
    ior(integer,integer) result(integer)  bit by bit logical or
    ishft(integer,shift) result(integer)  shift bits in argument by shift
    ishftc(integer, shift) result(integer)  shift circular bits in argument
    logical(logical [,kind=])  convert to logical
    matmul(matrix,matrix) result(vector_matrix)  on integer_real_complex_logical
    merge(true_source,false_source,mask) result(source_type)  choose by mask
    mvbits(from,frompos,len,to,topos) result(integer)  move bits
    not(integer) result(integer)  bit by bit logical complement
    transfer(source,mold [,size]) result(mold_type)  same bits, new type
```



#####  Intrinsic Functions (Character or string)

```
    achar(integer) result(character)  integer to character
    adjustl(character)  result(character) left adjust, blanks go to back
    adjustr(character)  result(character) right adjust, blanks to front
    char(integer [,kind=]) result(character)  integer to character [of kind]
    iachar(character) result(integer)  position of character in ASCII sequence
    ichar(character) result(integer)  pos in collating sequence of character
    index(string,substring [,back=])  result(integer)  pos of substring
    len(character) result(integer)  number of characters that can be in argument
    len_trim(character) result(integer)  length without trailing blanks
    lge(string_a,string_b) result(logical)  string_a &ge; string_b
    lgt(string_a,string_b) result(logical)  string_a > string_b
    lle(string_a,string_b) result(logical)  string_a &le; string_b
    llt(string_a,string_b) result(logical)  string_a < string_b
    repeat(string,ncopies) result(string)  concatenate n copies of string
    scan(string,set [,back]) result(integer)  position of first of set in string
    trim(string) result(string)  trailing blanks are removed
    verify(string,set [,back]) result(integer)  pos in string not in set

```


### Fortran 95
- New Features
    -  The statement **FORALL** as an alternative to the DO-statement
    -  Partial nesting of FORALL and WHERE statements
    -  Masked ELSEWHERE
    -  Pure procedures
    -  Elemental procedures
    -  Pure procedures in specification expressions
    -  Revised MINLOC and MAXLOC
    -  Extensions to CEILING and FLOOR with the KIND keyword argument
    -  Pointer initialization
    -  Default initialization of derived type objects
    -  Increased compatibility with IEEE arithmetic
    -  A CPU_TIME intrinsic subroutine
    -  A function NULL to nullify a pointer
    -  **Automatic deallocation of allocatable arrays at exit of scoping unit**
    -  Comments in NAMELIST at input
    -  Minimal field at input
    -  Complete version of END INTERFACE
- Deleted Features
    -  real and double precision DO loop index variables
    -  branching to END IF from an outer block
    -  PAUSE statements
    -  ASSIGN statements and assigned GO TO statements and the use of an assigned
        integer as a FORMAT specification
    -  Hollerith editing in FORMAT
    -   See [http://www.nsc.liu.se/~boein/f77to90/f95.html#17.5](http://www.nsc.liu.se/~boein/f77to90/f95.html#17.5)
    
- - -
- - -


## References
- [http://www.fortran.com/fortran/](http://www.fortran.com/fortran/) Pointer to everything Fortran
- [http://meteora.ucsd.edu/~pierce/fxdr_home_page.html](http://meteora.ucsd.edu/~pierce/fxdr_home_page.html) Subroutines to do unformatted I/O across platforms, provided by David Pierce at UCSD
- [http://www.nsc.liu.se/~boein/f77to90/a5.html](http://www.nsc.liu.se/~boein/f77to90/a5.html) A good reference for intrinsic functions
- [https://wg5-fortran.org/N1551-N1600/N1579.pdf](https://wg5-fortran.org/N1551-N1600/N1579.pdf)New Features of Fortran 2003
- [https://wg5-fortran.org/N1701-N1750/N1729.pdf](https://wg5-fortran.org/N1701-N1750/N1729.pdf)New Features of Fortran 2008
- [http://www.nsc.liu.se/~boein/f77to90/](http://www.nsc.liu.se/~boein/f77to90/) Fortran 90 for the Fortran 77 Programmer
- <b>Fortran 90 Handbook Complete ANSI/ISO Reference</b>. Jeanne Adams, Walt Brainerd, Jeanne Martin, Brian Smith, Jerrold Wagener
- <b>Fortran 90 Programming</b>. T. Ellis, Ivor Philips, Thomas Lahey
- [https://github.com/llvm/llvm-project/blob/master/flang/docs/FortranForCProgrammers.md](https://github.com/llvm/llvm-project/blob/master/flang/docs/FortranForCProgrammers.md)
- [FFT stuff](../mkl/)
- [Fortran 95 and beyond](../95/)

<!-- 
- [French Translation provided by Mary Orban](http://www.pkwteile.ch/science/avancee-fortran-90/)
- [Belorussian translation](http://webhostingrating.com/libs/pashyrany-fortran-90-be) provided by [webhostingrating.com](http://webhostingrating.com/)
- [http://www.dmoz.org/Computers/Programming/Languages/Fortran/FAQs,_Help,_and_Tutorials/Fortran_90_and_95/](http://www.dmoz.org/Computers/Programming/Languages/Fortran/FAQs,_Help,_and_Tutorials/Fortran_90_and_95/) Cray's Manual
- [http://www.nova.edu/ocean/psplot.html](http://www.nova.edu/ocean/psplot.html) Postscript plotting library
- [http://www.fortran.com/fortran_storenew/Html/Info/books/key10.pdf](http://www.fortran.com/fortran_storenew/Html/Info/books/key10.pdf)Key Features of Fortran 90
-->

- - -
- - -
