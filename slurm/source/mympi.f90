!!!!!! module mympi
!!!!!! To use this file to compile fhostone.f90
!!!!!! as a pure Openmp program
!!!!!!     ln -s mympi.f90 mpif.h
!!!!!!     gfortran -fopenmp -fallow-argument-mismatch -w -lm fhostone.f90
!!!!!!     unlink mpif.h
    integer,parameter :: MPI_MAX_LIBRARY_VERSION_STRING=4
    integer,parameter :: MPI_MAX_PROCESSOR_NAME=16
    integer,parameter :: MPI_STATUS_SIZE=1
    integer,parameter :: MPI_COMM_WORLD=0
    integer,parameter :: MPI_INTEGER=0
    integer,parameter :: MPI_CHARACTER=0
    integer,parameter :: MPI_Double_precision=0

    INTERFACE  MPI_Bcast
        module procedure MPI_Bcasti
        module procedure MPI_Bcastr
    END INTERFACE 

    contains
    function mpi_wtime() 
        IMPLICIT NONE
        double precision MPI_Wtime,omp_get_wtime
        mpi_wtime=omp_get_wtime()
    end function

    subroutine MPI_Get_library_version(version, vlan,ierr)
        IMPLICIT NONE
        character(len=MPI_MAX_LIBRARY_VERSION_STRING+1)::version
        integer vlan,ierr
        write(version,"(a)")"none"
        vlan=4
        ierr=0
    end subroutine
    
    subroutine MPI_Comm_size(c,numprocs,ierr)
        IMPLICIT NONE
        integer c,numprocs,ierr
        numprocs=1
        ierr=0
    end subroutine
    
    subroutine MPI_Comm_rank(c,rank,ierr)
        IMPLICIT NONE
        integer c,rank,ierr
        rank=0
        ierr=0
    end subroutine
    
    subroutine MPI_Barrier( c,ierr)
        IMPLICIT NONE
        integer c,ierr
        ierr=0
    end subroutine

    subroutine MPI_Finalize(ierr)
        IMPLICIT NONE
        integer ierr
        ierr=0
    end subroutine
    
    subroutine MPI_Send(s , c,  t,  f,  tg,  com ,ierr)
        IMPLICIT NONE
        character(len=*) ::s
        integer c,t,f,tg,com,ierr
        ierr=0
    end subroutine
    
    subroutine MPI_Recv(s , c,  t,  f,  tg,  com ,stat,ierr)
        IMPLICIT NONE
        character(len=*) ::s
        integer c,t,f,tg,com,ierr
        integer stat(1)
        ierr=0
    end subroutine

    subroutine MPI_Bcasti(r,  c,  t,  f,  com,ierr)
        IMPLICIT NONE
        integer r
        integer c,t,f,com,ierr
        ierr=0
    end subroutine

    subroutine MPI_Bcastr(r,  c,  t,  f,  com,ierr)
        IMPLICIT NONE
        double precision r
        integer c,t,f,com,ierr
        ierr=0
    end subroutine

    subroutine MPI_Init(ierr)
        IMPLICIT NONE
        integer ierr
        ierr=0
    end subroutine

    subroutine MPI_Comm_split( oc,  mycolor,  myid,  node_comm,ierr)
        IMPLICIT NONE
        integer oc,mycolor,myid,node_comm
        integer ierr
        node_comm=0
        ierr=0
    end subroutine

    subroutine MPI_Get_processor_name(lname, resultlen,ierr)
        IMPLICIT NONE
        integer resultlen,ierr
        character(len=128)::name
        character(len=MPI_MAX_PROCESSOR_NAME+1)::lname
        ierr= hostnm( name )
        resultlen=LEN_TRIM(name)
        if ( resultlen .le. MPI_MAX_PROCESSOR_NAME)then
            write(lname,"(a)")trim(name)
            ierr=0
        else
            write(lname,"(a)")"too small"
            ierr=1
        lname=adjustr(lname)
        endif
    end subroutine

!!!!!! end module
