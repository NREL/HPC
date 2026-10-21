!*********************************************
!  This is a simple hello world program. Each processor 
!  prints out its name, rank and  number of processors 
!  in the current MPI run. 
!*********************************************
      program hello
      use iso_fortran_env
      include "mpif.h"
      integer myid,numprocs,ierr,nlength
      character(len=MPI_MAX_LIBRARY_VERSION_STRING+1) :: version
      character (len=MPI_MAX_PROCESSOR_NAME+1):: myname
      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      if (myid .eq. 0)then
          call MPI_Get_library_version(version, nlength, ierr)
          write(*,*)trim(version)
          write(*,*)"compiler: ",compiler_version()
          write(*,*)              
          ! call printenv()
      endif
      call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
      call MPI_Get_processor_name(myname,nlength,ierr)
      write (*,*) "Hello from ",trim(myname)," (F) ",myid," of ",numprocs
      if(numprocs .gt. 1)call pass(myid,numprocs)
      if (myid .eq. 0)write(*,*)"SUCCESS"
      call waste(1.0d0)
      call MPI_FINALIZE(ierr)
      stop
      end
      subroutine pass(myid,numprocs)
              implicit none
              include "mpif.h"
              integer myid,numprocs
              integer status(MPI_STATUS_SIZE)
              integer my_tag,to,from,i,ierr
              my_tag=1234
              i=myid
              to=myid+1
              from=myid-1
              call mpi_bcast(i,1,mpi_integer,0, MPI_COMM_WORLD,ierr)
              if( i .ne. 0)then
                  write(*,*)"bcast failed",myid
                  call MPI_Abort(MPI_COMM_WORLD,-1,ierr)
              endif
              if (myid .eq. 0)then
                      from=numprocs-1
                      i=1
                      call MPI_Send(i,1,MPI_INTEGER,to,my_tag,MPI_COMM_WORLD,ierr)
                      call chkerr(ierr,myid,"send")
                      call MPI_Recv(i,1,MPI_INTEGER,from,my_tag,MPI_COMM_WORLD,status,ierr)
                      call chkerr(ierr,myid,"recv")
                      if(i .ne. numprocs)then
                          write(*,*)"send / recv failed",i
                          call MPI_Abort(MPI_COMM_WORLD,-2,ierr)
                      endif
                      return
              endif
              if (myid .eq. numprocs-1)then
                      to=0
              endif
              call MPI_Recv(i,1,MPI_INTEGER,from,my_tag,MPI_COMM_WORLD,status,ierr)
              call chkerr(ierr,myid,"recv")
              i=i+1
              call MPI_Send(i,1,MPI_INTEGER,to,my_tag,MPI_COMM_WORLD,ierr)
              call chkerr(ierr,myid,"send")
      end subroutine

subroutine chkerr(ierr,myid,routine)
      include "mpif.h"
      integer ierr,myid
      character (len=*)  routine
      if (ierr .ne. 0)then
         write(*,*)routine," failed on ",myid
         call MPI_Abort(MPI_COMM_WORLD,-3,ierr)
      endif
end subroutine
         
subroutine waste(t)
    implicit none
    include "mpif.h"
    double precision t
    double precision t1
    integer , allocatable:: avect(:)
    integer i,j,msize
    msize=100000;
    t1=MPI_Wtime()+t;
    do while (MPI_Wtime() < t1)
        allocate(avect(msize))
        do i=1,msize
            avect(i)=i
        enddo
        j=0
        do i=1,msize
            j=j+avect(i)
        enddo
        if (j == 0)write(*,*)"nope"
        deallocate(avect)
    enddo
end subroutine
