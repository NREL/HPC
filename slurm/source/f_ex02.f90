! Program shows how to use probe and get_count to find the size
! of an incomming message
      program hello

      include "mpif.h"
 
      integer myid,numprocs
      integer status(MPI_STATUS_SIZE)
      integer mytag,icount,ierr,j,nlength
      integer,allocatable :: i(:)
      character (len=MPI_MAX_PROCESSOR_NAME):: myname 
      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
      call MPI_Get_processor_name(myname,nlength,ierr)
      write (*,1234)myid,numprocs,trim(myname)
 1234 format("Hello from fortran process: ",i4,"  Numprocs is ",i4,1x,a)
      
      mytag=123
      if(myid .eq. 0)then
           j=100
           icount=1
           call MPI_SEND(j,icount,MPI_INTEGER,1,mytag,MPI_COMM_WORLD,ierr)
      endif
      if(myid .eq. 1)then
      	call mpi_probe(0,mytag,MPI_COMM_WORLD,status,ierr)
      	call mpi_get_count(status,MPI_INTEGER,icount,ierr)
      	write(*,*)"getting ", icount
      	allocate(i(icount))
      	call mpi_recv(i,icount,MPI_INTEGER,0,mytag,MPI_COMM_WORLD,status,ierr)
      	write(*,*)"i=",i
      endif
      call mpi_finalize(ierr)
      stop
      end

