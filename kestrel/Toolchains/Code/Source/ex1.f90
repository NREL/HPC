! This is a simple hybrid hello world program.
! Prints MPI information
! For each task/thread
!  task id
!  node name for task
!  thread id
!  # of threads for the task
!  core on which the thread is running

module getit
contains
!! Get the core on which a thread is running
  function get_core_c()
      USE ISO_C_BINDING, ONLY: c_long, c_char, C_NULL_CHAR, c_int
      implicit none
      integer, parameter :: in8 = selected_int_kind(12)
      integer(in8) :: get_core_c
      interface
         integer(c_long) function cfunc() BIND(C, NAME='sched_getcpu')
            USE ISO_C_BINDING, ONLY: c_long, c_char
         end function cfunc
      end interface
      get_core_c = cfunc()
   end function
!! runtriad runs "triad", in parallel, for 4 seconds to give threads time to settle
   function runtriad(myin)
      USE ISO_C_BINDING, ONLY: c_double,c_int
      implicit none
      double precision :: runtriad
      integer myin
      integer(c_int) :: myid

      interface
            real(c_double) function cfunc(myid) BIND(C, NAME='dotriad')
            USE ISO_C_BINDING, ONLY: c_double,c_int
            integer(c_int)  :: myid
         end function cfunc
      end interface
      myid=myin
      runtriad = cfunc(myin)
   end function
end module

program hybrid
    use getit
    use ISO_FORTRAN_ENV
    implicit none
    include 'mpif.h'
    integer numtasks,myid,ierr
    character (len=MPI_MAX_PROCESSOR_NAME):: myname
    character(len=MPI_MAX_LIBRARY_VERSION_STRING+1) :: version
    integer mylen,vlan,mycore,tin
    double precision wait
    integer OMP_GET_MAX_THREADS,OMP_GET_THREAD_NUM
    call MPI_INIT( ierr )
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
    call MPI_COMM_SIZE( MPI_COMM_WORLD, numtasks, ierr )
    call MPI_Get_processor_name(myname,mylen,ierr)
! print the MPI libraty version
    if (myid .eq. 0)then
      write(*,*)"Fortran MPI TASKS ",numtasks
      call MPI_Get_library_version(version, vlan, ierr)
      write(*,*)"MPI VERSION: ",trim(version)
      write(*,*)"BACKEND COMPILER: ",trim(ADJUSTL(COMPILER_VERSION()))
    endif
!! runtriad runs "triad", in parallel, for 4 seconds to give threads time to settle
!!  if input to triad is negative run for -# seconds
!!  if >=0 run "triad", in parallel one more time and give report to stderr

    tin=-4
    wait=runtriad(tin)
!$OMP PARALLEL
!$OMP CRITICAL
    mycore=get_core_c()
    write(unit=*,fmt="(a,i4.4,a,a)",advance="no") &
                " task ",myid, " is running on ",trim(myname)
    write(unit=*,fmt="(a,i3,a,i3,a,1x,i3.3)") &
            " thread= ",OMP_GET_THREAD_NUM(), &
            " of ",OMP_GET_MAX_THREADS(),     &
            " is on core ",mycore
!$OMP END CRITICAL
!$OMP END PARALLEL
    if (myid .eq. 0) write(*,fmt="(a,f10.2,a)")" ran triad for ",wait," seconds"
!! run "triad", in parallel one more time and give report to stderr
!!    wait=runtriad(myid)
    call MPI_FINALIZE(ierr)
end program
