!mpif90 -fopenmp -fallow-argument-mismatch -w -lm fhostone.f90 
!See mympi.f90 on how to compile this program without MPI - pure openmp
module mympi
   include "mpif.h"
end module

!****************************************************************
!  This is a hello world program in MPI and OpenMP. Each thread
!  prints out its thread and MPI id.
!
!  It also shows how to create a collection of node specific
!  MPI communicators based on the name of the node on which a
!  task is running.  Each node has it own "node_com" so each
!  thread also prints its MPI rank in the node specific
!  communicator.
!****************************************************************
subroutine dohelp
   write (*, *) "phostname arguments:"
   write (*, *) "          -h : Print this help message"
   write (*, *)
   write (*, *) "no arguments : Print a list of the nodes on which the command is run."
   write (*, *)
   write (*, *) " -f or -1    : Same as no argument but print MPI task id and Thread id"
   write (*, *) "               If run with OpenMP threading enabled OMP_NUM_THREADS > 1"
   write (*, *) "               there will be a line per MPI task and Thread."
   write (*, *)
   write (*, *) " -F or -2    : Add columns to tell first MPI task on a node and and the"
   write (*, *) "               numbering of tasks on a node. (Hint: pipe this output in"
   write (*, *) "               to sort -r"
   write (*, *)
   write (*, *) " -E or -B    : Print thread info at 'E'nd of the run or 'B'oth the start and end"
   write (*, *)
   write (*, *) " -a          : Print a listing of the environmental variables passed to"
   write (*, *) "               MPI task. (Hint: use the -l option with SLURM to prepend MPI"
   write (*, *) "               task #."
   write (*, *)
   write (*, *) " -s ######## : Where ######## is an integer.  Sum a bunch on integers to slow"
   write (*, *) "               down the program.  Should run faster with multiple threads."
   write (*, *)
   write (*, *) " -t ######## : Where is a time in seconds.  Sum a bunch on integers to slow"
   write (*, *) "               down the program and run for at least the given seconds."
   write (*, *)
   write (*, *) " -T          : Print time/date at the beginning/end of the run."
   write (*, *)
end subroutine
module numz
! define the basic real type and pi (not used in this example)
   integer, parameter:: b8 = selected_real_kind(14)
   real(b8), parameter :: pi = 3.141592653589793239_b8
   integer, parameter :: in8 = selected_int_kind(12)
end module

module getit
contains
   function get_core_c()
      USE ISO_C_BINDING, ONLY: c_long, c_char, C_NULL_CHAR, c_int
      use numz
      implicit none
      integer(in8) :: get_core_c
      interface
         !integer(c_long) function cfunc() BIND(C, NAME='pthread_self')
         integer(c_long) function cfunc() BIND(C, NAME='sched_getcpu')
            USE ISO_C_BINDING, ONLY: c_long, c_char
         end function cfunc
      end interface
      get_core_c = cfunc()
   end function
   function strcmp(astr, bstr)
      implicit none
      integer strcmp
      character(len=*) astr, bstr
      ! write(*,*)trim(astr),trim(bstr),trim(astr).eq. trim(bstr)
      strcmp = -1
      if (trim(astr) .eq. trim(bstr)) strcmp = 0
   end function

   subroutine str2real(str, rl, stat)
      implicit none
      ! Arguments
      character(len=*), intent(in) :: str
      double precision, intent(out)         :: rl
      integer, intent(out)         :: stat
      read (str, *, iostat=stat) rl
   end subroutine str2real
end module

program hello
   use mympi
   use numz
   use getit
   implicit none
   character(len=MPI_MAX_PROCESSOR_NAME + 1):: name
   integer tn, omp_get_thread_num
   integer myid, ierr, numprocs, nlen
   integer mycol, node_comm, new_id, new_nodes
   integer ib, ie
   integer narg, isum
   character(len=32) cmdlinearg
   integer full, envs, help, dotime, when, argc, iarg, wait, slow, vlan
   character(len=256) :: argv
   character(len=MPI_MAX_LIBRARY_VERSION_STRING+1) :: version
   integer i, nints
   real(b8) dt, dummy, t1, t2
   call MPI_INIT(ierr)
   call MPI_Get_library_version(version, vlan, ierr)
   call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
   call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
   call MPI_Get_processor_name(name, nlen, ierr)
   if (myid == 0) then
      full = 0
      envs = 0
      help = 0
      dotime = 0
      when = 1
      argc = COMMAND_ARGUMENT_COUNT()
      if (argc > 0) then
         do iarg = 1, argc
            CALL get_command_argument(iarg, argv)
            !write(*,*)argv
            if ((strcmp(argv, "-h") == 0) .or. &
                (strcmp(argv, "--h") == 0) .or. &
                (strcmp(argv, "-help") == 0)) help = 1
            !**!
            if ((strcmp(argv, "-f") == 0) .or. (strcmp(argv, "-1") == 0)) full = 1
            !**!
            if ((strcmp(argv, "-F") == 0) .or. (strcmp(argv, "-2") == 0)) full = 2
            !**!
            if (strcmp(argv, "-s") == 0) slow = 1
            !**!
            if (strcmp(argv, "-t") == 0) wait = 1
            !**!
            if (strcmp(argv, "-a") == 0) envs = 1
            !**!
            if (strcmp(argv, "-T") == 0) dotime = 1
            if (strcmp(argv, "-B") == 0) when = 3
            if (strcmp(argv, "-E") == 0) when = 2
         end do
      end if
      !write(*,*)full,envs,help,dotime,when
   end if

   call MPI_BCAST(help, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
   if (help == 1) then
      if (myid == 0) then
         call dohelp()
      end if
      call MPI_Finalize(ierr)
      stop
   end if
   call MPI_Bcast(full, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
   call MPI_Bcast(envs, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
   call MPI_Bcast(when, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
   if (myid == 0 .and. dotime == 1) call ptime()

   if (myid == 0 .and. full == 2) then
      write (*, '("MPI Version:",a)') trim(version)
      write (*, '(a)') "task    thread             node name  first task    # on node  core"
   end if
   call node_color(mycol)
   call MPI_COMM_Split(mpi_comm_world, mycol, myid, node_comm, ierr)
   call MPI_COMM_Rank(node_comm, new_id, ierr)
   call MPI_COMM_Size(node_comm, new_nodes, ierr)
   tn = -1
   do i = 0, numprocs - 1
      call MPI_Barrier(MPI_COMM_WORLD, ierr)
      if (i .eq. myid) then
         if (when == 3) call str_low(name)
         if (when .ne. 2) call dothreads(full, name, myid, mycol, new_id)
         if (envs == 1 .and. (myid == -1 .or. myid == 0)) then
            write (*, *) "Env not supported"
         end if
      end if
   end do
   if (myid == 0) then
      dt = 0
      if (wait == 1) then
         slow = 0
         do iarg = 1, argc
            CALL get_command_argument(iarg, argv)
            call str2real(argv, dummy, ierr)
            if (ierr == 0) dt = dummy
         end do
      end if
   end if
   call MPI_Bcast(dt, 1, MPI_Double_precision, 0, MPI_COMM_WORLD, ierr)
   if (dt > 0) then
      nints = 100000
      t1 = MPI_Wtime()
      t2 = t1
      do while (dt > (t2 - t1))
         do i = 1, 1000
            call slowit(nints, i)
         end do
         t2 = MPI_Wtime()
      end do
      if (myid == 0) write (*, '(a,f10.2)') "total time ", t2 - t1
      nints = 0
   end if
   if (myid == 0) then
      nints = 0; 
      if (slow == 1) then
         do iarg = 1, argc
            CALL get_command_argument(iarg, argv)
            call str2real(argv, dummy, ierr)
            if (ierr == 0) nints = nint(dummy)
         end do
      end if
   end if
   call MPI_Bcast(nints, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
   if (nints > 0) then
      t1 = MPI_Wtime()
      do i = 1, 1000
         call slowit(nints, i)
      end do
      t2 = MPI_Wtime()
      if (myid == 0) write (*, "(a,f10.2)") "total time ", t2 - t1
   end if

   if (myid == 0 .and. dotime == 1) call ptime()
   if (when > 1) then
      do i = 0, numprocs - 1
         call MPI_Barrier(MPI_COMM_WORLD, ierr)
         if (i == myid) then
            if (when == 3) call str_upr(name)
            call dothreads(full, name, myid, mycol, new_id)
         end if
      end do
   end if
   call MPI_FINALIZE(ierr)
   stop
end
subroutine node_color(mycol)
! return a integer which is unique to all mpi
! tasks running on a particular node.  It is
! equal to the id of the first MPI task running
! on a node.  This can be used to create
! MPI communicators which only contain tasks on
! a node.
   use mympi
   use numz
   implicit none
   integer mycol
   integer status(MPI_STATUS_SIZE)
   integer xchng, i, n2, myid, numprocs
   integer ierr, nlen
   integer ib, ie
   character(len=MPI_MAX_PROCESSOR_NAME + 1):: name
   character(len=MPI_MAX_PROCESSOR_NAME + 1)::nlist
   real(b8) t1, t2

   call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
   call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
   call MPI_Get_processor_name(name, nlen, ierr)
! this next line is required on the BGQ
! the BGQ gives a different MPI name to each MPI task,
! encoding the task id and the location in the torus.
! we need to strip all of this off to just give us the
! node name
   !ib=index(trim(name)," ",.true.) ie=len_trim(name) name=name(ib:ie)
   nlist = name
   mycol = myid
   ! find n2, the power of two >= numprocs
   n2 = 1
   do while (n2 < numprocs)
      n2 = n2*2
   end do
!    write(*,*)"myid=",myid
   do i = 1, n2 - 1
      ! do xor to find the processor xchng
      xchng = xor(i, myid)
      if (xchng <= (numprocs - 1)) then
         ! do an exchange if our "current" partner exists
         if (myid < xchng) then
!          write(*,*)i,myid,"send from ",myid," to ", xchng
            call MPI_Send(name, MPI_MAX_PROCESSOR_NAME, &
                          MPI_CHARACTER, xchng, 12345, &
                          MPI_COMM_WORLD, ierr)
!          write(*,*)i,myid,"recv from ",xchng," to ",myid
! add 1 here = hack to work with the C version of this code
            call MPI_Recv(nlist, MPI_MAX_PROCESSOR_NAME + 1, &
                          MPI_CHARACTER, xchng, 12345, &
                          MPI_COMM_WORLD, status, ierr)
         else
!          write(*,*)i,myid,"recv from ",xchng," to ",myid
! add 1 here = hack to work with the C version of this code
            call MPI_Recv(nlist, MPI_MAX_PROCESSOR_NAME + 1, &
                          MPI_CHARACTER, xchng, 12345, &
                          MPI_COMM_WORLD, status, ierr)
!          write(*,*)i,myid,"send from ",myid," to ",xchng
            call MPI_Send(name, MPI_MAX_PROCESSOR_NAME, &
                          MPI_CHARACTER, xchng, 12345, &
                          MPI_COMM_WORLD, ierr)
         end if
         if (nlist == name .and. xchng < mycol) mycol = xchng
      else
         ! skip this stage
      end if
   end do
!    write(*,*)
end subroutine

subroutine sumit(nvals, val)
   use numz
   use mympi
   implicit none
   integer nvals, val
   integer, allocatable :: block(:)
   integer ktimes,ijk,i
   integer(in8) sum
   real(b8) t1, t2
   allocate (block(nvals))
   ktimes = 50
   t1 = mpi_wtime()
!$OMP PARALLEL do
   do i = 1, nvals
      block(i) = val
   end do
   sum = 0
   do ijk = 1, ktimes
!$omp parallel do reduction(+:sum)
      do i = 1, nvals
         sum = sum + block(i)
      end do
   end do
   t2 = MPI_Wtime()
   write (*, '("sum of integers ",i15,f10.3)') sum/ktimes, t2 - t1
   deallocate (block)
end subroutine

subroutine dothreads(full, myname, myid, mycolor, new_id)
   use getit
   implicit none
   integer full, myid, mycolor, new_id
   character(len=*) myname
   integer nt, tn
   integer omp_get_num_threads, omp_get_thread_num
!$OMP PARALLEL
   nt = omp_get_num_threads()
   if (nt == 0) nt = 1
!$OMP CRITICAL
   if (nt < 2) then
      nt = 1
      tn = 0
   else
      tn = omp_get_thread_num()
   end if
   if (full == 0) then
      if (tn == 0) write (*, 1236) trim(myname)
   end if
   if (full == 1) then
      write (*, 1235) trim(myname), myid, tn
   end if
   if (full == 2) then
      write (*, 1234) myid, tn, trim(myname), mycolor, new_id, get_core_c()
   end if
!$OMP END CRITICAL
!$OMP END PARALLEL

1234 format(i4.4, 6x, i4.4, 4x, a18, 8x, i4.4, 9x, i4.4, 2x, i4.3)
1235 format(a, 1x, i4.4, 1x, i4.4)
1236 format(a)

end subroutine

subroutine ptime()
   implicit none
   character(len=8) :: date
   character(len=10) :: time
   CALL DATE_AND_TIME(DATE, TIME)
   write (*, *) date(1:4), "/", date(5:6), "/", date(7:8), " ", &
      time(1:2), ":", time(3:4), ":", time(5:10)
end subroutine

subroutine str_low(str)

!   ==============================
!   Changes a string to upper case
!   ==============================

   Implicit None
   Character(*) :: str
   Integer :: ic, i

   Character(26), Parameter :: cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
   Character(26), Parameter :: low = 'abcdefghijklmnopqrstuvwxyz'

!   Capitalize each letter if it is lowecase
   do i = 1, LEN_TRIM(str)
      ic = INDEX(cap, str(i:i))
      if (ic > 0) str(i:i) = low(ic:ic)
   end do

End subroutine str_low

subroutine str_upr(str)

!   ==============================
!   Changes a string to upper case
!   ==============================

   Implicit None
   Character(*) :: str
   Integer :: ic, i

   Character(26), Parameter :: cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
   Character(26), Parameter :: low = 'abcdefghijklmnopqrstuvwxyz'

!   Capitalize each letter if it is lowecase
   do i = 1, LEN_TRIM(str)
      ic = INDEX(low, str(i:i))
      if (ic > 0) str(i:i) = cap(ic:ic)
   end do

End subroutine str_upr

subroutine slowit(nints, val)
   integer nints, val
   integer, allocatable :: block(:)
   integer i, sum
   allocate (block(nints))
!$OMP  parallel do
   do i = 1, nints
      block(i) = val
   end do
   sum = 0
!$OMP  parallel do reduction(+ : sum)
   do i = 1, nints
      sum = sum + block(i)
   end do
   if (sum .lt. 0.0) write (*, *) "WTF"
   deallocate (block)
end subroutine

