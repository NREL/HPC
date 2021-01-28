!version: mpi with 1d decomposition
! solves the 2d Stommel Model of Ocean Circulation  
! using a Five-point stencil and Jacobi iteration
!
! gamma*((d(d(psi)/dx)/dx) + (d(d(psi)/dy)/dy))
! +beta(d(psi)/dx)=-alpha*sin(pi*y/(2*ly))
!
module numz
! module defines the basic real type and pi
    integer, parameter:: b8 = selected_real_kind(14)
    real(b8), parameter :: pi = 3.141592653589793239_b8
end module
!*********************
module input
! module contains the inputs 
    use numz
    integer nx,ny             ! number of interior points for our grid (50 50)
    real(b8) lx,ly            ! physical size of our grid (2000 2000)
    real(b8) alpha,beta,gamma ! parameters of the calculation (1.0e-9 2.25e-11 3.0e-6)
    integer steps             ! number of Jacobi iteration steps (60)
end module
!*********************
module constants
! this module contains the invariants (constants) of the
! calculation.  these values are determined in the main
! routine and used in the do_jacobi Jacobi iteration subroutine
! a6 is used in the force function
    use numz
    real(b8) dx,dy,a1,a2,a3,a4,a5,a6
    real(b8), allocatable:: for(:,:)     ! our force grid
end module
!*********************
module face
! this module contains the interface for the two subroutines
! that modify the grid.  an interface is a good idea in this
! case because we are passing allocatable arrays
    interface bc
        subroutine bc (psi,i1,i2,j1,j2)
! sets the boundary conditions
! input is the grid and the indices for the interior cells
            use numz
            integer,intent(in):: i1,i2,j1,j2
            real(b8),dimension(i1:i2,j1:j2):: psi
        end subroutine
    end interface
    interface do_jacobi
        subroutine do_jacobi (psi,new_psi,diff,i1,i2,j1,j2)
! does a single Jacobi iteration step
! input is the grid and the indices for the interior cells
! new_psi is temp storage for the the updated grid
! output is the updated grid in psi and diff which is
! the sum of the differences between the old and new grids
            use numz
            integer,intent(in):: i1,i2,j1,j2
            real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: psi
            real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: new_psi
            real(b8) diff
        end subroutine
    end interface
! interface for the forcing function
    interface force
        function force(y)
            use numz
            real(b8) force,y
        end function force
    end interface   
! interface for setting  the force routine
    interface do_force
        subroutine do_force (i1,i2,j1,j2)
! sets the force array
! input is the grid and the indices for the interior cells
            use numz
            integer,intent(in):: i1,i2,j1,j2
        end subroutine
    end interface
! interface for routine to write the grid
    interface write_grid
        subroutine write_grid (psi,i1,i2,j1,j2)
! input is the grid and the indices for the interior cells
            use numz
            integer,intent(in):: i1,i2,j1,j2
            real(b8),dimension(i1:i2,j1:j2):: psi
        end subroutine
    end interface
    interface do_transfer
        subroutine do_transfer (psi,i1,i2,j1,j2)
! sets the boundary conditions
! input is the grid and the indices for the interior cells
            use numz
            integer,intent(in):: i1,i2,j1,j2
            real(b8),dimension(i1:i2,j1:j2):: psi
        end subroutine
    end interface
    interface unique
        function unique(name)
            character (len=*) name
            character (len=20) unique
        end function
    end interface
end module
module mympi
    use mpi
!   include "mpif.h"
    integer numnodes,myid,mpi_err
    integer, parameter::mpi_master=0
    integer status(MPI_STATUS_SIZE)
end module
!*********************
program stommel
    use numz
    use input
    use constants
    use face
    use mympi
    implicit none
    real(b8) t1,t2
    real(b8)diff,mydiff
    real(b8)dx2,dy2,bottom
    real(b8), allocatable:: psi(:,:)     ! our calculation grid
    real(b8), allocatable:: new_psi(:,:) ! temp storage for the grid
    integer i,j,i1,i2,j1,j2
    integer iout
    integer ic,COMMAND_ARGUMENT_COUNT
    character (len=128) arg
    character (len=64) :: instr
    real(b8) dj
! do the mpi init stuff
    call MPI_INIT( mpi_err )
    call MPI_COMM_SIZE( MPI_COMM_WORLD, numnodes, mpi_err )
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, mpi_err )
! get the input.  see above for typical values
    if(myid .eq. mpi_master)then
        instr="200 200"                ; read(instr,*)nx,ny
        instr="2000000 2000000"        ; read(instr,*)lx,ly
        instr="1.0e-9 2.25e-11 3.0e-6" ; read(instr,*)alpha,beta,gamma
        instr="75000"                  ; read(instr,*)steps
        ic=COMMAND_ARGUMENT_COUNT()
        if (ic .gt. 0)then
            do i=1,ic
                CALL get_command_argument(i, arg)
                WRITE (*,*) "command line argument ",i,TRIM(arg)
            enddo
        endif
    endif
!send the data to other processors
    call MPI_BCAST(nx,   1,MPI_INTEGER,         mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(ny,   1,MPI_INTEGER,         mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(steps,1,MPI_INTEGER,         mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(lx,   1,MPI_DOUBLE_PRECISION,mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(ly,   1,MPI_DOUBLE_PRECISION,mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(alpha,1,MPI_DOUBLE_PRECISION,mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(beta, 1,MPI_DOUBLE_PRECISION,mpi_master,MPI_COMM_WORLD,mpi_err)
    call MPI_BCAST(gamma,1,MPI_DOUBLE_PRECISION,mpi_master,MPI_COMM_WORLD,mpi_err)
! calculate the constants for the calculations
    dx=lx/(nx+1)
    dy=ly/(ny+1)
    dx2=dx*dx
    dy2=dy*dy
    bottom=2.0_b8*(dx2+dy2)
    a1=(dy2/bottom)+(beta*dx2*dy2)/(2.0_b8*gamma*dx*bottom)
    a2=(dy2/bottom)-(beta*dx2*dy2)/(2.0_b8*gamma*dx*bottom)
    a3=dx2/bottom
    a4=dx2/bottom
    a5=dx2*dy2/(gamma*bottom)
    a6=pi/(ly)
! set the indices for the interior of the grid
! we stripe the grid across the processors
    i1=1
    i2=ny
    dj=real(nx,b8)/real(numnodes,b8)
    j1=nint(1.0_b8+myid*dj)
    j2=nint(1.0_b8+(myid+1)*dj)-1
    if(myid == mpi_master)write(*,'("rows= ",i4)')numnodes
    write(*,101)myid,i1,i2,j1,j2
101 format("myid= ",i4,3x,&
           " (",i3," <= i <= ",i3,") , ",            &
           " (",i3," <= j <= ",i3,")")

! allocate the grid to (i1-1:i2+1,j1-1:j2+1) this includes boundary cells
    allocate(psi(i1-1:i2+1,j1-1:j2+1))
    allocate(new_psi(i1-1:i2+1,j1-1:j2+1))
    allocate(for(i1-1:i2+1,j1-1:j2+1))
! set initial guess for the value of the grid
    psi=1.0_b8
! set boundary conditions
    call bc(psi,i1,i2,j1,j2)
! set the force array
    call do_force(i1,i2,j1,j2)
    call do_transfer(psi,i1,i2,j1,j2)
! do the jacobian iterations
    iout=steps/10
    if(iout == 0)iout=1
    t1=MPI_Wtime()
    do i=1,steps
        call do_jacobi(psi,new_psi,mydiff,i1,i2,j1,j2)
        call do_transfer(psi,i1,i2,j1,j2)
!       write(*,*)myid,i,mydiff
	call MPI_REDUCE(mydiff,diff,1,MPI_DOUBLE_PRECISION, &
	                MPI_SUM,mpi_master,MPI_COMM_WORLD,mpi_err)
	if(myid .eq. mpi_master .and. mod(i,iout) .eq. 0)write(*,'(i6,1x,g20.10)')i,diff
    enddo
    t2=MPI_Wtime()
    if(myid .eq. mpi_master)write(*,'("run time =",f10.2)')t2-t1
    !call write_grid(psi,i1,i2,j1,j2)
    call MPI_Finalize(mpi_err)
end program stommel
!*********************
subroutine bc(psi,i1,i2,j1,j2)
! sets the boundary conditions
! input is the grid and the indices for the interior cells
    use numz
    use mympi
    use input, only : nx,ny
    implicit none
    integer,intent(in):: i1,i2,j1,j2
    real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: psi
! do the top edges
    if(i1 .eq.  1) psi(i1-1,:)=0.0_b8
! do the bottom edges
    if(i2 .eq. ny) psi(i2+1,:)=0.0_b8
! do left edges
    if(j1 .eq.  1) psi(:,j1-1)=0.0_b8
! do right edges
    if(j2 .eq. nx) psi(:,j2+1)=0.0_b8
end subroutine bc
!*********************
subroutine do_jacobi(psi,new_psi,diff,i1,i2,j1,j2)
! does a single Jacobi iteration step
! input is the grid and the indices for the interior cells
! new_psi is temp storage for the the updated grid
! output is the updated grid in psi and diff which is
! the sum of the differences between the old and new grids
    use numz
    use constants
    implicit none
    integer,intent(in) :: i1,i2,j1,j2
    real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: psi
    real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: new_psi
    real(b8) diff
    integer i,j
    real(b8) y
    diff=0.0_b8
    do j=j1,j2
       do i=i1,i2
!            y=j*dy
            new_psi(i,j)=a1*psi(i+1,j) + a2*psi(i-1,j) + &
                         a3*psi(i,j+1) + a4*psi(i,j-1) - &
                         a5*for(i,j)
!                         a5*force(y)
            diff=diff+abs(new_psi(i,j)-psi(i,j))
         enddo
     enddo
    psi(i1:i2,j1:j2)=new_psi(i1:i2,j1:j2)
end subroutine do_jacobi
!*********************
function force(y)
    use numz
    use input
    use constants
    implicit none
    real(b8) force,y
    force=-alpha*sin(y*a6)
end function force
!*********************
subroutine do_force (i1,i2,j1,j2)
! sets the force conditions
! input is the grid and the indices for the interior cells
    use numz
    use constants, only:for,dy
    use face, only : force
    implicit none
    integer,intent(in):: i1,i2,j1,j2
    real(b8) y
    integer i,j
    do i=i1,i2
        do j=j1,j2
            y=j*dy
            for(i,j)=force(y)
        enddo
    enddo
end subroutine
!*********************
subroutine write_grid(psi,i1,i2,j1,j2)
! input is the grid and the indices for the interior cells
    use numz
    use mympi
    use face ,only : unique
    implicit none
    integer,intent(in):: i1,i2,j1,j2
    real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: psi
    integer i,j
    integer istart,iend,jstart,jend
! each processor writes its section of the grid
    istart=i1-1
    iend=i2+1
    jstart=j1-1
    jend=j2+1
    open(18,file=unique("out1d_"),recl=max(80,15*((jend-jstart)+3)+2))
    write(18,101)myid,istart,iend,jstart,jend

101 format("myid= ",i3,3x,                 &
           " (",i3," <= i <= ",i3,") , ", &
           " (",i3," <= j <= ",i3,")")

    do i=istart,iend
       do j=jstart,jend
           write(18,'(g14.7)',advance="no")psi(i,j)
           if(j .ne. j2)write(18,'(" ")',advance="no")
       enddo
       write(18,*)
    enddo
    close(18)
end subroutine write_grid
!*********************
subroutine do_transfer(psi,i1,i2,j1,j2)
! sets the boundary conditions
! input is the grid and the indices for the interior cells
    use numz
    use mympi
    use input
    implicit none
    integer,intent(in):: i1,i2,j1,j2
    real(b8),dimension(i1-1:i2+1,j1-1:j2+1):: psi
    integer num_x,myleft,myright
    logical even
    num_x=i2-i1+3
    myleft=myid-1
    myright=myid+1
    if(myleft .le. -1)myleft=MPI_PROC_NULL
    if(myright .ge. numnodes)myright=MPI_PROC_NULL
!   write(*,*)"left,mid,right",myleft,myid,myright
    if(even(myid))then
! send to left
            call MPI_SEND(psi(:,j1),  num_x,MPI_DOUBLE_PRECISION,myleft, &
                                      100,MPI_COMM_WORLD,mpi_err)
!	    write(*,*)"sl",myid,psi(:,j1)
! rec from left
            call MPI_RECV(psi(:,j1-1),num_x,MPI_DOUBLE_PRECISION,myleft, &
                                      100,MPI_COMM_WORLD,status,mpi_err)
!	    write(*,*)"rl",myid,psi(:,j1-1)
! rec from right
            call MPI_RECV(psi(:,j2+1),num_x,MPI_DOUBLE_PRECISION,myright, &
                                      100,MPI_COMM_WORLD,status,mpi_err)
!	    write(*,*)"rr",myid,psi(:,j2+1)
! send to right
            call MPI_SEND(psi(:,j2),  num_x,MPI_DOUBLE_PRECISION,myright, &
                                      100,MPI_COMM_WORLD,mpi_err)
!	    write(*,*)"sr",myid,psi(:,j2)
    else    ! we are on an odd col processor
! rec from right
            call MPI_RECV(psi(:,j2+1),num_x,MPI_DOUBLE_PRECISION,myright, &
                                      100,MPI_COMM_WORLD,status,mpi_err)
!	    write(*,*)"rr",myid,psi(:,j2+1)
! send to right
            call MPI_SEND(psi(:,j2),  num_x,MPI_DOUBLE_PRECISION,myright, &
                                      100,MPI_COMM_WORLD,mpi_err)
!	    write(*,*)"sr",myid,psi(:,j2)
! send to left
            call MPI_SEND(psi(:,j1),  num_x,MPI_DOUBLE_PRECISION,myleft, &
                                      100,MPI_COMM_WORLD,mpi_err)
!	    write(*,*)"sl",myid,psi(:,j1)
! rec from left
            call MPI_RECV(psi(:,j1-1),num_x,MPI_DOUBLE_PRECISION,myleft, &
                                      100,MPI_COMM_WORLD,status,mpi_err)
!	    write(*,*)"rl",myid,psi(:,j1-1)
    endif
end subroutine do_transfer
!*********************
function unique(name)
    use numz
    use mympi
    character (len=*) name
    character (len=20) unique
    character (len=80) temp
    if(myid .gt. 99)then
      write(temp,"(a,i3)")trim(name),myid
    else
        if(myid .gt. 9)then
            write(temp,"(a,'0',i2)")trim(name),myid
        else
            write(temp,"(a,'00',i1)")trim(name),myid
        endif
    endif
    unique=temp
    return
end function unique
!*********************
function even(i)
  integer i
  logical even
  j=i/2
  if(j*2 .eq. i)then
      even = .true.
  else
      even = .false.
  endif
  return
end function even

