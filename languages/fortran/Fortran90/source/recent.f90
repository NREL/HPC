module numz 
  integer, parameter:: b8 = selected_real_kind(14) 
end module 

module face 
    interface fitness 
        function fitness(vector) 
          use numz 
          implicit none 
          real(b8) fitness 
          integer, dimension(:),target ::  vector 
        end function fitness 
    end interface 

    interface add1  ! interface for our function 
                    ! that returns a vector 
        function add1(vector,max) result (rtn) 
        integer, dimension(:),intent(in) ::  vector 
        integer max 
        integer,dimension(size(vector)) :: rtn 
        end function 
    end interface 
end module 
  

module list_stuff 
    type llist                     ! type for the linked list 
        integer index              ! our data 
        type(llist),pointer::next  ! pointer to the next element 
    end type llist 

    type states                     ! "map" data type 
        character(len=2)name        ! mane of the state 
        type(llist),pointer :: list ! list of neighbors 
    end type states 
end module 
  

module global_test  ! stuff used by the fitness function 
    integer,pointer,dimension(:) :: test_vect 
    integer my_color 
end module 

module map_stuff 
    use list_stuff 
    interface insert ! inserts elements into the linked list 
        recursive subroutine insert (item, root) 
            use list_stuff 
            implicit none 
            type(llist), pointer :: root 
            integer item 
        end subroutine 
    end interface 

    interface ltest ! used by the fitness function 
        recursive function ltest(list) result (connect) 
             use list_stuff 
             use global_test 
            type(llist),pointer :: list 
            integer connect 
        end function 
    end interface 
  
    interface printit ! prints out a "map" 
        recursive subroutine printit(list,map) 
            use list_stuff 
            type(llist),pointer :: list 
            type(states),dimension(:):: map 
        end subroutine 
    end interface 
    ! map is the description of our map 
    type(states),allocatable,dimension(:),save :: map 
end module 
  
  

program abcd 
      use numz 
      use list_stuff 
      use map_stuff 
      use face 
      implicit none 
      character (len=2) a 
      character (len=8) c_date ! the date 
      character (len=10)c_time ! the time 
      character (len=5) c_zone ! time zone 
      character (len=12)tmpstr ! used to create a file name 
                               ! base on the date and time 
      character (len=30)fstr   ! create a format on the fly 
      integer hr,minut,sec 
      integer ivalues(8) ! used with date_and_time routine 
      integer nstates 
      integer len_b8,len_real 
      integer ,allocatable,dimension(:) :: test_vect 
      integer i,j,found 
      real(b8) result 
      integer ncolor ! number of colors in our map 
      logical force  ! find brute force solution 
      namelist /the_input/ncolor,force,nstates
      ncolor=4 
      force=.true. 
      nstates=5
      read(13,the_input) 
      write(*,the_input) 
      allocate(map(nstates)) ! allocate our map 
      do i=1,nstates 
        read(13,"(a2)",advance="no")map(i)%name ! read the state name 
        write(*,*)"state:",map(i)%name 
        nullify(map(i)%list)                    ! "zero out" our list 
        do 
        read(13,"(1x,a2)",advance="no")a      ! read list of states 
                                              ! without going to the 
                                              ! next line 
            if(lge(a,"xx") .and. lle(a,"xx"))then  ! if state == xx 
                backspace(13)
                read(13,"(1x,a2)",end=1)a  ! go to the next line 
                exit 
            endif 
          1 continue 
            if(llt(a,map(i)%name))then   ! we only add a state to 
                                         ! our list if its name 
                                         ! is before ours thus we 
                                         ! only count boarders 1 time 
! what we want put into  our linked list is an index 
! into our map where we find the boarding state 
! thus we do the search here 
! any ideas on a better way of doing this? 
                found=-1 
                do j=1,i-1 
                    if(lge(a,map(j)%name) .and. lle(a,map(j)%name))then 
                        write(*,*)a 
                        found=j 
                        exit 
                    endif 
                enddo 
                if(found == -1)then 
                    write(*,*)"error" 
                    stop 
                endif 
! found the index of the boarding state insert it into our list 
                call insert(found,map(i)%list) 
            endif 
        enddo 
       enddo 
       do i=1,nstates ! print our "purged" list of states 
          write(*,"(a2)",advance="no")map(i)%name 
          if(associated(map(i)%list))then 
             call printit(map(i)%list,map) 
          else 
             write(*,*)" xx" 
          endif 
       enddo 
    ! test our fitness function 
       allocate(test_vect(nstates)) 
       test_vect=1 
       write(*,*)"connectivity of map = ",fitness(test_vect) 
       test_vect=0 
       if(nstates < 10 .and. force)then 
        ! show size of our "b8" real compared to a regular real 
     ! on most machines ratio is 2/1 
           inquire(iolength=len_real)1.0  ! a regular real 
           inquire(iolength=len_b8)1.0_b8 ! our "b8" real 
           write(*,*)"len_b8  ",len_b8 
           write(*,*)"len_real",len_real 
           write(*,*)"len_b8",len_b8 
           call date_and_time(date=c_date, &  ! character(len=8) ccyymmdd 
                              time=c_time, &  ! character(len=10) hhmmss.sss 
                              zone=c_zone, &  ! character(len=10) +/-hhmm (time zone) 
                              values=ivalues) ! integer ivalues(8) all of the above 
           !create a file name based on the date and time 
     write(tmpstr,"(a12)")(c_date(5:8)//c_time(1:4)//".dat") 
           write(*,*)"name of file= ",tmpstr 
           !open(14,file=tmpstr) 
           write(*,*)c_time,c_zone 
           do 
               test_vect=add1(test_vect,ncolor-1) 
               result=fitness(test_vect) 
               if(sum(test_vect) == 0)exit 
               if(result < 1.0_b8)then 
          !create a format on the fly 
                   write(fstr,'("(",i4,"i1,1x,f10.5)")')nstates 
                   write(*,*)"format= ",fstr 
                   write(*,fstr)test_vect,result 
                   call date_and_time(time=c_time) 
                   write(*,*)c_time 
                   read(c_time,"(3i2)")hr,minut,sec 
                   write(*,*)hr,minut,sec 
                   stop 
               endif 
           enddo 
           write(*,*)"no answer found" 
           call date_and_time(time=c_time) 
           write(*,*)c_time 
       endif 
       deallocate(test_vect) 
end program 
  

! our linked list insert routine 
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

recursive subroutine printit(list,map) 
        use list_stuff 
        type(llist),pointer :: list 
        type(states),dimension(:) :: map 
        if(.not. associated(list))then 
            write(*,*)" xx" 
            return 
        else 
            write(*,"(1x,2a)",advance="no")map(list%index)%name 
            call printit(list%next,map) 
        endif 
end subroutine printit 
  

function fitness(vector) 
    use numz 
    use global_test 
    use map_stuff 
    implicit none 
    real(b8) fitness 
    integer, dimension(:),target ::  vector 
    integer isize,i,tot 
    test_vect=>vector 
    isize=size(vector) 
    tot=0 
! our function compares the color of every state 
! to those to which it shares a boarder 
! map(i)%list is the list of boarding states for state i 
    do i=1,isize 
        my_color=vector(i) 
        tot=tot+ltest(map(i)%list) 
    enddo 
    fitness=tot 
end function 

! function used by the fitness function 
! compares the color of a list of states 
! to "mycolor" adds 1 for each match 
recursive function ltest(list) result (connect) 
        use list_stuff 
        use global_test 
        type(llist),pointer :: list 
        integer connect,tmp 
        if(.not. associated(list))then 
            connect=0 
        else 
            if(test_vect(list%index)  == my_color)then 
                connect=1+ltest(list%next) 
            else 
                connect=ltest(list%next) 
            endif 
        endif 
end function ltest 
  



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

