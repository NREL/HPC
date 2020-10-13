program darwin
      use numz
      use list_stuff
      use map_stuff
      use face
      use control
      use sort_mod
      use ran_mod
      implicit none
      character (len=2) a
      character (len=8) c_date ! the date
      character (len=10)c_time ! the time
      character (len=5) c_zone ! time zone
      character (len=12)tmpstr ! used to create a file name
                               ! base on the date and time
      character (len=25) fstr
      integer ivalues(8)
      integer ,allocatable,dimension(:) :: test_vect
      integer i,j,found
      real(b8) result,dt,dummy
      integer hr,minut,sec,ic1,icr,max_c,ic2
      namelist /the_input/mute_rate,fitmax, &
                          ncolor,num_gen,num_genes,nstates,seed, &
                          force,the_top,do_one
      call date_and_time(date=c_date, &  ! character(len=8) ccyymmdd 
                         time=c_time, &  ! character(len=10) hhmmss.sss 
                         zone=c_zone, &  ! character(len=10) +/-hhmm (time zone) 
                         values=ivalues) ! integer ivalues(8) all of the above 
      !create a file name based on the date and time
      write(tmpstr,"(a12)")(c_date(5:8)//c_time(1:4)//".dat")
      write(*,*)"name of output file= ",tmpstr
      open(output,file=tmpstr)
      write(output,'("start time: ",a2,":",a2,":",a4)') &
                      c_time(1:2),c_time(3:4),c_time(5:8)         
      mute_rate=0.01_b8
      fitmax=0.0_b8
      ncolor=4
      num_gen=100
      num_genes=100
      nstates=15
      seed=-12345
      force=.true.
      the_top=.true.
      do_one=.true.
      read(input,the_input)
      write(output,the_input)
      dummy=ran1(seed)  !init our random number generator

      allocate(map(nstates)) ! allocate our map
      do i=1,nstates
        read(12,"(a2)",advance="no")map(i)%name ! read the state name
        !write(*,*)"state:",map(i)%name
        nullify(map(i)%list)                    ! "zero out" our list
        do 
        read(12,"(1x,a2)",advance="no")a      ! read list of states 
                                              ! without going to the 
                                              ! next line 
        !write(*,*)"a=",a
            if(lge(a,"xx") .and. lle(a,"xx"))then  ! if state == xx 
                !read(12,"(/)",advance="no",end=1)  ! go to the next line 
                backspace(12)
                read(12,"(1x,a2)",end=1)a  ! go to the next line 
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
       if(force)then
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
           stop
       endif
       deallocate(test_vect)
! allocate and init our population
       write(fstr,'("(",i4,"i1,1x,f10.5)")')nstates
       allocate(fit(num_genes))            ! allocate the data type to hold 
       allocate(gene(nstates,num_genes))   ! allocate our collection of genes
       allocate(kids(nstates,num_genes))   ! temp space for next generation
       call init_genes(gene)               ! starting data
       call system_clock(count=ic1,   &  ! count of system clock (clicks) 
                         count_rate=icr,  &  ! clicks / second 
                         count_max=max_c)    ! max value for count 
       do i=1,num_gen
           do j=1,num_genes
               fit(j)%val=fitness(gene(:,j))
               fit(j)%index=j
           enddo
           call sort(fit,num_genes)
           write(*,'(i5,f10.3)')i,fit(1)%val
           if(fit(1)%val >= fitmax)exit
           if(i < num_gen)then
               call reproduce()
               call mutate()
           endif
       enddo
       write(*,     fstr)gene(:,fit(1)%index),fit(1)%val
       write(output,fstr)gene(:,fit(1)%index),fit(1)%val
       call system_clock(count=ic2)
       if(ic2 < ic1)then 
          dt=(max_c-ic1)+ic2
       else
          dt=ic2-ic1
       endif
       write(*,'("generations =",i5)')min(i,num_gen)
       write(output,'("generations =",i5)')min(i,num_gen)
       write(*,'("seconds of run time :",f10.2)')dt/real(icr,b8)
       write(output,'("seconds of run time :",f10.2)')dt/real(icr,b8)
          
end program





 


      
