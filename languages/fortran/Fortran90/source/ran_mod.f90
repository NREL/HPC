module ran_mod 
! ran1 returns a uniform random number between 0-1
contains 
   function ran1(my_seed)
      use numz
      implicit none
      real(b8) ran1,r
      integer, optional,intent(in) :: my_seed
      integer,allocatable :: seed(:)
      integer the_size,j
      if(present(my_seed))then            ! use the seed if present
          call random_seed(size=the_size) ! how big is the seed?
          allocate(seed(the_size))        ! allocate space for seed
          do j=1,the_size
             seed(j)=abs(my_seed)+(j+1)        ! create the seed
          enddo
          call random_seed(put=seed)      ! assign the seed
          deallocate(seed)                ! deallocate space    
      endif
      call random_number(r)
      ran1=r
  end function ran1
  

  function spread(min,max)  !returns random number between min - max 
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
            do while(rsq.ge.1.0_b8.or.rsq.eq.0.0_b8) 
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
    end function normal 

end module ran_mod 


