subroutine mutate()
  use numz
  use ran_mod
  use galapagos
  use control
  implicit none
  integer i,j,ndo,k
  real(b8) sigma,mu
  real(b8)x1,x2
  if(mute_rate .le. 0.0)return
  if(do_one)then
          do j = 1 , nstates
              if(mute_rate > ran1())then
                  x1=ran1()
!                  i=nint((nstates-1)*ran1()+1)
                   i=nint((nstates-1)*x1    +1)
                  gene(i,j)=ncolor*ran1()
              endif
          enddo
  else
        mu=nstates*mute_rate
        sigma=sqrt(mu*(1.0_b8-mute_rate))
        do j = 1 , nstates
                ndo=nint(normal(sigma,mu))
                ndo=max(0,min(ndo,nstates))
            do i=1,ndo
                  x1=ran1()
!                 k=nint((nstates-1)*ran1()+1)
                  k=nint((nstates-1)*x1    +1)
                  gene(k,j)=ncolor*ran1()
            enddo
        enddo
  endif
  return
end subroutine mutate
