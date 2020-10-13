subroutine reproduce()
    use control
    if(the_top)then
        call top_half
    else
        call roulette
    endif
end subroutine reproduce

subroutine top_half()
  use numz
  use ran_mod
  use control
  implicit none
  integer i,j,k,m,nstart,t,nend,j1,j2
  real(b8)x1,x2,x3,x4
! we allow the top half of the population to reproduce
! to produce a new gene we select two at random from the top half
  t=nstates/2-1
  do j = 1 , nstates
    x1=ran1()
    x2=ran1()
    x3=ran1()
    x4=ran1()
!    k=nint(t*ran1()+1)
!    i=nint(t*ran1()+1)
!    nstart=nint((nstates-1)*ran1()+1)
!    nend=nint((nstates-1)*ran1()+1)
    k=nint(t*x1+1)
    i=nint(t*x2+1)
    nstart=nint((nstates-1)*x3+1)
    nend=nint((nstates-1)*x4+1)
    if(nend .eq. nstart)then
            do m=1 , nstart-1
                kids(m,j)=gene(m,fit(k)%index)
            enddo
            do m=nstart , nstates
                kids(m,j)=gene(m,fit(i)%index)
            enddo
    else
        j1=min(nstart,nend)
        j2=max(nstart,nend)
            do m=1 , j1
                kids(m,j)=gene(m,fit(i)%index)
            enddo
            do m=j1+1 , j2-1
                kids(m,j)=gene(m,fit(k)%index)
            enddo
            do m=j2 , nstates
                kids(m,j)=gene(m,fit(i)%index)
            enddo
    endif
  enddo
  gene=kids
  return
end subroutine top_half

module locate_mod
        contains
                function locate(vect,x)
! given a sorted vector return i such that x(i) < x < x(i+1)
                    use numz
            use galapagos
                    implicit none
                    type(thefit), intent(inout) :: vect(:)
                    real(b8) x
                    integer locate
                    integer low,mid,high,n
                    n=ubound(vect,1)
                    low=0
                    high=n+1
       10   if(high-low .gt. 1)then
                        mid=(high+low)/2
                        if((vect(n)%val .gt. vect( 1)%val) .eqv. &
                           (x             .gt. vect(mid)%val))      then
                            low=mid
                        else
                            high=mid
                        endif
                goto 10
            endif
                    locate=low
                end function locate
end module


subroutine roulette()
! reproduce using classic roulette wheel selection
! see fogels evolutionary computation page 91
  use numz
!  use problem
  use ran_mod
  use galapagos
  use control
  use locate_mod
  implicit none
  integer i,j,k,m,nstart,nend,t,j1,j2
  real(b8)tot,x1,x2,z1,z2
  integer local_pop,gene_size
  local_pop=num_genes
  gene_size=nstates
! sum the values
  tot=0.0
  do j=1,local_pop
      tot=tot+fit(j)%val
  enddo
  if(tot .eq. 0.0_b8)tot=1.0_b8
! scale the values
  do j=1,local_pop
      fit(j)%val=fit(j)%val/tot
  enddo
! assign area
  fit(local_pop)%val=1.0_b8-fit(local_pop)%val
  do j=local_pop-1,1,-1
      fit(j)%val=fit(j+1)%val-fit(j)%val
  enddo
  fit(1)%val=0.0_b8
  do j = 1 , local_pop
    x1=ran1()
    x2=ran1()
    k=locate(fit,x1)
    i=locate(fit,x2)
    z1=ran1()
    z2=ran1()
    nstart=nint((gene_size-1)*z1+1)
    nend=nint((gene_size-1)*z2+1)
    if(nend .eq. nstart)then
            do m=1 , nstart-1
                kids(m,j)=gene(m,fit(k)%index)
            enddo
            do m=nstart , gene_size
                kids(m,j)=gene(m,fit(i)%index)
            enddo
    else
        j1=min(nstart,nend)
        j2=max(nstart,nend)
            do m=1 , j1
                kids(m,j)=gene(m,fit(i)%index)
            enddo
            do m=j1+1 , j2-1
                kids(m,j)=gene(m,fit(k)%index)
            enddo
            do m=j2 , gene_size
                kids(m,j)=gene(m,fit(i)%index)
            enddo
    endif
  enddo
  gene=kids
  return
end subroutine roulette

