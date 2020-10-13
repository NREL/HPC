subroutine init_genes(jean)
    use numz
    use ran_mod
    use control
    implicit none
    integer,dimension(:,:)::jean
    real(b8) y
    integer i,j,k(2)
    k=ubound(jean) ! return a 2d array with both upper bounds
    do i=1,k(1)
      do j=1,k(2)
        jean(i,j)=int(spread(0.0_b8,real(ncolor,b8))) ! gives us range of 0-ncolor-1
      enddo
    enddo
end subroutine


! our function that returns a vector
! take an integer input vector which represents
! an integer in some base and adds 1 
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
  do while(carry)
      i=i+1
      rtn(i)=rtn(i)+1
      if(rtn(i) .gt. max)then
          if(i == len)then
              rtn=0
          else
              rtn(i)=0
          endif
      else
          carry=.false.
      endif
  enddo
end function

