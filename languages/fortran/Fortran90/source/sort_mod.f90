module merge_mod_types
    use galapagos
    type(thefit),allocatable :: work(:)
    type(thefit), pointer:: a_pntr(:)
end module merge_mod_types

module sort_mod
!defining the interfaces
  interface operator (.lt.)  ! overloads standard .lt.
    module procedure theless ! the function that does it
  end interface

  interface operator (.gt.)   ! overloads standard .gt.
    module procedure thegreat ! the function that does it
  end interface

  interface operator (.ge.)  ! overloads standard .ge. 
    module procedure thetest ! the function that does it
  end interface

  interface operator (.converged.)  ! new operator 
    module procedure index_test     ! the function that does it
  end interface

contains


  function theless(a,b) ! overloads < for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical theless
    if(a%val < b%val)then ! this is where we do the test
        theless=.true.
    else
        theless=.false.
    endif
    return
  end function theless

  function thegreat(a,b) ! overloads > for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical thegreat
    if(a%val > b%val)then
        thegreat=.true.
    else
        thegreat=.false.
    endif
    return
  end function thegreat


function thetest(a,b)   ! overloads >= for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical thetest
    if(a%val >= b%val)then
        thetest=.true.
    else
        thetest=.false.
    endif
    return
end function thetest

function index_test(a,b) ! defines a new operation for the type (thefit)
    use galapagos
    implicit none
    type(thefit), intent (in) :: a,b
    logical index_test
    if(a%index > b%index)then ! check the index value for a difference
        index_test=.true.
    else
        index_test=.false.
    endif
    return
end function index_test

subroutine Sort(ain, n)
    use Merge_mod_types
    implicit none
    integer n
    type(thefit), target:: ain(n)
    allocate(work(n))
    nullify(a_pntr)
    a_pntr=>ain
    call RecMergeSort(1,n)
    deallocate(work)
    return
end subroutine Sort

! recursive merge sort taken from pascal routine in
! moret & shapiro
! algorithms from p to np, volume 1, design and efficiency
! benjamin/cummings 1991

recursive subroutine RecMergeSort(left, right)
    use Merge_mod_types
    implicit none
    integer,intent(in):: left,right
    integer  middle
    if (left < right) then
        middle = (left + right) / 2
        call RecMergeSort(left,middle)
        call RecMergeSort(middle+1,right)
        call Merge(left,middle-left+1,right-middle)
    endif
    return
end subroutine RecMergeSort

subroutine Merge(s, n, m)
    use Merge_mod_types
    implicit none
    integer s,n,m
    integer i,  j, k, t, u
    k = 1
    t = s + n
    u = t + m
    i = s
    j = t
    if ((i < t) .and. (j < u))then
        do while ((i < t) .and. (j < u))
            if (a_pntr(i) .ge.  a_pntr(j))then
                work(k) = a_pntr(i)
                i = i + 1
                k = k + 1
            else
                work(k) = a_pntr(j)
                j = j + 1
                k = k + 1 
            endif
         enddo
    endif
    if(i < t )then
        do while (i < t )
            work(k) = a_pntr(i)
            i = i + 1
            k = k + 1
        enddo
    endif
    if(j < u)then
        do while (j < u )
            work(k) = a_pntr(j)
            j = j + 1
            k = k + 1
        enddo
    endif
    i = s
! the next line is not in moret & shapiro's book
! but should be
    k=k-1
    do j = 1 , k  
        a_pntr(i) = work(j)
        i = i + 1
    enddo
    return
end subroutine Merge


end module sort_mod





