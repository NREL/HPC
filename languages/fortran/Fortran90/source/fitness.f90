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
    fitness=(100.0_b8-tot)/100.0_b8
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


