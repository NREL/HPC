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

! our linked list print routine
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
