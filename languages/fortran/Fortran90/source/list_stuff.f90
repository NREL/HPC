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
