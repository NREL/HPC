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

