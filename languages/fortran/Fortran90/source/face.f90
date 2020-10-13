module face 
    interface fitness 
        function fitness(vector) 
          use numz 
          implicit none 
          real(b8) fitness 
          integer, dimension(:),target ::  vector 
        end function fitness 
    end interface 

    interface add1  ! interface for our function 
                    ! that returns a vector
        function add1(vector,max) result (rtn)
        integer, dimension(:),intent(in) ::  vector
        integer max
        integer,dimension(size(vector)) :: rtn
        end function
    end interface

    interface findmin
! note we have two functions within the same interface
! this is how we indicate function overloading
! both functions are called "findmin" in the main program
!
! the first is called with an array of reals as input
        recursive function realmin(ain) result (themin) 
          use numz
          real(b8) themin
          real(b8) ,dimension(:) :: ain
        end function

! the second is called with a array of data structures as input
        recursive function typemin(ain) result (themin) 
          use numz
          use galapagos
          real(b8) themin
          type (thefit) ,dimension(:) :: ain
        end function
    end interface

    interface pntmin
        recursive function pntmin(ain) result (themin) 
          use numz
          use galapagos
          use sort_mod
          type (thefit),pointer:: themin
          type (thefit) ,dimension(:),target :: ain
        end function
     end interface

     interface boink1
         subroutine boink1(a,n)
         use numz
         integer, intent(in):: n
         real(b8),dimension(n:):: a
         end subroutine
     end interface

     interface boink2
         subroutine boink2(a,n)
         use numz
         integer, intent(in):: n
         real(b8),dimension(n:):: a
         end subroutine
     end interface

     interface boink3
         subroutine boink3(a)
         use numz
         real(b8),dimension(:),pointer:: a
         end subroutine
     end interface

     interface init_genes
         subroutine init_genes(jean)
         integer,dimension(:,:):: jean
         end subroutine
     end interface

end module 

