program check_mpi_version
  use mpi

  ! https://www.open-mpi.org/doc/v3.1/man3/MPI_Get_library_version.3.php

  implicit none
  integer :: ierr, resultlen
  character(len=200) :: version

  call MPI_Init(ierr)
  call MPI_Get_library_version(version, resultlen, ierr)

  if (ierr == 0) then
    print *, "MPI Library Version: ", version
  else
    print *, "Error getting MPI library version"
  end if

  call MPI_Finalize(ierr)

end program check_mpi_version
