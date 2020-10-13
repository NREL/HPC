module control
    use numz
    use galapagos
    real(b8) mute_rate
    real(b8) fitmax
    integer  ncolor
    integer  num_gen
    integer  num_genes
    integer  nstates
    integer  seed
    logical  force
    logical  the_top
    logical  do_one

    integer,allocatable :: gene(:,:) ! our genes for the ga    
    integer,allocatable :: kids(:,:) ! temp for next generation
    type (thefit),allocatable ,target :: fit(:) 

end module

