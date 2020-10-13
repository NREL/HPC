##### for ibm sp #####

## file name extension ##
F= f

## link line ##
L= xlf90 -O3

## compile line ##
P= xlf90 -c -O3

## copies *.f90 to *.f ##
.SUFFIXES: .f90
.f90.f:
	cp $< $*.f

##### end for ibm sp #####

##### for sgi #####
#SIZE= -64
##### for cray/sgi #####

## file name extension ##
F= f90 

## link line ##
L= f95 $(SIZE) -g

## compile line ##
P= f95 -c $(SIZE) -g

##### end for cray/sgi #####

OBJ1= face.o          control.o       more.o          reproduce.o
OBJ2= fitness.o       linked.o        mutate.o        rest.o
OBJ3= galapagos.o     list_stuff.o    numz.o          sort_mod.o
OBJ4= global_test.o   map_stuff.o     ran_mod.o 
OBJS= $(OBJ1) $(OBJ2) $(OBJ3) $(OBJ4)

darwin: $(OBJS)
	$(L) -o darwin $(OBJS)

face.o: face.$(F) galapagos.o numz.o sort_mod.o
	$(P) face.$(F)

fitness.o: fitness.$(F) global_test.o list_stuff.o map_stuff.o numz.o
	$(P) fitness.$(F)

galapagos.o: galapagos.$(F) numz.o
	$(P) galapagos.$(F)

control.o: control.$(F) galapagos.o numz.o
	$(P) control.$(F)

linked.o:linked.$(F) list_stuff.o
	$(P) linked.$(F)

list_stuff.o:list_stuff.$(F)
	$(P) list_stuff.$(F)

map_stuff.o: map_stuff.$(F) global_test.o list_stuff.o
	$(P) map_stuff.$(F)

more.o: more.$(F) control.o numz.o ran_mod.o
	$(P) more.$(F)

mutate.o: mutate.$(F) control.o galapagos.o numz.o ran_mod.o
	$(P) mutate.$(F)

ran_mod.o: ran_mod.$(F) numz.o
	$(P) ran_mod.$(F)

reproduce.o: reproduce.$(F) control.o galapagos.o  numz.o ran_mod.o
	$(P) reproduce.$(F)

rest.o: rest.$(F) control.o face.o list_stuff.o map_stuff.o numz.o sort_mod.o
	$(P) rest.$(F)

sort_mod.o: sort_mod.$(F) galapagos.o
	$(P) sort_mod.$(F)

numz.o: numz.$(F)
	$(P) numz.$(F)

global_test.o: global_test.$(F)
	$(P) global_test.$(F)


clean:
	rm  -f *.mod *.o darwin

