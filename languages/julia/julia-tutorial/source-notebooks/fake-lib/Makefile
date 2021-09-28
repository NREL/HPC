# generally useful things

SHELL = /bin/sh

CC =		gcc
CCFLAGS =	-std=gnu99 -fPIC
DEPFLAGS =	-M

LINK =		gcc
SHLIBFLAGS =	-shared

EXE = test

LIB = silly

PLATFORM = $(shell uname)
ifeq ($(PLATFORM), Darwin)
	LIB_FULL = lib$(LIB).dylib
else
	LIB_FULL = lib$(LIB).so
endif

SRC = silly.c
INC = silly.h
MAIN = driver.c

OBJ = $(SRC:.c=.o)

# ---------------------------------------------------------------------
# Link target

default: run

run: exe
	./$(EXE)

clean:
	rm -f *.o
	rm -f *.d
	rm -f $(LIB_FULL)
	rm -f $(EXE)

lib: $(LIB)

exe: $(EXE)

$(EXE):	$(LIB)
	$(LINK) $(MAIN) $(LINKFLAGS) -o $(EXE) -L. -l$(LIB)

$(LIB): $(OBJ)
	$(LINK) $(LINKFLAGS) $(SHLIBFLAGS) -o $(LIB_FULL) $(OBJ)

# Compilation rules

%.o:%.c $(EXTRA_CPP_DEPENDS)
	$(CC) $(CCFLAGS) $(SHFLAGS) $(EXTRA_INC) -c $< -o $@

%.d:%.c $(EXTRA_CPP_DEPENDS)
	$(CC) $(CCFLAGS) $(EXTRA_INC) $(DEPFLAGS) $< > $@

# Individual dependencies

DEPENDS = $(OBJ:.o=.d)
sinclude $(DEPENDS)
