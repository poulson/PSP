#
#  This file is part of MUMPS 4.9.2, built on Thu Nov  5 07:05:08 UTC 2009
#
topdir = ..
libdir = $(topdir)/lib

default: d

.PHONY: default all s d c z clean
.SECONDEXPANSION:

all:	s d c z

s:	ssimpletest
d:	dsimpletest c_example
c:	csimpletest
z:	zsimpletest


include $(topdir)/Makefile.inc

LIBMUMPS_COMMON = $(libdir)/libmumps_common$(PLAT).a


LIBSMUMPS = $(libdir)/libsmumps$(PLAT).a $(LIBMUMPS_COMMON)

ssimpletest:  $(LIBSMUMPS) $$@.o 
	$(FL) -o $@ $(OPTL) ssimpletest.o $(LIBSMUMPS)  $(LORDERINGS) $(LIB) $(LIBBLAS) $(LIBOTHERS)


LIBDMUMPS = $(libdir)/libdmumps$(PLAT).a $(LIBMUMPS_COMMON)

dsimpletest: $(LIBDMUMPS) $$@.o 
	$(FL) -o $@ $(OPTL) dsimpletest.o $(LIBDMUMPS)  $(LORDERINGS) $(LIB) $(LIBBLAS) $(LIBOTHERS)


LIBCMUMPS = $(libdir)/libcmumps$(PLAT).a $(LIBMUMPS_COMMON)

csimpletest: $(LIBCMUMPS) $$@.o 
	$(FL) -o $@ $(OPTL) csimpletest.o $(LIBCMUMPS)  $(LORDERINGS) $(LIB) $(LIBBLAS) $(LIBOTHERS)


LIBZMUMPS = $(libdir)/libzmumps$(PLAT).a $(LIBMUMPS_COMMON)

zsimpletest: $(LIBZMUMPS) $$@.o 
	$(FL) -o $@ $(OPTL) zsimpletest.o $(LIBZMUMPS)  $(LORDERINGS) $(LIB) $(LIBBLAS) $(LIBOTHERS)


c_example:	$(LIBDMUMPS) $$@.o
	$(FL) -o $@ $(OPTL) $@.o $(LIBDMUMPS) $(LORDERINGS) $(LIB) $(LIBBLAS) $(LIBOTHERS)


.SUFFIXES: .c .F .o
.F.o:
	$(FC) $(OPTF) $(INC) -I. -I$(topdir)/include -c $*.F
.c.o:
	$(CC) $(OPTC) $(INC) -I. -I$(topdir)/include -c $*.c


$(libdir)/libsmumps$(PLAT).a:
	@echo 'Error: you should build the library' $@ 'first'
	exit -1

$(libdir)/libdmumps$(PLAT).a:
	@echo 'Error: you should build the library' $@ 'first'
	exit -1

$(libdir)/libcmumps$(PLAT).a:
	@echo 'Error: you should build the library' $@ 'first'
	exit -1

$(libdir)/libzmumps$(PLAT).a:
	@echo 'Error: you should build the library' $@ 'first'
	exit -1

$(LIBMUMPS_COMMON):
	@echo 'Error: you should build the library' $@ 'first'
	exit -1

clean:
	$(RM) *.o [sdcz]simpletest c_example
