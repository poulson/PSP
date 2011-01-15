include make.inc

INCMUMPS = -I$(MUMPS_DIR)/include 
LIBMUMPS = -L$(MUMPS_DIR)/lib -l$(ARITH)mumps -lmumps_common

INC = $(INCMUMPS) $(IORDERINGS) $(INCSEQ)
LIB = $(LIBMUMPS) $(LORDERINGS) $(LIBSEQ) $(LIBBLAS) $(LIBFORT)

all:	d z

d:
	$(MAKE) ARITH=d dmumpsmex.mexglx
z:
	$(MAKE) ARITH=z zmumpsmex.mexglx

$(ARITH)mumpsmex.mexglx: mumpsmex.c
	$(MEX) $(OPTC) mumpsmex.c -output $@ -DMUMPS_ARITH=MUMPS_ARITH_$(ARITH) $(INC) $(LIB)
