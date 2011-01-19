#
#  This file is part of MUMPS 4.9.2, built on Thu Nov  5 07:05:08 UTC 2009
#
topdir = .
libdir = $(topdir)/lib

default:	dexamples

.PHONY: default alllib all s d c z \
	sexamples dexamples cexamples zexamples \
	mumps_lib requiredobj libseqneeded clean

alllib:		s d c z
all:		sexamples dexamples cexamples zexamples

s:
	$(MAKE) ARITH=s mumps_lib
d:
	$(MAKE) ARITH=d mumps_lib
c:
	$(MAKE) ARITH=c mumps_lib
z:
	$(MAKE) ARITH=z mumps_lib


# Is Makefile.inc available ?
Makefile.inc:
	@echo "######################################################################"
	@echo "# BEFORE COMPILING MUMPS, YOU SHOULD HAVE AN APPROPRIATE FILE"
	@echo "# Makefile.inc AVALAIBLE. PLEASE LOOK IN THE DIRECTORY ./Make.inc FOR"
	@echo "# EXAMPLES OF Makefile.inc FILES, AT Make.inc/Makefile.inc.generic"
	@echo "# IN CASE YOU NEED TO BUILD A NEW ONE AND READ THE MAIN README FILE"
	@echo "######################################################################"
	@exit 1

include Makefile.inc

mumps_lib: requiredobj
	(cd src ; $(MAKE) $(ARITH))

sexamples:	s
	(cd examples ; $(MAKE) s)

dexamples:	d
	(cd examples ; $(MAKE) d)

cexamples:	c
	(cd examples ; $(MAKE) c)

zexamples:	z
	(cd examples ; $(MAKE) z)


requiredobj: Makefile.inc $(LIBSEQNEEDED) $(libdir)/libpord$(PLAT).a

# dummy MPI library (sequential version)

libseqneeded:
	(cd libseq; $(MAKE))

clean:
	(cd src; $(MAKE) clean)
	(cd examples; $(MAKE) clean)
	(cd $(libdir); $(RM) *$(PLAT).a)
	(cd libseq; $(MAKE) clean)

