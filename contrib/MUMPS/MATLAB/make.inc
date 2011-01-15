MEX       = /opt/matlab/bin/mex -g
MUMPS_DIR = $(HOME)/MUMPS

LMETISDIR = $(HOME)/metis-4.0
LMETIS = -L$(LMETISDIR) -lmetis
LPORDDIR = $(MUMPS_DIR)/PORD/lib
LPORD = -L$(LPORDDIR) -lpord
LORDERINGS = $(LPORD) $(LMETIS)

LIBSEQ  = -L$(MUMPS_DIR)/libseq -lmpiseq

LIBFORT = /usr/local/lib/libgfortran.a

LIBBLAS = -L/usr/lib/atlas -lblas

OPTC    = -O 

