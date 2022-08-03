##########################################################################
#    mikenet - a simple, fast, portable neural network simulator.
#    Copyright (C) 1995  Michael W. Harm
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
#    See file COPYING for a copy of the GNU General Public License.
#
#    For more info, contact: 
#
#    Michael Harm                  
#    HNB 126 
#    University of Southern California
#    Los Angeles, CA 90089-2520
#
#    email:  mharm@gizmo.usc.edu
#
##########################################################################
##
## make <target> where <target> is one of linux,ansi,cc,gcc,sparc,hp,cray,t3e
##        - builds the library in place
##        - copies library and header files over to appropriate 
##          subdirectory in build tree
##
##########################################################################

foo:
	@echo "To build the simulator, type 'make' followed by "
	@echo "your architecture type"
	@echo " "
	@echo " For example, choose from:"
	@echo "    gcc, linux, cc, ansi, sgi, hp, sparc, "
	@echo "    linux_mpi, t3e, cray"
	@echo " "
	@echo "If in doubt, use 'ansi'"

BLAS_INC="-I/home/mharm/ATLAS/include"

##########################################
## linux pentium/pentium pro 
## (I'm told it won't work on a cyrix chip; for those use vanilla gcc)
##########################################
linux:
	make all CC=gcc DEBUG="-O3 -Wall -m486 -fomit-frame-pointer -fno-strength-reduce -malign-loops=2 -malign-functions=2 -malign-jumps=2" SPECIAL="-funroll-loops"

linux_blas:
	make all CC=gcc DEBUG="-DUSE_BLAS ${BLAS_INC} -O3 -Wall -m486 -fomit-frame-pointer -fno-strength-reduce -malign-loops=2 -malign-functions=2 -malign-jumps=2" SPECIAL="-funroll-loops"

linux_gcc3:
	make all CC=gcc3 DEBUG="-O3 -fsched-spec -fsched-spec-load -fschedule-insns2 -ffast-math -Wall -march=i686 -fomit-frame-pointer -fno-strength-reduce -malign-loops=2 -malign-functions=2 -malign-jumps=2" SPECIAL="-funroll-loops -frerun-loop-opt -frerun-cse-after-loop"

linux_intel:
	make all CC=icc DEBUG="-O3 -axK -xK -tpp6" SPECIAL="-unroll 10 -prefetch -Qvec -vec_report3"

linux_prof:
	make all CC=gcc DEBUG="-DUSE_BLAS -pg ${BLAS_INC} -O3 -Wall -m486 -fno-strength-reduce -malign-loops=2 -malign-functions=2 -malign-jumps=2" SPECIAL=-funroll-loops

linux_alpha:
	make all CC=gcc DEBUG="-O3 -Wall -fomit-frame-pointer -fno-strength-reduce" SPECIAL=-funroll-loops

## this is the location of the MPI installation
MPI_DIR_ALPHA=/home/mharm/mpich-1.2.1/${HOSTTYPE}
linux_alpha_mpi:
	make all CC=ccc DEBUG="-I${MPI_DIR_ALPHA}/include -DUSE_MPI -float -fp_reorder -O3 -fast -tune ev6 -arch ev6" SPECIAL="-unroll 5"

## this is the location of the MPI installation
MPI_DIR_ALPHA=/home/mharm/mpich-1.2.1/${HOSTTYPE}
gcc_alpha_mpi:
	make all CC=gcc DEBUG="-I${MPI_DIR_ALPHA}/include -DUSE_MPI -Wall -fomit-frame-pointer -fno-strength-reduce" SPECIAL="-funroll-loops"

ccc_alpha:
	make all CC=ccc DEBUG="-float -fp_reorder -O3 -fast -tune ev6 -arch ev6" SPECIAL="-unroll 5"

##########################################
## linux, with parallel MPI extensions.  
##########################################
## this is the location of the MPI installation
MPI_DIR_LINUX=${HOME}/mpich-1.2.1/i386-linux

linux_mpi:
	make all CC=gcc DEBUG="-I${MPI_DIR_LINUX}/include -DUSE_MPI -O3 -Wall -m486 -fomit-frame-pointer -fno-strength-reduce -malign-loops=2 -malign-functions=2 -malign-jumps=2" SPECIAL=-funroll-loops

linux_blas_mpi:
	make all CC=gcc DEBUG="-DUSE_BLAS ${BLAS_INC} -I${MPI_DIR_LINUX}/include -DUSE_MPI -O3 -Wall -m486 -fomit-frame-pointer -fno-strength-reduce -malign-loops=2 -malign-functions=2 -malign-jumps=2" SPECIAL=-funroll-loops

##############################################

egcs:
	make all CC=egcs DEBUG="-O9 -Wall -fprofile-arcs -march=pentiumpro -ffast-math -fbranch-probabilities" SPECIAL="-funroll-loops"


### if you have gnu cc, probably should use this.
gcc:	
	make all CC=gcc DEBUG="-O3 -Wall" SPECIAL="-funroll-loops"

debug:	
	make all CC=gcc DEBUG="-g -Wall" SPECIAL=""

debug_mpi:	
	make all CC=gcc DEBUG="-DUSE_MPI -I${MPI_DIR}/include -g -Wall" SPECIAL=""

## vanilla ansi c compiler
cc:
	make all CC=cc DEBUG=-O2

cc_alpha:
	make all CC=c89 DEBUG="-float -O3 -tune ev6 -fp_reorder -arch ev6" SPECIAL="-unroll 5"

MPI_DIR_ALPHA=/home/mharm/mpich-1.2.1/alpha
cc_alpha_mpi:
	make all CC=c89 DEBUG="-float -O3 -I${MPI_DIR_ALPHA}/include -tune ev6 -DUSE_MPI -fp_reorder -arch ev6" SPECIAL="-unroll 5"

cc_alpha_old:
	make all CC=c89 DEBUG="-O4 -tune ev6 " SPECIAL="-unroll 0"

cc_alpha_wimpy:
	make all CC=c89 DEBUG="-O2 -tune ev6 -D_FASTMATH -fptm n -fptm u" 

## Cray T3E (we assume its got MPI)
t3e:
	make all CC=cc DEBUG="-DUSE_MPI -DCRAY -O2" SPECIAL="-h unroll"

t3e_debug:
	make all CC=cc DEBUG="-happrentice -DUSE_MPI -DCRAY -O2" SPECIAL="-h unroll"


## Cray T3E (we assume its got MPI)
cray:
	make all CC=cc DEBUG="-DUSE_MPI -DCRAY -O2" SPECIAL="-h unroll"

## convex exemplar
convex:
	make all CC=cc DEBUG="-O -DCONVEX"

## SGI machines don't use ranlib
sgi:
	make all RANLIB=""

### another way of saying 'cc'
ansi:	cc

### for HP 735 workstations
hp:
	make all CC=c89 DEBUG="+O4 -DHPC89" LINK_ARGS=+O4 RANLIB=""

## hp with mpi extensions 
hp_mpi:
	make all CC=c89 DEBUG="-DUSE_MPI -I${MPI_DIR}/include +O4 -DHPC89" LINK_ARGS=+O4 RANLIB=""

## for sun sparc 
sparc:
	make all CC=cc DEBUG="-fast -xO5" SPECIAL="-xunroll=10" 


### some generic settings
CC=cc
DEBUG= -O
LINK_ARGS=
SPECIAL=


############################################
## Shouldn't have to edit things below here 
############################################

AR= ar
LIB=libmikenet.a
RANLIB= ranlib ${LIB}
LIBHOME=${MIKENET_DIR}/lib/${ARCH}
INCLUDEHOME=${MIKENET_DIR}/include

MIKENET_LIB_OBJS= example.o net.o tools.o crbp.o \
	random.o token.o parallel.o \
	bptt.o linesearch.o weights.o error.o stats.o  apply.o \
	analyze.o dbm.o elman.o benchmark.o dotprod.o matrix.o fastexp.o

all:	${LIB} install


${LIB}:	${MIKENET_LIB_OBJS}
	${AR} r ${LIB} $?
	${RANLIB}

lib:	${LIB}

install: 
	cp ${LIB} ${LIBHOME}
	cp *.h ${INCLUDEHOME}/mikenet

clean:
	rm -f *.o mikenet *.a *~ #* *.T

depend:
	makedepend -Y *.c *.h 

bptt.o:	bptt.c
	${CC} ${DEBUG} -c bptt.c

elman.o:	elman.c
	${CC} ${DEBUG} -c elman.c

crbp.o:	crbp.c
	${CC} ${DEBUG}  -c crbp.c

dotprod.o:	dotprod.c
	${CC} ${DEBUG} ${SPECIAL} -c dotprod.c

matrix.o:	matrix.c
	${CC} ${DEBUG} ${SPECIAL} -c matrix.c

dbm.o:	dbm.c
	${CC} ${DEBUG} -c dbm.c

.c.o:
	${CC} ${DEBUG}  -c $<

.c:	${LIB} 
	${CC} ${DEBUG} ${SPECIAL} -o $@ $< ./libmikenet.a -lm

# DO NOT DELETE

analyze.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
analyze.o: ex_struct.h ex_set_struct.h tools.h random.h analyze.h
apply.o: const.h net.h groupstruct.h connstruct.h netstruct.h weights.h
apply.o: tools.h
benchmark.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
benchmark.o: ex_struct.h ex_set_struct.h tools.h random.h benchmark.h bptt.h
benchmark.o: crbp.h
bptt.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
bptt.o: ex_struct.h ex_set_struct.h weights.h bptt.h random.h tools.h error.h
crbp.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
crbp.o: ex_struct.h ex_set_struct.h weights.h bptt.h crbp.h random.h tools.h
crbp.o: error.h dotprod.h matrix.h
dbm.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
dbm.o: ex_struct.h ex_set_struct.h weights.h dbm.h bptt.h random.h tools.h
dbm.o: error.h
dotprod.o: const.h dotprod.h
elman.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
elman.o: ex_struct.h ex_set_struct.h weights.h bptt.h random.h tools.h
elman.o: elman.h
error.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
error.o: ex_struct.h ex_set_struct.h error.h tools.h
example.o: const.h net.h groupstruct.h connstruct.h netstruct.h random.h
example.o: tools.h example.h ex_struct.h ex_set_struct.h
fastexp.o: fastexp.h
linesearch.o: const.h net.h groupstruct.h connstruct.h netstruct.h tools.h
linesearch.o: weights.h linesearch.h
matrix.o: const.h matrix.h
net.o: const.h net.h groupstruct.h connstruct.h netstruct.h tools.h random.h
parallel.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
parallel.o: ex_struct.h ex_set_struct.h tools.h weights.h error.h crbp.h
parallel.o: parallel.h
random.o: random.h
stats.o: stats.h
token.o: token.h
tools.o: const.h net.h groupstruct.h connstruct.h netstruct.h example.h
tools.o: ex_struct.h ex_set_struct.h tools.h random.h fastexp.h
weights.o: const.h net.h groupstruct.h connstruct.h netstruct.h tools.h
weights.o: weights.h random.h
dotprod.o: const.h
example.o: ex_struct.h ex_set_struct.h
net.o: groupstruct.h connstruct.h netstruct.h
