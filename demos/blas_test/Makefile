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

#to use gcc, use these settings 
CC = ${MYCC}
DEBUG = ${MYDEBUG}
LINK_ARGS= 
####SPECIAL= -funroll-loops
SPECIAL=

# a generic c compiler setting
#CC = cc
#DEBUG= -O
#LINK_ARGS=  

### note: if mikenet is installed at the system level,
### (like, /usr/local/include, /usr/local/lib) then you don't need
### this gunk.  

## this uses ATLAS
LIBS="-L${MIKENET_DIR}/lib/${ARCH} -L/home/mharm/CBLAS/lib/LINUX -lmikenet -lcblas -lblas  -lm"

##LIBS="/home/mharm/Mikenet/lib/i686/libmikenet.a ${HOME}/ATLAS/lib/Linux_PIIISSE1/libcblas.a ${HOME}/ATLAS/lib/Linux_PIIISSE1/libatlas.a -lm"


MIKENETLIB=${MIKENET_DIR}/lib/${ARCH}/libmikenet.a


INCLUDEHOME=${MIKENET_DIR}/include

all:	ps_atlas_sse ps_atlas_PIII ps_atlas_3dnow ps_vanilla


ps_blas:	eval_hearing.c hearing_model.c Makefile
	gcc -o ps_blas eval_hearing.c hearing_model.c ${DEBUG} -DBLAS -I${INCLUDEHOME} ${MIKENETLIB} ${HOME}/CBLAS/lib/LINUX/cblas_LINUX.a ${HOME}/install/blas/blas_src/libmyblas.a -lm

ps_atlas_sse:	eval_hearing.c hearing_model.c Makefile
	gcc -o ps_atlas_sse eval_hearing.c hearing_model.c ${DEBUG} -DBLAS -I${INCLUDEHOME} ${MIKENETLIB} ${HOME}/ATLAS/lib/Linux_PIIISSE1/libcblas.a ${HOME}/ATLAS/lib/Linux_PIIISSE1/libatlas.a -lm

ps_vanilla:	eval_hearing.c hearing_model.c Makefile
	gcc -o ps_vanilla eval_hearing.c hearing_model.c ${DEBUG} -I${INCLUDEHOME} ${MIKENETLIB} -lm

ps_atlas_PIII:	eval_hearing.c hearing_model.c Makefile
	gcc -o ps_atlas_PIII eval_hearing.c hearing_model.c ${DEBUG} -DBLAS -I${INCLUDEHOME} ${MIKENETLIB} ${HOME}/ATLAS/lib/Linux_PIII/libcblas.a ${HOME}/ATLAS/lib/Linux_PIII/libatlas.a -lm

ps_atlas_native:	eval_hearing.c hearing_model.c Makefile
	gcc -o ps_atlas_native eval_hearing.c hearing_model.c ${DEBUG} -DBLAS -I${INCLUDEHOME} ${MIKENETLIB} /usr/lib/3dnow/libcblas.so.2 /usr/lib/3dnow/libatlas.so.2 -lm

ps_atlas_3dnow:	eval_hearing.c hearing_model.c Makefile
	gcc -o ps_atlas_3dnow eval_hearing.c hearing_model.c ${DEBUG} -DBLAS -I${INCLUDEHOME} ${MIKENETLIB} ${HOME}/ATLAS_3DNOW/lib/Linux_ATHLON3DNow2/libcblas.a ${HOME}/ATLAS_3DNOW/lib/Linux_ATHLON3DNow2/libtstatlas.a ${HOME}/ATLAS_3DNOW/lib/Linux_ATHLON3DNow2/libatlas.a -lm

clean:
	rm -f benchmark *~ *.o benchit ps_*



