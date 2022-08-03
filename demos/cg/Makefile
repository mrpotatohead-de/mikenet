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
CC = gcc 
DEBUG = -O3 -Wall
LINK_ARGS= 
SPECIAL= -funroll-loops

# a generic c compiler setting
#CC = cc
#DEBUG= -O
#LINK_ARGS=  

### note: if mikenet is installed at the system level,
### (like, /usr/local/include, /usr/local/lib) then you don't need
### this gunk.  

LIBHOME=${MIKENET_DIR}/lib/${ARCH}/
INCLUDEHOME=${MIKENET_DIR}/include

.c:	
	${CC} -o $@ $< ${DEBUG} -I${INCLUDEHOME} -L${LIBHOME} -lmikenet -lm

clean:
	rm -f cg *~
