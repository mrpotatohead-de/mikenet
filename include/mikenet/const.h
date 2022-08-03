/*
    mikenet - a simple, fast, portable neural network simulator.
    Copyright (C) 1995  Michael W. Harm

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

    See file COPYING for a copy of the GNU General Public License.

    For more info, contact: 

    Michael Harm                  
    HNB 126 
    University of Southern California
    Los Angeles, CA 90089-2520

    email:  mharm@gizmo.usc.edu

*/

#ifndef CONST_H
#define CONST_H

#define MIKENET_MAJOR_VERSION 8
#define MIKENET_MINOR_VERSION 0
#define MIKENET_UPDATE_NUMBER 1
#define MIKENET_BUG_FIX_NUMBER 'a'

#define MIKENET_VERSION_TAG ""


#define Real float

#define ERR_NORMAL 1
#define ERR_OJA 2

/* bogus cludge */
#define VAL(x) (x > (-500.0))
#define SUM_SQUARED_ERROR 1
#define CROSS_ENTROPY_ERROR 2

#define NO_NOISE 0
#define ADDITIVE_NOISE 1
#define MULTIPLICATIVE_NOISE 2

#define SCALE_NONE 0
#define SCALE_PROB 1

#define LOGISTIC_ACTIVATION 0
#define TANH_ACTIVATION 1
#define FAST_LOGISTIC_ACTIVATION 2
#define LINEAR_ACTIVATION 3
#define STEP_ACTIVATION 4

#define RAMP_ERROR 1
#define NO_RAMP_ERROR 0

/* now some site-specific constants and stuff */

#ifdef HPC89
#ifndef unix
#define unix
#endif
#define COMPILER "c89"
#define MACHINE "HP"
#ifndef hpux
#define hpux
#endif
#endif

#ifdef _AIX
#ifndef unix
#define unix
#endif
#endif

#ifdef __unix__
#ifndef unix
#define unix
#endif
#endif


#ifdef _POWER
#ifndef MACHINE
#define MACHINE "IBM"
#endif
#endif

#ifdef hpux
#ifndef MACHINE
#define MACHINE "HP"
#endif
#endif

#ifdef convex
#define MACHINE "Convex"
#endif

#ifdef __convex__
#define MACHINE "Convex"
#endif

#ifdef __hp9000s800
#ifndef unix
#define unix
#endif
#ifndef MACHINE
#define MACHINE "HP 9000/800"
#endif
#endif

#ifdef __hp9000s300
#ifndef unix
#define unix
#endif
#ifndef MACHINE
#define MACHINE "HP 9000/300"
#endif
#endif

#ifdef __hp9000s700
#ifndef unix
#define unix
#endif
#ifndef MACHINE
#define MACHINE "HP 9000/700"
#endif
#endif

/* this states whether the "getrusage" system call exists */
/* undefine HAS_RUSAGE if it isn't on your system */
#if defined(unix) && !defined(HPC89)
#define HAS_RUSAGE 
#endif


/* HP doesn't include 
   the getrusage call.  I've hacked it in.  
   This is from the HP FAQ.   -mwh */
#if defined(hpux) && !defined(HPC89)
#include <sys/syscall.h>
#define getrusage(a, b)  syscall(SYS_GETRUSAGE, a, b)
#define HAS_RUSAGE
#endif /* hpux */


/* just in case your compiler
   doesn't define these macros.  -mwh  */

#ifndef __TIME__
#define __TIME__ "???"
#endif

#ifndef __DATE__
#define __DATE__ "???"
#endif

/* define the machine type */

#ifdef GNUC
#define COMPILER "gcc"
#endif

#ifdef __GNUC__
#define COMPILER "gcc"
#endif

#ifdef __DECC
#define COMPILER "DEC_C"
#endif

#ifdef sun
#define MACHINE "SUN"
#endif

#ifdef sgi
#define MACHINE "SGI"
#endif

#ifdef sun
#define MACHINE "SUN"
#endif

#ifdef linux
#define MACHINE "Linux"
#ifdef __i386__
#undef MACHINE
#define MACHINE "Linux/x86"
#endif
#ifdef __alpha
#undef MACHINE
#define MACHINE "Linux/Alpha"
#endif
#endif

/* metrowerks c compiler implies mac, cause there
   is no mac constant */
#ifdef __MWERKS__
#define mac
#endif

#ifdef _CRAYT3E
#define MACHINE "CRAY T3E"
#else
#if _CRAYT3D
#define MACHINE "CRAY T3D"
#else
#if _CRAY
#define MACHINE "CRAY"
#endif
#endif
#endif

#ifdef __powerpc__
#define MACHINE "PowerPC"
#endif

#ifdef mac
#if defined(__powerc) || defined(__powerpc__)
#define MACHINE "Macintosh (PowerPC)"
#else 
#if __MC68881__
#define MACHINE "Macintosh (68k, FPU)"
#else
#define MACHINE "Macintosh (68k, no fpu)"
#endif /* __MC68881__ */
#endif /* powerc */
#endif /* mac */

#ifdef sgi
#define MACHINE "SGI"
#endif

#ifdef ultrix
#define MACHINE "DEC"
#ifdef vax
#undef MACHINE
#define MACHINE "VAX"
#endif
#ifdef mips
#undef MACHINE
#define MACHINE "DEC/MIPS"
#endif
#endif

#ifdef __OpenBSD__
#define MACHINE "BSD"
#ifdef __I386__
#undef MACHINE
#define MACHINE "BSD/i386"
#endif
#endif

#ifdef __NetBSD__
#define MACHINE "NetBSD"
#ifdef __I386__
#undef MACHINE
#define MACHINE "NetBSD/i386"
#endif
#endif

#ifdef __FreeBSD__
#define MACHINE "FreeBSD"
#ifdef __I386__
#undef MACHINE
#define MACHINE "FreeBSD/i386"
#endif
#endif

#ifndef MACHINE
#define MACHINE "<unknown>"
#endif

#ifndef COMPILER 
#define COMPILER "native compiler"
#endif

extern int default_useBlas;

#endif /* CONST_H */
