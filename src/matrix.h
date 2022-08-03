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


#ifndef MIKENET_MATRIX_H
#define MIKENET_MATRIX_H 1

void mikenet_matrix_vec_mult(Real * outvec,int nout,Real *invec,
			  int nin,Real **mat);

void mikenet_matrix_vec_mult_t(Real * outvec,int nout,Real *invec,
			  int nin,Real **mat);

void mikenet_matrix_outer_product(Real ** matrix,
				  Real * v1,
				  int n1,
				  Real * v2,
				  int n2);
	
extern int default_useBlasThreshold;

#endif
