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

#ifndef MIKENET_MPI_H
#define MIKENET_MPI_H



void parallel_finish();
float * parallel_get_weight_buffer(Net *net);
int parallel_proc_id();
int parallel_num_procs();
void parallel_proc_name(char *out);
void parallel_sync();
void parallel_init(int *argc,char ***argv);
void parallel_broadcast_weights(Net *net);
void parallel_sum_gradients(Net *net);

/* do forward and backward computation */
float parallel_g(Net *net,ExampleSet *examples,float p);

float parallel_sum_float(float x);
int parallel_sum_int(int x);

float parallel_wall_clock_time();

/* do just forward computation */
float parallel_f(Net *net,ExampleSet *examples,float p);


#endif 
