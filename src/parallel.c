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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "const.h"
#include "net.h"
#include "example.h"
#include "tools.h"
#include "weights.h"
#include "error.h"
#include "crbp.h"
#include "parallel.h"

#ifdef USE_MPI
#include <mpi.h>
#endif



static char *processor_name;
static int myid=-1;
static int numprocs=-1;
#ifdef USE_MPI
static float startTime=0;
#endif



int parallel_sum_int(int x)
{
  int out=0;
#ifdef USE_MPI
  int in=x;
  MPI_Reduce(&in, &out, 1, MPI_INT, MPI_SUM, 
             0, MPI_COMM_WORLD);  
  MPI_Bcast(&out,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
  return out;
}



float parallel_sum_float(float x)
{
  float out=0.0;
#ifdef USE_MPI
  float in=x;
  MPI_Reduce(&in, &out, 1, MPI_FLOAT, MPI_SUM, 
	     0, MPI_COMM_WORLD);  
  MPI_Bcast(&out,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return out;
}


/* do a forward (and maybe backward) propagation of net using examples */
float parallel_f0(Net *net,ExampleSet *examples,float p,int dograd)
{
  float global_e=0.0;
#ifdef USE_MPI

  int i;
  int from,end;
  float e=0.0,r,e0;
  Example  *ex;

  set_error_scale(net,p);

  r=(float)examples->numExamples / (float)numprocs;
  from = r * myid;
  end = r * (myid+1);

  for(i=from;i<end;i++)
    {
      ex=&examples->examples[i];
      crbp_forward(net,ex);
      if (dograd)
	crbp_compute_gradients(net,ex);
      e0= compute_error(net,ex);
      e+=e0;
    }

  MPI_Reduce(&e, &global_e, 1, MPI_FLOAT, MPI_SUM, 
	     0, MPI_COMM_WORLD);  
  MPI_Bcast(&global_e,1,MPI_FLOAT,0,MPI_COMM_WORLD);

#endif
  return global_e;
}

/* do a forward and backward propagation of net using examples */
float parallel_g(Net *net,ExampleSet *examples,float p)
{
  return parallel_f0(net,examples,p,1);
}

/* do a forward propagation of net using examples */
float parallel_f(Net *net,ExampleSet *examples,float p)
{
  return parallel_f0(net,examples,p,0);
}



void parallel_init(int *argc,char ***argv)
{
#ifdef USE_MPI  
  int namelen;
  MPI_Init(argc,argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  processor_name=(char *)mh_malloc(MPI_MAX_PROCESSOR_NAME);
  MPI_Get_processor_name(processor_name,&namelen);
  startTime=MPI_Wtime();
#else
  fprintf(stderr,"Mikenet Library not compiled with parallel extensions.\n");
  fprintf(stderr,"Recompile the library with USE_MPI defined\n");
  exit(-1);
#endif
}


float parallel_wall_clock_time()
{
#ifdef USE_MPI
  return (float)(MPI_Wtime()-startTime);
#else
  return 0.0;
#endif
}


void parallel_finish()
{
#ifdef USE_MPI
  MPI_Finalize();
#endif
}

void parallel_sync()
{
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}


int parallel_proc_id()
{
  return myid;
}

int parallel_num_procs()
{
  return numprocs;
}

void parallel_proc_name(char *n)
{
  strcpy(n,processor_name);
}


void parallel_broadcast_weights(Net *net)
{
#ifdef USE_MPI
  Connections *c;
  int i;

  for(i=0;i<net->numConnections;i++)
    {
      c=net->connections[i];
      MPI_Bcast(&c->weights[0][0],
		c->from->numUnits * c->to->numUnits,
		MPI_FLOAT,0,MPI_COMM_WORLD);
    }
  
#endif
}

/* at the conclusion of this call, processor id =0 has the gradients */
/* of all the procs */
void parallel_sum_gradients(Net *net)
{
#ifdef USE_MPI
  static float *to=NULL;
  static int currSize=0;
  int numweights,i;
  Connections *c;
  float *from;
  for(i=0;i<net->numConnections;i++)
    {
      c=net->connections[i];
      if (c->locked)
	continue;


      numweights=c->from->numUnits * c->to->numUnits;
      if (to==NULL)
	{
	  to=(float *)mh_malloc(sizeof(float)*numweights);
	  currSize=numweights;
	}
      else if (numweights > currSize)  /* not big enough */
	{
	  to=(float *)mh_realloc(to,sizeof(float)*numweights);
	  currSize=numweights;
	}

      from=&c->gradients[0][0];

      MPI_Reduce(from,to,numweights,
		 MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
      if (myid==0)
	{
	  memcpy(from,to,sizeof(float) * numweights);
	}
    }
#endif  
}


