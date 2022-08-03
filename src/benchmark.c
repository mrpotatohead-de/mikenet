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
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#include <time.h>


#include "const.h"
#include "net.h"
#include "example.h"
#include "tools.h"
#include "random.h"
#include "benchmark.h"
#include "bptt.h"
#include "crbp.h"


static void bench_randomize_example(Example *ex,Group *input,
				    Group *output,int maxtime)
{
  int i,t,flip;
  float v;

  for(i=0;i<input->numUnits;i++)
    {
      v=mikenet_random();
      if (v < 0.2)
	flip=1;
      else flip=0;
      for(t=0;t<maxtime;t++)
	{
	  ex->inputs[input->index][t]->values.expanded.value[i]=(float)flip;
	}
    }
  for(i=0;i<output->numUnits;i++)
    {
      v=mikenet_random();
      if (v < 0.2)
	flip=1;
      else flip=0;
      for(t=0;t<maxtime;t++)
	{
	  ex->targets[output->index][t]->values.expanded.value[i]=(float)flip;
	}
    }
}


float benchmark(int niters)
{
  Net *net;
  int maxtime=50;
  Group *input,*hidden,*output;
  Example *ex;
  Connections *c;
  int i;
  float numwts;
  int seed;
  float cps;
  time_t start,end;

  net=create_net(maxtime);
  net->integrationConstant=0.1;
  net->tai=1;
  
  input=init_group("input",500,maxtime);
  output=init_group("output",500,maxtime);
  hidden=init_group("hidden",200,maxtime);

  input->errorComputation=CROSS_ENTROPY_ERROR;
  hidden->errorComputation=CROSS_ENTROPY_ERROR;
  output->errorComputation=CROSS_ENTROPY_ERROR;

  bind_group_to_net(net,input);
  bind_group_to_net(net,output);
  bind_group_to_net(net,hidden);

  c=connect_groups(input,hidden);
  bind_connection_to_net(net,c);

  c=connect_groups(hidden,output);
  bind_connection_to_net(net,c);

  ex=create_example(maxtime);


  seed=mikenet_get_seed();
  mikenet_set_seed(500);

  start=time(NULL);

  bench_randomize_example(ex,input,output,maxtime);
  for(i=0;i<niters;i++)
    {
      crbp_forward(net,ex);
      crbp_compute_gradients(net,ex);
    }

  end=time(NULL);

  numwts = (((float)input->numUnits * (float)hidden->numUnits) +
	    ((float)output->numUnits * (float)hidden->numUnits)) * 
    (float)maxtime * (float)niters * 2.0;

  
  cps = (float)numwts/(float)(end-start);

  free_net(net);
  free_example(ex);
  mikenet_set_seed(seed); /* set seed back */
  return cps;
}


