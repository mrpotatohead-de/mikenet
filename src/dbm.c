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
#include <math.h>
#include "const.h"
#include "net.h"
#include "example.h"
#include "weights.h"
#include "dbm.h"
#include "bptt.h"
#include "random.h"
#include "tools.h"
#include "error.h"

#define CLAMP_TARGETS 0
#define DONT_CLAMP_TARGETS 1

void dbm_forward(Net *net,Example *ex,int doclamp);
void dbm_record_states(Net *net);
void dbm_update_connections(Net *net,Connections *c,int t);
void dbm_positive(Net *net,Example *ex)
{
  dbm_forward(net,ex,CLAMP_TARGETS);
  dbm_record_states(net);
}

void dbm_record_states(Net *net)
{
  int g,i;
  Group *gr;
  for(g=0;g<net->numGroups;g++)
    {
      gr=net->groups[g];
      if (gr->storedStates==NULL)
	gr->storedStates=mh_calloc(gr->numUnits,sizeof(Real));
      for(i=0;i<gr->numUnits;i++)
	{
	  gr->storedStates[i]=gr->outputs[net->runfor-1][i];
	}
    }
}

void dbm_negative(Net *net,Example *ex)
{
  dbm_forward(net,ex,DONT_CLAMP_TARGETS);
}

void dbm_update(Net *net)
{
  int i,j,k;
  Connections *c;
  Real *g,si_plus,si_neg,*sj_pluses,*sj_negs;

  for(i=0;i<net->numConnections;i++)
    {
      c = net->connections[i];
      for(k=0;k<c->to->numUnits;k++)
	{
	  g = c->gradients[k];
	  si_plus = c->to->storedStates[k];
	  si_neg = c->to->outputs[net->runfor-1][k];
	  sj_pluses = c->from->storedStates;
	  sj_negs = c->from->outputs[net->runfor-1];
	  for(j=0;j<c->from->numUnits;j++)
	    {
	      *g++ -= (si_plus *  *sj_pluses++ - 
		       si_neg * *sj_negs++);
	    }
	}
    }

}


void dbm_forward(Net *net,Example *ex,int doclamp)
{
  int i,t,nu,nt,ng;
  int nc,c,g;
  Group *gto;
  float prev;

  nt=net->time;
  ng=net->numGroups;
  nc=net->numConnections;
  for(t=0;t<nt;t++)
    {
      /* first, zero out the input field */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  nu=gto->numUnits;
	  for(i=0;i<nu;i++)
	    {
	      gto->inputs[t][i]=0;
	    }
	  gto->temperature = 
	    1.0/(gto->temperature + gto->temporg * 
		 pow(gto->tempmult,t));
	}
      
      for(c=0;c<nc;c++)
	{
	  if (!is_group_in_net(net->connections[c]->from,net))
	    Error1("dbm_update: Group %s is not in net",
		   net->connections[c]->from->name);

	  if (!is_group_in_net(net->connections[c]->to,net))
	    Error1("dbm_forward: Group %s is not in net",
		   net->connections[c]->to->name);
	  
	  dbm_update_connections(net,net->connections[c],t);
	}
      
      /* now, having computed all the inputs from the
	 previous time tick's outputs, update the current
	 time tick's outputs from its inputs (got that?). */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  apply_example_clamps(ex,gto,t);
	  nu=gto->numUnits;
	  for(i=0;i<nu;i++)
	    {
	      /* is it a bias */
	      if (gto->bias)
		gto->outputs[t][i]=gto->biasValue;
	      /* does it have a target */
	      else if (VAL(gto->exampleData[i]))
		{
		  gto->outputs[t][i]=gto->exampleData[i];
		  if (gto->clampNoise != 0)
		    {
		      gto->outputs[t][i] += 
			gto->clampNoise * get_gaussian();
		      gto->outputs[t][i]=
			clip_output(gto,gto->outputs[t][i]);
		    }
		}
	      else if (VAL(gto->exampleData[i]) &&
		       (doclamp == CLAMP_TARGETS))
		{
		  gto->outputs[t][i]=gto->exampleData[i];
		  if (gto->clampNoise != 0)
		    {
		      gto->outputs[t][i] += 
			gto->clampNoise * get_gaussian();
		      gto->outputs[t][i]=
			clip_output(gto,gto->outputs[t][i]);
		    }
		}
	      /* else compute normally */
	      else
		{
		  if (t==0) prev=0.0;
		  else prev = gto->outputs[t-1][i];
		  
		  gto->outputs[t][i] = 
		    (1.0-net->integrationConstant) * prev + 
		    net->integrationConstant * 
		    bptt_unit_activation(gto,i,t);
		  
		  if (gto->activationNoise != 0)
		    {
		      gto->outputs[t][i] 
			+= gto->activationNoise * get_gaussian();
		      gto->outputs[t][i]=
			clip_output(gto,gto->outputs[t][i]);
		    }
		}
	    }
	}
    }
  return;
}

void dbm_apply_deltas(Net *net)
{
  bptt_apply_deltas(net);
}

void dbm_update_connections(Net *net,Connections *c,int t)
{
  Real *invec,*outvec,*w,dot,*outs,*ins,out;
  int i,j,nto,nfrom;
  Group *gfrom,*gto;

  gfrom=c->from;
  gto=c->to;
  /* if we reset activation and t=0, don't do anything */
  if (gto->resetActivation==1 && t==0)
    return;
  /* already zeroed if t==0 */
  invec=gto->inputs[t];
  /* we won't be here with t==0 unless we're not resetting */
  if (t==0)  /* so don't reset */
    outvec=gfrom->outputs[net->time-1];
  else
    outvec=gfrom->outputs[t-1];
  nto=gto->numUnits;
  nfrom=gfrom->numUnits;

  if (c->weightNoiseType==NO_NOISE)
    {
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  dot=0.0;
	  outs=outvec;
	  for(j=0;j<nfrom;j++)
	    {
	      dot += *w++ * *outs++;
	    }
	  invec[i] +=dot;
	}
      /* now reverse, for bidirectional update */
      invec = gfrom->inputs[t];
      outvec = gto->outputs[t-1];
      for(i=0;i<nto;i++)
	{
	  w=c->weights[i];
	  dot=0.0;
	  ins=invec;
	  out = outvec[i];
	  for(j=0;j<nfrom;j++)
	    {
	      *ins++ += *w++ * out;
	    }
	}
    }
  else 
    Error0("Weight Noise not yet implemented for dbm");
}
