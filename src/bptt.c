/*
    mikenet - a simple, fast, portable neural network simulator.
    Copyright (C) 1995  Michael W. Harm

    This program is free software you can redistribute it and/or modify
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
#include "bptt.h"
#include "random.h"
#include "tools.h"
#include "error.h"

void bptt_update_connections(Net *net,Connections *c,int t);
void bptt_backprop_error(Net *net,Connections *c,int t);

/* given value of o, what input to group would produce that output */
Real invert_activation(Group *g,Real o)
{
  if (g->activationType==LOGISTIC_ACTIVATION)
    return sigmoid_inverse(o,g->temperature);
  else if (g->activationType==TANH_ACTIVATION)
    return tanh_inverse(o,g->temperature);
  else if (g->activationType==FAST_LOGISTIC_ACTIVATION)
    return sigmoid_inverse(o,g->temperature);
  else if (g->activationType==LINEAR_ACTIVATION)
    return o/g->temperature;
  else if (g->activationType==STEP_ACTIVATION)
    {
      if (o > 0.0)
	return 1.0;
      else return -1.0;
    }
  else Error0("Unknown activation type in invert_activation");
  return 0;
}

Real min_activation(Group *g)
{
  if (g->activationType==LOGISTIC_ACTIVATION || 
      g->activationType==FAST_LOGISTIC_ACTIVATION)
    return LOGISTIC_MIN;
  else if (g->activationType==TANH_ACTIVATION)
    return TANH_MIN;
  else Error0("Unknown activationType in min_activation");
  return 0.0;
}

Real max_activation(Group *g)
{
  if (g->activationType==LOGISTIC_ACTIVATION ||
      g->activationType==FAST_LOGISTIC_ACTIVATION)
    return LOGISTIC_MAX;
  else if (g->activationType==TANH_ACTIVATION)
    return TANH_MAX;
  else Error0("Unknown activationType in max_activation");
  return 0.0;
}



Real clip_output(Group *g,Real o)
{
  Real min=0.0,max=1.0;
  
  if (g->activationType==LOGISTIC_ACTIVATION || 
      g->activationType==FAST_LOGISTIC_ACTIVATION)
    {
      min=LOGISTIC_MIN;
      max=LOGISTIC_MAX;
    }
  else if (g->activationType==TANH_ACTIVATION)
    {
      min=TANH_MIN;
      max=TANH_MAX;
    }
  else if (g->activationType ==LINEAR_ACTIVATION)
    return o;
  else if (g->activationType == STEP_ACTIVATION)
    return o;
  else Error0("Unknown activation type");
  if (o<min)
    return min;
  else if (o>max)
    return max;
  else return o;
}

/* bound o for group between min+bound and max-bound. */
Real bounded_clip_output(Group *g,Real o,Real bound)
{
  Real min,max;
  min=min_activation(g)+bound;
  max=max_activation(g)-bound;
  return CLIP(o,min,max);
}

Real bptt_unit_derivative(Group *group,int unit,int tick)
{
  Real d;
  if (group->activationType==LOGISTIC_ACTIVATION)
    d= (sigmoid_derivative(group->outputs[tick][unit],
			   group->temperature));
  else if (group->activationType==TANH_ACTIVATION)
    d= (tanh_derivative(group->outputs[tick][unit],
			group->temperature));

  else
    {
      Choke0("Group %s does not have legal activationType",group->name);
      exit(-1);
      return 0;  /* just so stupid compilers don't complain */
    }
  d += group->primeOffset;
  return d;
}

Real bptt_unit_activation(Group *group,int unit,int tick)
{
  if (group->activationType==LOGISTIC_ACTIVATION) 
    return (CLIP(sigmoid_activation(group->inputs[tick][unit],
				    group->temperature),
		 LOGISTIC_MIN,
		 LOGISTIC_MAX));
  else if (group->activationType==TANH_ACTIVATION)
    return (CLIP(tanh_activation(group->inputs[tick][unit],
				 group->temperature),
		 TANH_MIN,
		 TANH_MAX));
  else if (group->activationType==FAST_LOGISTIC_ACTIVATION)
    return (CLIP(fast_sigmoid_activation(group->inputs[tick][unit],
				 group->temperature),
		 TANH_MIN,
		 TANH_MAX));
  else if (group->activationType==LINEAR_ACTIVATION)
    return (linear_activation(group->inputs[tick][unit],
			      group->temperature));
  else if (group->activationType==STEP_ACTIVATION)
    return (step_activation(group->inputs[tick][unit],
			    group->temperature));
  else
    {
      Choke0("Group %s does not have legal activationType",group->name);
      exit(-1);
      return 0;  /* just so stupid compilers don't complain */
    }
}

void bptt_update_connections(Net *net,Connections *c,int t)
{
  Group *gfrom,*gto;
  Real *invec,*outvec,*w,dot,*outs,weight;
  int i,j,nto,nfrom,d;

  /* if we reset activation and t=0, don't do anything */
  if (c->to->resetActivation==1 && t==0)
    return;
  gfrom=c->from;
  gto=c->to;

  invec=gto->inputs[t];
  /* we won't be here with t==0 unless we're not resetting */
  nto=gto->numUnits;
  nfrom=gfrom->numUnits;
  
  if (c->weightNoiseType==NO_NOISE)
    {
      for(i=0;i<nto;i++)
	{
	  if (t==0)  /* already zeroed if t=0 */
	    {
	      d=net->time-1;
	    }
	  else d= t - gto->delays[i];
	  if (d>=0) 
	    {
	      outvec=gfrom->outputs[d];
	      w=c->weights[i];
	      dot=0.0;
	      outs=outvec;
	      for(j=0;j<nfrom;j++)
		{
		  dot += *w++ * *outs++;
		}
	      invec[i] +=dot;
	    }
	}
    }
  else if (c->weightNoiseType==ADDITIVE_NOISE)
    {
      for(i=0;i<nto;i++)
	{
	  if (t==0)  /* already zeroed if t=0 */
	    {
	      d=net->time-1;
	    }
	  else d= t - gto->delays[i];
	  if (d>=0)
	    {
	      outvec=gfrom->outputs[d];
	      w=c->weights[i];
	      dot=0.0;
	      outs=outvec;
	      for(j=0;j<nfrom;j++)
		{
		  weight = *w++ + 
		    (get_gaussian() * c->weightNoise);
		  dot += weight * *outs++;
		}
	      invec[i] +=dot;
	    }
	}
    }
  else if (c->weightNoiseType==MULTIPLICATIVE_NOISE)
    {
      for(i=0;i<nto;i++)
	{
	  if (t==0)  /* already zeroed if t=0 */
	    {
	      d=net->time-1;
	    }
	  else d= t - gto->delays[i];
	  if (d>=0)
	    {
	      outvec=gfrom->outputs[d];
	      w=c->weights[i];
	      dot=0.0;
	      outs=outvec;
	      for(j=0;j<nfrom;j++)
		{
		  weight = *w++ * 
		    (1.0 + (get_gaussian() * c->weightNoise));
		  dot += weight * *outs++;
		}
	      invec[i] +=dot;
	    }
	}
    }
  else 
    Error0("Unknown weightNoiseType");
}


/* this populates each unit's "output" slot */
int bptt_forward(net,ex)
Net *net;
Example *ex;
{
  int i,t,nu,nt,ng;
  int nc,c,g,rc;
  Group *gto;
  Connections *conn;
  ExampleData *edata;

  nt=net->runfor;
  ng=net->numGroups;
  nc=net->numConnections;
  for(t=0;t<nt;t++)
    {
      rc=(*net->preForwardMethod)(net,ex,t);
      if (rc==0)
	break;

      /* first, zero out the input field */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  nu=gto->numUnits;
	  for(i=0;i<nu;i++)
	    {
	      gto->inputs[t][i]=0;
	    }
	}
      
      for(c=0;c<nc;c++)
	{
	  conn = net->connections[c];
	  if (!is_group_in_net(net->connections[c]->from,net))
	    Error1("bptt_forward: Group %s is not in net",
		   net->connections[c]->from->name);

	  if (!is_group_in_net(net->connections[c]->to,net))
	    Error1("bptt_forward: Group %s is not in net",
		   net->connections[c]->to->name);

	  rc=(*conn->preForwardPropagateMethod)(net->connections[c],ex,t);
	  if (rc) /* do we go on */
	    bptt_update_connections(net,conn,t);
	  (*conn->postForwardPropagateMethod)(net->connections[c],ex,t);
	}
      
      /* now, having computed all the inputs from the
	 previous time tick's outputs, update the current
	 time tick's outputs from its inputs (got that?). */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  rc=(*gto->preUpdateUnitMethod)(gto,ex,t);
	  if (rc) /* do we go on */
	    {
	      apply_example_clamps(ex,gto,t);
	      rc=(*gto->postApplyExampleClampsMethod)(gto,ex,t);
	      if (rc)
		{
		  nu=gto->numUnits;
		  for(i=0;i<nu;i++)
		    {
		      /* is it a bias */
		      if (gto->bias)
			gto->outputs[t][i]=gto->biasValue;
		      /* does it have a target */
		      else if (VAL(gto->exampleData[i]))
			{
			  edata = ex->inputs[gto->index][t];
			  if  (edata->clampType==CLAMP_SOFT)
			    Error0("Soft Clamping not supported for bptt");
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
			  if (gto->inputNoise > 0.0)
			    gto->inputs[t][i] += 
			      gto->inputNoise * get_gaussian();
			  
			  gto->outputs[t][i]=
			    bptt_unit_activation(gto,i,t);
			  
			  if (gto->activationNoise > 0.0)
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
	  (*gto->postUpdateUnitMethod)(gto,ex,t);
	}
      rc=(*net->postForwardMethod)(net,ex,t);
      if (rc==0)
	break;
    }
  return 1;
}



void bptt_backprop_error(Net *net,Connections *c,int t)
{
  Group *gto,*gfrom;
  Real *w;
  Real mysquig,*outs,*gradient,*squig;
  int nu,nfrom,j,i,d;

  gto=c->to;
  gfrom=c->from;
  nu=gto->numUnits;
  nfrom=gfrom->numUnits;
  /* backprop squiggles and tweek gradients */

  if (gfrom->numIncoming==0)
    {
      for(i=0;i<nu;i++)
        {
          mysquig=gto->dedx[t][i] * c->scaleGradients;
	  d = t - gto->delays[i];
	  /* d is the time on the 'from' unit which affected
	     our 'to' unit. default is 1 */
	  if (d>=0) 
	    {
	      outs = gfrom->outputs[d];
	      gradient = c->gradients[i];
	      for(j=0;j<nfrom;j++)
		{
		  *gradient++ += mysquig * *outs++;  
		}
	    }
        }
    }
  else
    {
      for(i=0;i<nu;i++)
        {
	  /* d is the time on the 'from' unit which affected
	     our 'to' unit. default is 1 */
	  d = t - gto->delays[i];
	  if (d>=0) 
	    {
	      outs = gfrom->outputs[d];
	      w=c->weights[i];
	      mysquig=gto->dedx[t][i] * c->scaleGradients;
	      squig=gfrom->dedx[d];
	      gradient = c->gradients[i];
	      for(j=0;j<nfrom;j++)
		{
		  *squig++ += 
		    *w++ * mysquig;
		  *gradient++ += mysquig * *outs++;  
		}
	    }
        }
    }
}



void bptt_compute_dedx(Net *net,Group *gto,Example *ex,int t,int i)
{
  Real local_err=0.0,deriv,d=0.0,out,target;
  
  /* note: here, dedx is a holder for dE/dy,
     not dE/dx */
  local_err=0.0;
  if (VAL(gto->exampleData[i]))
    {
      out=gto->outputs[t][i];
      target=gto->exampleData[i];
      if (gto->targetNoise > 0.0)
	{
	  target += gto->targetNoise * get_gaussian();
	}
      local_err=unit_error(gto,ex,i,out,target) * 2.0;
      d=unit_ce_distance(gto,ex,i,out,target);
    }

  if (gto->errorRamp==RAMP_ERROR)
    {
      local_err *= (float)t/((float)net->runfor-1);
      d *= (float)t/((float)net->runfor-1);
    }

  deriv=
    bptt_unit_derivative(gto,i,t);
  if (gto->errorComputation==SUM_SQUARED_ERROR)
    {
      gto->dedx[t][i] += local_err;
      gto->dedx[t][i] *= deriv;
    }
  else if (gto->errorComputation==CROSS_ENTROPY_ERROR)
    {
      gto->dedx[t][i] *=deriv;
      gto->dedx[t][i] +=d;
    }
  else 
    Error0("Unknown errorComputation type");
}
  
/* assumes each units "output" field is populated */
int bptt_compute_gradients(Net *net,Example *ex)
{
  int ng,nt,nc,t,i,g,nu,c,rc;
  Group *gto;
  Connections *connection;

  ng=net->numGroups;
  nt=net->runfor;
  nc=net->numConnections;

  /* zero squiggles */
  for(t=0;t<nt;t++)
    for(g=0;g<ng;g++)
      {

	gto=net->groups[g];
	nu=gto->numUnits;
	for(i=0;i<nu;i++)
	  {
	    gto->dedx[t][i]=0.0;
	  }
      }

  for(t=nt-1;t>=0;t--)
    {
      rc=(*net->preComputeGradientsMethod)(net,ex,t);
      if(rc==0)
	continue;

      /* First we compute the dE/dY (e(k) from Williams
	 and Peng) */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  rc=(*gto->preTargetSetMethod)(gto,ex,t);
	  if (rc)
	    apply_example_targets(ex,gto,t);
	  (*gto->postTargetSetMethod)(gto,ex,t);
	  rc=(*gto->preComputeDeDxMethod)(gto,ex,t);
	  if (rc)
	    {
	      nu=gto->numUnits;
	      for(i=0;i<nu;i++)
		{
		  bptt_compute_dedx(net,gto,ex,t,i);
		}
	    }
	  (*gto->postComputeDeDxMethod)(gto,ex,t);
	}
      /* all dedx values now computed. */
      
      
      /* now we propagate back other e(k)s if not
	 initial time tick.  this is the big time sink. */
      if (t > 0)
	{
	  for(c=0;c<nc;c++)
	    {
	      connection=net->connections[c];
	      bptt_backprop_error(net,connection,t);
	    }
	}
      (*net->postComputeGradientsMethod)(net,ex,t);
    }
  return 1;
}

