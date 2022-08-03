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
#include "bptt.h"
#include "crbp.h"
#include "random.h"
#include "tools.h"
#include "error.h"
#include "dotprod.h"
#include "matrix.h"

Real crbp_unit_derivative(Group *group,int unit,int tick)
{
  Real d;
  if (group->activationType==LOGISTIC_ACTIVATION)
    d= (sigmoid_derivative(group->goalOutputs[tick][unit],
			       group->temperature));
  else if (group->activationType==TANH_ACTIVATION)
    d= (tanh_derivative(group->goalOutputs[tick][unit],
			    group->temperature));
  else if (group->activationType==FAST_LOGISTIC_ACTIVATION)
    d= (sigmoid_derivative(group->goalOutputs[tick][unit],
			    group->temperature));
  else
    {
      Choke0("Group %s does not have legal activationType",group->name);
      exit(-1);
      return 0;  /* just so stupid compilers don't complain */
    }
  return d + group->primeOffset;
}


void crbp_update_connections(Net *net,Connections *c,int t)
{
  Group *gfrom,*gto;
  Real *invec,*outvec,*w,dot,weight,*contrib;
  int i,j,nto,nfrom,d;

  /* if we reset activation and t=0, don't do anything */
  if (c->to->resetActivation==1 && t==0)
    return;

  contrib=c->contribution;

  gfrom=c->from;
  gto=c->to;

  /* already zeroed if t==0 */

  if (t < gto->whenDataLive)
    return;

  if (t-1 < gfrom->whenDataLive)
    return;

  /* if time averaged inputs, ramp up the goal inputs, then
     hit it with crbp_ramp_inputs later */
  if (net->tai)
    invec=gto->goalInputs[t];
  else
    invec=gto->inputs[t];  /* otherwise, inputs are inputs */

  outvec=gfrom->outputs[t-1];
  nto=gto->numUnits;
  nfrom=gfrom->numUnits;
  if (c->weightNoiseType==NO_NOISE)
    {
      if (t-1 >= gfrom->whenDataLive)
	mikenet_matrix_vec_mult(invec,nto,outvec,nfrom,c->weights);
    }
  else if (c->weightNoiseType==ADDITIVE_NOISE)
    {
      for(i=0;i<nto;i++)
	{
	  d=gto->delays[i];  /* delay line; normally 1 */
	  if (t-d >= gfrom->whenDataLive)
	    {
	      if (t==0) 
		outvec=gfrom->outputs[net->runfor-1];
	      else
		outvec=gfrom->outputs[t-d];
	      w=c->weights[i];
	      dot=0.0;
	      for(j=0;j<nfrom;j++)
		{
		  weight = *w++ + 
		    (get_gaussian() * c->weightNoise);
		  dot += weight * outvec[j];
		}
	      invec[i] +=dot;
	      contrib[i] = dot;
	    }
	}
    }
  else if (c->weightNoiseType==MULTIPLICATIVE_NOISE)
    {
      for(i=0;i<nto;i++)
	{
	  d=gto->delays[i];  /* delay line; normally 1 */
	  if (t-d >= gfrom->whenDataLive)
	    {
	      if (t==0) 
		outvec=gfrom->outputs[net->runfor-1];
	      else
		outvec=gfrom->outputs[t-d];
	      w=c->weights[i];
	      dot=0.0;
	      for(j=0;j<nfrom;j++)
		{
		  weight = *w++ * 
		    (1.0 + (get_gaussian() * c->weightNoise));
		  dot += weight * outvec[j];
		}
	      invec[i] +=dot;
	      contrib[i] = dot;
	    }
	}
    }
  else
    Error0("Unknown weightNoiseType");

}


void ramp_output(Net *net,Group *gto,int t, int i)
{
  Real out;  /* previous output */
  Real x,tao;

  if ((gto->resetActivation==1) &&
      (t==0))
    {
      out=0;
    }
  else if (t==0)
    out=gto->outputs[net->runfor-1][i];
  else 
    out=gto->outputs[t-1][i];

  if (gto->inputNoise > 0.0)
    gto->inputs[t][i] += gto->inputNoise * get_gaussian();
  
  x = bptt_unit_activation(gto,i,t);
  gto->goalOutputs[t][i]=x;
  /* instant input, time average output */
  tao = net->integrationConstant * gto->taos[i];

  /* the instantaneous derivative */
  gto->dydtao[t][i] = net->integrationConstant * (x - out);

  gto->outputs[t][i]= 
    ((1.0 - tao) * out)
    + (tao * x);
  if (gto->activationNoise > 0.0)
    {
      gto->outputs[t][i] 
	+= gto->activationNoise * get_gaussian();
    }
  gto->outputs[t][i] = clip_output(gto,gto->outputs[t][i]);
}

void ramp_input(Net *net,Group *gto,int t, int i)
{
  Real prev;  /* previous input */
  Real x,tao;



  if ((gto->resetActivation==1) &&
      (t==0))
    {
      prev=0;
    }
  else if (t==0)
    prev=gto->inputs[net->runfor-1][i];
  else 
    prev=gto->inputs[t-1][i];
  
  tao = net->integrationConstant * gto->taos[i];

  gto->inputs[t][i] = ((1.0 - tao) * prev) + 
    (tao * gto->goalInputs[t][i]);

  if (gto->inputNoise > 0.0)
    gto->inputs[t][i] += gto->inputNoise * get_gaussian();
  
  x = bptt_unit_activation(gto,i,t);

  gto->outputs[t][i]=gto->goalOutputs[t][i]=x;

  /* derivitive with respect to tao */
  gto->dxdtao[t][i] = (gto->inputs[t][i] - prev);

  if (gto->activationNoise > 0.0)
    {
      gto->outputs[t][i] 
	+= gto->activationNoise * get_gaussian();
    }
  gto->outputs[t][i] = clip_output(gto,gto->outputs[t][i]);
}

/* The idea here is that a bias projecting onto a group 
   is an intrinsic property of the units of that group.
   So, that value should be the starting point */
int set_initial_bias(Connections *c)
{
  int i;
  float x;
  Group *gfrom,*gto;
  gfrom=c->from;
  gto=c->to;

  for(i=0;i<gto->numUnits;i++)
    {
      x=c->weights[i][0] * gfrom->biasValue;
      gto->goalInputs[0][i] += x;
      gto->inputs[0][i] += x;
    }
  return 1;
}


/* this populates each unit's "output" slot */
int crbp_forward(net,ex)
Net *net;
Example *ex;
{
  int i,t,nu,nt,ng;
  int nc,c,g;
  Real x;
  Real xin;
  Connections *conn;
  Group *gto;
  int ramp,rc;

  nt=net->runfor;
  ng=net->numGroups;
  nc=net->numConnections;
  for(t=0;t<nt;t++)
    {
      net->t=t;
      rc=(*net->preForwardMethod)(net,ex,t);
      if(rc==0)
	break;

      /* first, zero out the input field */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  nu=gto->numUnits;
	  for(i=0;i<nu;i++)
	    {
	      gto->inputs[t][i]=0;
	      gto->goalInputs[t][i]=0;
	    }
	}

      /* now, populate the input (or goalInput) field 
	 with activity from 'from' units */
      for(c=0;c<nc;c++)
	{
	  conn = net->connections[c];
	  if (!is_group_in_net(net->connections[c]->from,net))
	    Error1("crbp_forward: Group %s is not in net",
		   net->connections[c]->from->name);

	  if (!is_group_in_net(net->connections[c]->to,net))
	    Error1("crbp_forward: Group %s is not in net",
		   net->connections[c]->to->name);

	  rc=(*conn->preForwardPropagateMethod)(conn,ex,t);
	  if (rc) /* do we go on */
	    {
	      if (conn->from->bias && t==0) /* is this from a bias? */
		{
		  set_initial_bias(conn);
		}
	      crbp_update_connections(net,conn,t);
	    }
	  (*conn->postForwardPropagateMethod)(conn,ex,t);
	}
      
      /* now, having computed all the inputs from the
	 previous time tick's outputs, update the current
	 time tick's outputs from its inputs (got that?). */
      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  rc=(*gto->preUpdateUnitMethod)(gto,ex,t); 
	  if (rc)
	    {
	      apply_example_clamps(ex,gto,t);
	      rc=(*gto->postApplyExampleClampsMethod)(gto,ex,t);
	      if (rc)
		{
		  nu=gto->numUnits;
		  for(i=0;i<nu;i++)
		    {
		      ramp=1; /* assume we need to ramp input or output */
		      if (t==0)
			ramp=0; /* assume we don't ramp on t0 */
		      
		      /* is it a bias */
		      if (gto->bias)
			{
			  gto->goalOutputs[t][i]=
			    gto->outputs[t][i] = gto->biasValue;
			  ramp = 0;  /* no need to ramp */
			}
		      
		      /* does it have a clamp? */
		      else if (VAL(gto->exampleData[i]))
			{
			  x=gto->exampleData[i];
			  if (gto->clampNoise > 0.0)
			    {
			      x += 
				gto->clampNoise * get_gaussian();
			      x=clip_output(gto,x);
			    }
			  if (gto->clampType==CLAMP_HARD)
			    {
			      gto->goalOutputs[t][i]=gto->outputs[t][i]=x;
			      x=clip_output(gto,x);
			      xin=invert_activation(gto,x);
			      gto->goalInputs[t][i] = gto->inputs[t][i] = xin;
			      ramp=0;  /* no need to ramp anything */
			    }
			  else if (gto->clampType == CLAMP_SOFT)
			    {
			      x=bounded_clip_output(gto,x,
						    gto->softClampThresh);
			      
			      xin = invert_activation(gto,x);
			      
			      if (net->tai) /* time average input? */ 
				gto->goalInputs[t][i] += xin;
			      else gto->inputs[t][i] +=  xin;
			      ramp=1; /* need to ramp input */
			    }
			  else
			    Error0("Example has invalid flag for clampType field\n");
			}
		      
		      if (ramp && t>=gto->whenDataLive) 
			{
			  if (net->tai)
			    ramp_input(net,gto,t,i);
			  else ramp_output(net,gto,t,i);
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



void crbp_backprop_error(Net *net,Connections *c,int t)
{
  Group *gto,*gfrom;
  Real *w;
  Real *backdot,dedx;
  int nu,nfrom,i;
  
  gto=c->to;
  gfrom=c->from;
  nu=gto->numUnits;
  nfrom=gfrom->numUnits;

  
  if (t < gto->whenDataLive)
    return;
  if (t-1 < gfrom->whenDataLive)
    return;


  /* backprop squiggles and tweek gradients */

  /* clever optimization: in this stage,  we do two
     things. we backprop errors to the 'to' units,
     and we modify the gradients.  but: if there
     are no incoming groups, there's no reason
     to backprop gradients. and if a connection
     set is locked, there's no reason to update the 
     gradients.  so form a 2x2 grid of the possibilities,
     and only do what you need. replicate code rather than
     have conditionals *in* the loop, for speed.  */

  if (c->locked)
    {
      if (gfrom->numIncoming==0) /* locked, no incoming. do nothing! */
	{
	  /* nop */
	}
      else /* locked, but with incoming.  backprop errors. */
	{
	  for(i=0;i<nu;i++)
	    {
	      w=c->weights[i];
	      dedx=gto->dedx[t][i];
	      backdot=gfrom->backz;
	      dotscalar(backdot,w,dedx,nfrom);
	    }
	}
    }
  else  /* not locked */
    {
      if (gfrom->numIncoming==0)  /* not locked, no incoming. */
	{
	  if (t-1 >= gfrom->whenDataLive)
	    {
	      for(i=0;i<nu;i++)
		gto->tempVector[i]=gto->dedx[t][i] * c->scaleGradients *
		  net->integrationConstant * gto->taos[i];
	      
	      mikenet_matrix_outer_product(c->gradients,
					   gto->tempVector,
					   nu,
					   gfrom->outputs[t-1],
					   nfrom);
	    }
#ifdef FOO
	  for(i=0;i<nu;i++)
	    {
	      mysquig=gto->dedx[t][i] * c->scaleGradients *
		net->integrationConstant * gto->taos[i];
	      d=gto->delays[i];
	      if (t-d >= gfrom->whenDataLive)
		{
		  outs = gfrom->outputs[t-d];
		  gradient = c->gradients[i];
		  dotscalar(gradient,outs,mysquig,nfrom);
		}
	    }
#endif
	}
      else /* not locked, incoming.  do the whole thing. */
	{
	  if (t-1 >= gfrom->whenDataLive)
	    {
	      for(i=0;i<nu;i++)
		gto->tempVector[i]=gto->dedx[t][i] * c->scaleGradients *
		  net->integrationConstant * gto->taos[i];
	      
	      mikenet_matrix_outer_product(c->gradients,
					   gto->tempVector,
					   nu,
					   gfrom->outputs[t-1],
					   nfrom);
	      
	      mikenet_matrix_vec_mult_t(gfrom->backz,
				      nfrom,
				      gto->dedx[t],
				      nu,
				      c->weights); 
	    }
				  
	  
#ifdef FOO
	  for(i=0;i<nu;i++) /* looping over 'to' units */
	    {
	      d=gto->delays[i];
	      if (t-d >= gfrom->whenDataLive)
		{
		  w=c->weights[i];
		  mysquig=gto->dedx[t][i] * c->scaleGradients *
		    net->integrationConstant * gto->taos[i];
		  dedx=gto->dedx[t][i];
		  backdot=gfrom->backz;
		  outs = gfrom->outputs[t-1];
		  gradient = c->gradients[i];
		  dotscalar(gradient,outs,mysquig,nfrom);
		  dotscalar(backdot,w,dedx,nfrom);
		}
	    }
#endif

	}
    }

}

void crbp_backprop_error_oja(Net *net,Connections *c,int t)
{
  Group *gto,*gfrom;
  Real *w;
  Real mysquig,*outs,*gradient,*backdot,dedx,mycontrib;
  int nu,nfrom,j,i,d;
  
  gto=c->to;
  gfrom=c->from;
  nu=gto->numUnits;
  nfrom=gfrom->numUnits;

  Error0("oja not really implemented");

  if (t < gto->whenDataLive)
    return;
  if (t-1 < gfrom->whenDataLive)
    return;

  for(i=0;i<nu;i++)
    {
      d=gto->delays[i];
      if (t-d >= gfrom->whenDataLive)
	{
	  w=c->weights[i];
	  mycontrib=
	    CLIP(sigmoid_activation(c->contribution[i],c->to->temperature),
		 LOGISTIC_MIN,LOGISTIC_MAX);

	  gto->dedx[t][i] = -1.0 * 
	    ((gto->outputs[t][i] - mycontrib) * c->epsilon);

	  mysquig=gto->dedx[t][i] * 
	    net->integrationConstant * gto->taos[i];
	  dedx=gto->dedx[t][i];
	  backdot=gfrom->backz;
	  outs = gfrom->outputs[t-1];
	  gradient = c->gradients[i];
	  for(j=0;j<nfrom;j++)
	    {
	      *backdot++ += 
		*w++ * dedx;
	      *gradient++ += mysquig * *outs++;  
	    }
	}
    }
}


/* dedy is integrated up */
void crbp_ramp_dedy(Net *net,Group *gto,Example *ex,int t,int i)
{
  Real local_err,deriv;
  Real out,target;

  local_err=0.0;
  if (VAL(gto->exampleData[i]))
    {
      out=gto->outputs[t][i];
      target = gto->exampleData[i];
      if (gto->targetNoise > 0.0)
	{
	  target += 
	    gto->targetNoise * get_gaussian();
	}

      if (gto->errorComputation==SUM_SQUARED_ERROR)
	local_err=unit_error(gto,ex,i,out,target) * 2.0;
      else if (gto->errorComputation==CROSS_ENTROPY_ERROR)
	{
	  local_err=unit_ce_error_deriv(gto,ex,i,out,target);
	}
      else
	Error0("Unknown error computation type");
    }


  if (gto->errorRamp==RAMP_ERROR)
    local_err *= (float)t/((float)net->runfor-1); 
  
  gto->backz[i] += local_err;

  /* z is dedy */
  gto->z[i] += net->integrationConstant * gto->taos[i] *
    (gto->backz[i] - gto->z[i]);

  deriv =
    crbp_unit_derivative(gto,i,t);

  /* instantaneous dedx */
  gto->dedx[t][i] = gto->z[i] * deriv;

  /* derivative based on tao */
  gto->dedtao[i] += gto->backz[i] * net->integrationConstant *
    gto->taos[i] * gto->dydtao[t][i];

  gto->backz[i]=0.0;
}

/* dedx is integrated up */
void crbp_ramp_dedx(Net *net,Group *gto,Example *ex,int t,int i)
{
  Real local_err,deriv,dedx=0.0,prev,out,target;

  local_err=0.0;
  if (VAL(gto->exampleData[i]))
    {
      out=gto->outputs[t][i];
      target=gto->exampleData[i];
      if (gto->targetNoise > 0.0)
	{
	  target += 
	    gto->targetNoise * get_gaussian();
	}

      if (gto->errorComputation==SUM_SQUARED_ERROR)
	local_err=unit_error(gto,ex,i,out,target) * 2.0;
      else if (gto->errorComputation==CROSS_ENTROPY_ERROR)
	local_err=unit_ce_distance(gto,ex,i,out,target);
      else
	Error0("Unknown error computation type");
    }

  if (gto->errorRamp==RAMP_ERROR)
    local_err *= (float)t / ((float)net->runfor-1);
  
  deriv =
    crbp_unit_derivative(gto,i,t);

  if (gto->errorComputation==CROSS_ENTROPY_ERROR)
    dedx = (gto->backz[i] * deriv) + local_err;
  else if (gto->errorComputation==SUM_SQUARED_ERROR)
    dedx = (gto->backz[i] + local_err) * deriv;

  if (t==net->runfor-1)
    prev=0;
  else prev=gto->dedx[t+1][i];

  gto->dedx[t][i] = prev + 
    (net->integrationConstant * gto->taos[i] * 
      (dedx - prev));

  /* dedx is instantaneous value. */
  gto->dedtao[i] += dedx * net->integrationConstant *
    gto->taos[i] * gto->dxdtao[t][i];


  gto->backz[i]=0.0;
}
  

int crbp_update_taos(Net *net)
{
  int i,nu,ng,j;
  Real max;
  Group *g;
  ng=net->numGroups;

  for(i=0;i<ng;i++)
    {
      g=net->groups[i];
      nu=g->numUnits;
      for(j=0;j<nu;j++)
	{
	  g->taos[j] -= g->dedtao[j] * g->taoEpsilon;

	  /* decay towards 1.0 */
	  g->taos[j] += (1.0-g->taos[j]) * g->taoDecay;

	  if (g->taoMaxMultiplier < 0)
	    max=1.0/net->integrationConstant;
	  else 
	    max=g->taoMaxMultiplier;
	  
	  if (g->taos[j] > max)
	    g->taos[j]=max;

	  if (g->taos[j] < g->taoMinMultiplier)
	    g->taos[j]=g->taoMinMultiplier;

	}
    }
  return 1;
}
  
/* assumes each units "output" field is populated */
int crbp_compute_gradients(Net *net,Example *ex)
{
  int ng,nt,nc,t,i,g,nu,c,rc;
  Group *gto;
  Connections *connection;

  ng=net->numGroups;
  nt=net->runfor;
  nc=net->numConnections;

  /* zero squiggles */
  for(g=0;g<ng;g++)
    {
      gto=net->groups[g];
      nu=gto->numUnits;
      for(i=0;i<nu;i++)
	{
	  gto->z[i]=0.0;
	  gto->backz[i]=0.0;
	  gto->dedtao[i]=0.0;
	  for(t=0;t<nt;t++)
	    gto->dedx[t][i]=0.0;
	}
    }
  
  for(t=nt-1;t>=0;t--)
    {
      rc=(*net->preComputeGradientsMethod)(net,ex,t);
      if(rc==0)
	continue;

      for(g=0;g<ng;g++)
	{
	  gto=net->groups[g];
	  nu=gto->numUnits;
	  rc=(*gto->preTargetSetMethod)(gto,ex,t);
	  if (rc)
	    apply_example_targets(ex,gto,t);
	  (*gto->postTargetSetMethod)(gto,ex,t);
	  rc=(*gto->preComputeDeDxMethod)(gto,ex,t);
	  if (rc)
	    {
	      if (net->tai)      
		{
		  for(i=0;i<nu;i++)
		    {
		      crbp_ramp_dedx(net,gto,ex,t,i); 
		    }
		}
	      else  /* time average outputs */
		{
		  for(i=0;i<nu;i++)
		    {
		      crbp_ramp_dedy(net,gto,ex,t,i); 
		    }
		}
	    }
	  (*gto->postComputeDeDxMethod)(gto,ex,t);
	}
      /* all z values now computed. */

      /* now we propagate back other e(k)s if not
	 initial time tick.  this is the big time sink. */
      if (t > 0)
	{
	  for(c=0;c<nc;c++)
	    {
	      connection=net->connections[c];
	      if (connection->errorType==ERR_NORMAL)
		crbp_backprop_error(net,connection,t);
	      else if (connection->errorType==ERR_OJA)
		{
		  if (net->tai)
		    crbp_backprop_error_oja(net,connection,t);
		  else Error0("OJA rule not implemented for TAO networks");
		}
	      else Error0("Unknown errorType");
	    }
	}
      (*net->postComputeGradientsMethod)(net,ex,t);
    }
  return 1;
}

/* just included for syntactic sugar */
int  crbp_apply_deltas(Net *net)
{
  return bptt_apply_deltas(net);
}

