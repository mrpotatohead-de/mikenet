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
#include "error.h"
#include "tools.h"

void set_gradient_scale(Net *net,Real scale)
{
  int i;
  for(i=0;i<net->numConnections;i++)
    net->connections[i]->scaleGradients=scale;
}

void set_error_scale(Net *net,Real scale)
{
  int i;
  for(i=0;i<net->numGroups;i++)
    net->groups[i]->errorScale=scale;
}

Real scale_error(Group *g,Example *ex,int unit,Real e)
{
  if (g->scaling==SCALE_PROB)
    e *= ex->prob;
  e *= g->errorScaling[unit];
  e *= g->errorScale;
  return e;
}


Real unit_error(Group *g,Example *ex,int unit,Real out,Real target)
{
  Real local_err=0;
  int gnum;
  gnum=g->index;

  local_err= out - target;

  if (fabs(local_err) < g->errorRadius)
    local_err=0;
  else if (local_err > g->errorRadius)
    local_err -= g->errorRadius;
  else if (local_err < g->errorRadius)
    local_err += g->errorRadius;

  local_err = scale_error(g,ex,unit,local_err);

  return local_err;
}

Real unit_ce_error_deriv(Group *g,Example *ex,int unit,Real out,Real target)
{
  Real local_error;
  int gnum;

  gnum=g->index;

  if (fabs(out-target) < g->errorRadius)
    return 0.0;

  if (g->activationType==TANH_ACTIVATION)
    {
      out = (out / 2.0)+0.5;  /* cast it to 0 and 1 */
      target = (target / 2.0)+0.5;  /* cast it to 0 and 1 */
    }

  out=CLIP(out,LOGISTIC_MIN,LOGISTIC_MAX);
  out=CLIP(out,g->errorRadius,(1.0-g->errorRadius));

  if (fabs(target)<0.001) /* close enought to zero */
    {
      local_error = 1.0/(1.0-out);
    }
  else if (fabs(target)>0.999) /* close enough to 1 */
    {
      local_error= -1.0 / out;
    }
  else
    {
      Error0("can't use cross entropy error without binary targets");
      return 0.0;
    }

  local_error = scale_error(g,ex,unit,local_error);
  return local_error;
}

Real ce_error(Group *g,Example *ex,int unit,Real out,Real target)
{
  Real local_error;
  int gnum;

  gnum=g->index;

  if (fabs(out-target) < g->errorRadius)
    return 0.0;

  if (g->activationType==TANH_ACTIVATION)
    {
      out = (out / 2.0)+0.5;  /* cast it to 0 and 1 */
      target = (target / 2.0)+0.5;  /* cast it to 0 and 1 */
    }

  out=CLIP(out,LOGISTIC_MIN,LOGISTIC_MAX);

  if (fabs(target)<0.001) /* close enought to zero */
    {
      /* can't be higher than 1-errorRadius */
      out = CLIP(out,LOGISTIC_MIN,(1.0-g->errorRadius)); 
      local_error = -log(1.0-out);
    }
  else if (fabs(target)>0.999) /* close enough to 1 */
    {
      /* can't be lower than errorRadius */
      out = CLIP(out,g->errorRadius,LOGISTIC_MAX); 
      local_error= -log(out);
    }
  else
    {
      local_error = (log((target)/(out))*(target) +
		     log((1.0 - (target)) /
			 (1.0 - (out)))*(1.0 - (target)));
    }

  local_error = scale_error(g,ex,unit,local_error);
  return local_error;
}

Real unit_ce_distance(Group *g,Example *ex,int unit,Real out,Real target)
{
  Real d;
  int gnum;

  gnum=g->index;

  if (fabs(out-target) < g->errorRadius)
    return 0.0;

  if (g->activationType==TANH_ACTIVATION)
    {
      out = (out / 2.0)+0.5;  /* cast it to 0 and 1 */
      target = (target / 2.0)+0.5;  /* cast it to 0 and 1 */
    }

  d=out-target;

  d = scale_error(g,ex,unit,d);

  return d;
}

Real compute_error(Net *net,Example *ex)
{
  int u,t,g;
  Group *gto;
  float ramp;
  Real sserr=0.0,out,target,g_sserr;
  Real local_err;

  for(g=0;g<net->numGroups;g++)
    {
      gto=net->groups[g];
      g_sserr=0.0;
      for(t=0;t<net->runfor;t++)
	{
	  apply_example_targets(ex,gto,t);
	  for(u=0;u<gto->numUnits;u++)
	    {
	      if (gto->errorRamp==RAMP_ERROR)
		ramp=(float)t/net->runfor;
	      else ramp=1.0;
	      if (VAL(gto->exampleData[u]))
		{
		  out = gto->outputs[t][u];
		  target = gto->exampleData[u];
		  if (gto->errorComputation==SUM_SQUARED_ERROR)
		    {
		      local_err= unit_error(gto,ex,u,out,target);
		      sserr += 
			ramp * (local_err * local_err);
		    }
		  else if (gto->errorComputation==CROSS_ENTROPY_ERROR)
		    {
		      local_err= ce_error(gto,ex,u,out,target);
		      sserr += 
			ramp * fabs(local_err);
		    }
		  else
		    Error0("Unknown errorCompuation\n");
		}
	    }
	}
    }
  return sserr;
}

Real continuous_compute_error(Net *net,Example *ex)
{
  Error0("The function continuous_compute_error is obsolete. -mwh");
  return 0.0;
}

