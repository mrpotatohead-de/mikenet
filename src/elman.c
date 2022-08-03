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
#include "elman.h"


int elmanUpdateUnitMethod(void *g,void *e, int t)
{
  Group *group=(Group *)g;
  int i;

  if (t==0)
    {
      for(i=0;i<group->numUnits;i++)
	group->outputs[t][i] = group->elmanValues[i] = 0;
      return 1;
    }
  else if ((t % group->elmanUpdate) == 0)
    {
      for(i=0;i<group->numUnits;i++)
	{
	  group->elmanValues[i] =
	    group->elmanCopyFrom->outputs[t-1][i];
	}
    }

  for(i=0;i<group->numUnits;i++)
    {
      group->outputs[t][i]=group->elmanValues[i];
    }
  
  
  return 1;
}

void elmanize(Group * context,Group *hidden,int howmanyticks)
{

  if (context->numUnits != hidden->numUnits)
    Error0("Can't create Elman context group: different number of units\n");
  context->elmanContext=1;
  context->elmanCopyFrom=hidden;
  context->elmanValues=(Real *)mh_calloc(sizeof(Real),context->numUnits);
  context->elmanUpdate=howmanyticks;
  context->postUpdateUnitMethod=elmanUpdateUnitMethod;
}
