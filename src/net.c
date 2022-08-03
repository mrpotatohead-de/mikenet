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
#include <string.h>
#include "const.h"
#include "net.h"
#include "tools.h"
#include "random.h"

/* please pardon the globals... */
int globalNumGroups=0;
Group **groups=NULL;

int default_tai=0;  
Real default_taoDecay=0.0;
int default_weightNoiseType=NO_NOISE;
Real default_weightNoise=0.0;
Real default_activationNoise=0.0;
Real default_inputNoise=0.0;
Real default_epsilon=1.0;
Real default_tao=1.0;
Real default_temperature=1.0;
Real default_temporg=0.0;
Real default_tempmult=0.9;  /* in dbm, temp = 
			       1/(temp + temporg * (tempmult ^ t)) */
int default_scaling=SCALE_NONE;
Real default_taoEpsilon=0.0;
Real default_errorRadius=0.0;
int default_activationType=LOGISTIC_ACTIVATION;
int default_errorComputation=SUM_SQUARED_ERROR;
int default_resetActivation=1;  
Real default_primeOffset=0.0;   
Real default_momentum=0.0;
Real default_dbdUp=0.1;
Real default_dbdDown=0.9;
int default_errorRamp=NO_RAMP_ERROR;
Real default_softClampThresh=0.001;
Real default_taoMaxMultiplier=-1;  /* -1 means use 1/net->integrationConstant*/
Real default_taoMinMultiplier = 0.001;

Net *create_net(time)
int time;
{
  Net *n;
  n=(Net *)mh_calloc(sizeof(Net),1);
  return init_net(n,time);
}

Net * init_net(net,time)
Net *net;
int time;
{
  Net *n;

  n=net;
  n->preForwardMethod=defaultNetMethod;
  n->postForwardMethod=defaultNetMethod;
  n->preComputeGradientsMethod=defaultNetMethod;
  n->postComputeGradientsMethod=defaultNetMethod;
  n->tai=default_tai;
  n->time=time;
  n->runfor=time;
  n->integrationConstant=1;
  n->groups=NULL;
  n->numGroups=0;
  n->numConnections=0;
  n->connections=NULL;
  return n;
}

void free_net(Net *net)
{
  int i;
  for(i=0;i<net->numGroups;i++)
    free_group(net->groups[i]);

  for(i=0;i<net->numConnections;i++)
    free_connections(net->connections[i]);

  free(net);
}

int is_group_in_net(Group *g,Net *net)
{
  int i;
  for(i=0;i<net->numGroups;i++)
    if (g==net->groups[i])
      return 1;
  return 0;
}

void unbind_connection_from_net(Net *net,Connections *c)
{
  int i,found=0;
  for(i=0;i<net->numConnections;i++)
    {
      if (found)
	{
	  net->connections[i-1]=net->connections[i];
	}
      else if (net->connections[i]==c)
	{
	  found=1;
	}
    }
  net->numConnections--;
}

void unbind_group_from_net(Net *net,Group *g)
{
  int i,found=0;
  for(i=0;i<net->numGroups;i++)
    {
      if (found)
	{
	  net->groups[i-1]=net->groups[i];
	}
      else if (net->groups[i]==g)
	{
	  found=1;
	}
    }
  net->numGroups--;
}

void bind_group_to_net(Net *net,Group *group)
{
  if (net->groups==NULL)
    net->groups=(Group **)mh_malloc(sizeof(Group *));
  else
    net->groups=(Group **)mh_realloc(net->groups,
				     (net->numGroups+1)*
				     sizeof(Group *));
  net->groups[net->numGroups]=group;
  net->numGroups++;
}

void bind_connection_to_net(Net *net,Connections *c)
{
  if (net->connections==NULL)
    net->connections=(Connections **)mh_malloc(sizeof(Connections *));
  else
    net->connections=(Connections **)mh_realloc(net->connections,
					  (net->numConnections+1)*
					  sizeof(Connections *));
  net->connections[net->numConnections]=c;
  net->numConnections++;
}

void free_group(Group *g)
{
  free(g->delays);
  free(g->dedtao);
  free(g->backz);
  free(g->z);
  free(g->name);
  free(g->exampleData);
  free(g->taos);

  free_real_array(g->inputs);
  free_real_array(g->dedx);
  free_real_array(g->dydtao);
  free_real_array(g->dxdtao);
  free_real_array(g->goalInputs);
  free_real_array(g->outputs);
  free_real_array(g->goalOutputs);

  free(g);
}

Group * init_group(char * name, int numUnits,int time)
{
  Group *g;
  int i;

  g= (Group *)mh_calloc(sizeof(Group),1);

  /* by default, not an elman net */
  g->elmanContext=0;
  g->elmanValues=NULL;
  g->elmanCopyFrom=NULL;
  g->elmanUpdate=0;




  g->preUpdateUnitMethod=defaultGroupMethod;
  g->postUpdateUnitMethod=defaultGroupMethod;
  
  g->preTargetSetMethod=defaultGroupMethod;
  g->postTargetSetMethod=defaultGroupMethod;
  
  g->preComputeDeDxMethod=defaultGroupMethod;
  g->postComputeDeDxMethod=defaultGroupMethod;

  g->postApplyExampleClampsMethod=defaultGroupMethod;
  
  g->userData=NULL;
  g->errorRamp=default_errorRamp;
  g->softClampThresh=default_softClampThresh;
  g->primeOffset=default_primeOffset;
  g->activationType=default_activationType;
  g->activationNoise=default_activationNoise;
  g->inputNoise=default_inputNoise;
  g->storedStates=NULL;
  g->numIncoming=0;
  g->numOutgoing=0;
  g->targetNoise=0;
  g->clampNoise=0;
  g->taoEpsilon=default_taoEpsilon;
  
  g->temperature=default_temperature;
  g->temporg=default_temporg;
  g->tempmult=default_tempmult;
  g->errorRadius=default_errorRadius;
  g->errorComputation=default_errorComputation;
  g->bias=0;
  g->index=globalNumGroups;
  /* don't scale error */
  g->scaling=default_scaling;

  g->taoMaxMultiplier=default_taoMaxMultiplier;
  g->taoMinMultiplier=default_taoMinMultiplier;

  /* stick on global group list */
  if (groups==NULL)
    groups=(Group **)mh_malloc(sizeof(Group *));
  else
    groups=(Group **)mh_realloc(groups,
				(globalNumGroups+1)*
				sizeof(Group *));

  g->delays=(int *)mh_calloc(sizeof(int),numUnits);
  for(i=0;i<numUnits;i++)
    g->delays[i]=1;

  g->errorScale=1.0; 
  g->whenDataLive=0;
  g->errorScaling=(Real *)mh_calloc(sizeof(Real),numUnits);
  for(i=0;i<numUnits;i++)
    g->errorScaling[i]=1.0;

  groups[globalNumGroups]=g;
  globalNumGroups++;

  g->biasValue=0.0;  /* not used normally */
  g->unitNames=NULL;
  g->resetActivation=default_resetActivation;
  g->dedx=make_real_array(time,numUnits);
  g->z=(Real *)mh_calloc(numUnits,sizeof(Real));
  g->dedtao=(Real *)mh_calloc(numUnits,sizeof(Real));
  g->backz=(Real *)mh_calloc(numUnits,sizeof(Real));
  g->dydtao=make_real_array(time,numUnits);
  g->dxdtao=make_real_array(time,numUnits);
  g->numUnits=numUnits;
  g->name=(char *)mh_malloc(strlen(name)+1);
  
  strcpy(g->name,name);
  g->inputs=make_real_array(time,numUnits);
  g->tempVector=(Real *)mh_calloc(numUnits,sizeof(Real));
  g->goalInputs=make_real_array(time,numUnits);
  g->outputs=make_real_array(time,numUnits);
  g->exampleData=(Real *)mh_calloc(numUnits,sizeof(Real));
  g->goalOutputs=make_real_array(time,numUnits);
  g->taos=(Real *)mh_calloc(g->numUnits,sizeof(Real));
  g->taoDecay=default_taoDecay;
  for(i=0;i<g->numUnits;i++)
    {
      g->taos[i]=default_tao;
    }
  return g;
}

Group * init_bias(float val,int time)
{
  Group *g;
  g=init_group("Bias",1,time);
  g->bias=1;
  g->biasValue=val;
  return g;
}


Group * find_group_by_name(char *name)
{
  int i;
  for(i=0;i<globalNumGroups;i++)
    {
      if (strcmp(groups[i]->name,name)==0)
	return groups[i];
    }
  return NULL;
}


Connections *
connect_groups(Group *gfrom,Group *gto)
{
  int i,j;
  Connections *c;

  c=(Connections *)mh_calloc(sizeof(Connections),1);
  c->locked=0;
  c->from=gfrom;
  c->errorType=ERR_NORMAL;
  c->to=gto;
  c->userData=NULL;
  gto->numIncoming++;
  gfrom->numOutgoing++;
  c->epsilon=default_epsilon;
  c->scaleGradients=1.0;
  c->weights=make_real_array(gto->numUnits,gfrom->numUnits);
  c->backupWeights=NULL;
  c->weightNoise=default_weightNoise;
  c->weightNoiseType=default_weightNoiseType;
  c->h=NULL;  /* used for conjugate gradients; null otherwise */
  c->g=NULL;
  c->momentum=default_momentum;
  c->gradients=make_real_array(gto->numUnits,gfrom->numUnits);
  c->prevDeltas=NULL;
  c->dbdWeight=NULL;
  c->dbdUp=default_dbdUp;
  c->dbdDown=default_dbdDown;

  c->preForwardPropagateMethod=defaultConnectionMethod;
  c->postForwardPropagateMethod=defaultConnectionMethod;

  c->numIncoming=(int *)mh_calloc(gto->numUnits,sizeof(int));

  c->incomingUnits=NULL;
  c->contribution=(Real *)mh_calloc(gto->numUnits,sizeof(Real));

  c->frozen=(unsigned char **)make_array(gto->numUnits,
					 gfrom->numUnits,
					 sizeof(unsigned char));
  
  for(i=0;i<gto->numUnits;i++)
    {
      c->numIncoming[i]=gfrom->numUnits;
      c->contribution[i]=0;
      for(j=0;j<gfrom->numUnits;j++)
	{
	  c->frozen[i][j]=0;
	  c->gradients[i][j]=0.0;
	}
    }

  return c;
}

void free_connections(Connections *c)
{
  free(c->numIncoming);
  free(c->contribution);
  free_array((void **)c->frozen);
  free_real_array(c->gradients);
  free_real_array(c->weights);
  free(c);
}

void
randomize_connections(Connections *c,Real weightRange)
{
  int i,j;
  Group *gto,*gfrom;
  gto=c->to;
  gfrom=c->from;

  for(i=0;i<gto->numUnits;i++)
    for(j=0;j<gfrom->numUnits;j++)
      {
	c->weights[i][j]=gen_random_weight(weightRange);
      }
}
      

int  name_units(g,fn)
Group *g;
char *fn;
{
  FILE *f;
  char line[4096];
  int i;

  f=fopen(fn,"r");
  if (f==NULL)
    {
      Error1("nameUnits: can't open file %s",fn);
      return -1;
    }
  if (g->unitNames==NULL)
    g->unitNames=(char **)mh_calloc(g->numUnits,sizeof(char *));
  for(i=0;i<g->numUnits;i++)
    {
      if (feof(f))
	{
	  Error0("nameUnits: premature end of file");
	  return -1;
	}
      fgets(line,4096,f);
      if (line[strlen(line)-1]== '\n')
	line[strlen(line)-1]=0; /* kill carriage return */
      g->unitNames[i]=(char *)mh_malloc(strlen(line)+1);
      strcpy(g->unitNames[i],line);
    }
  fclose(f);
  return 0;
}


void precompute_topology(Net *net,Group *input)
{
  int i;
  int changed;
  for(i=0;i<net->numGroups;i++)
    {
      net->groups[i]->whenDataLive=-1;
    }
  
  input->whenDataLive=0;
  
  changed=1;
  while(changed)
    {
      changed=0;
      for(i=0;i<net->numConnections;i++)
	{
	  if (net->connections[i]->from->bias==0)
	    {
	      if ((net->connections[i]->to->whenDataLive == -1 &&
		   net->connections[i]->from->whenDataLive != -1) 
		  ||
		  (net->connections[i]->to->whenDataLive > 
		   net->connections[i]->from->whenDataLive+1))
		{
		  changed=1;
		  net->connections[i]->to->whenDataLive=
		    net->connections[i]->from->whenDataLive+1;
		  
		}
	    }
	}
    }
  for(i=0;i<net->numGroups;i++)
    if (net->groups[i]->bias)
      net->groups[i]->whenDataLive=0;

}

void precompute_topology2(Net *net,Group *input,Group *input2)
{
  int i;
  int changed;
  for(i=0;i<net->numGroups;i++)
    {
      net->groups[i]->whenDataLive=-1;
    }
  
  input->whenDataLive=0;
  input2->whenDataLive=0;

  changed=1;
  while(changed)
    {
      changed=0;
      for(i=0;i<net->numConnections;i++)
	{
	  if (net->connections[i]->from->bias==0)
	    {
	      if ((net->connections[i]->to->whenDataLive == -1 &&
		   net->connections[i]->from->whenDataLive != -1) 
		  ||
		  (net->connections[i]->to->whenDataLive > 
		   net->connections[i]->from->whenDataLive+1))
		{
		  changed=1;
		  net->connections[i]->to->whenDataLive=
		    net->connections[i]->from->whenDataLive+1;
		  
		}
	    }
	}
    }
  for(i=0;i<net->numGroups;i++)
    if (net->groups[i]->bias)
      net->groups[i]->whenDataLive=0;

}

int defaultNetMethod(void *n,void *ex,int t)
{
  return 1;
}

int defaultGroupMethod(void *g, void *ex,int t)
{
  return 1;
}

int defaultConnectionMethod(void *c,void *ex,int t)
{
  return 1;
}
