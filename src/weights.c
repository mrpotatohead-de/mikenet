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

#include "const.h"
#include "net.h"
#include "tools.h"
#include "weights.h"
#include "random.h"

void save_full_connection(Connections *c,FILE *f);
void load_full_connections(Connections *c,FILE *f);

int compress_weight_file=1;

int num_weights_in_net(Net *net)
{
  int t=0;
  int i;
  for(i=0;i<net->numConnections;i++)
    t += net->connections[i]->to->numUnits *
      net->connections[i]->from->numUnits;
  return t;
}

void init_dbdWeight(Connections *c)
{
  int i,j,nto,nfrom;
  nto=c->to->numUnits;
  nfrom=c->from->numUnits;

  c->dbdWeight=make_real_array(nto,nfrom);
  for(i=0;i<nto;i++)
    for(j=0;j<nfrom;j++)
      c->dbdWeight[i][j]=1.0;
}

void init_prevDeltas(Connections *c)
{
  int i,j,nto,nfrom;
  nto=c->to->numUnits;
  nfrom=c->from->numUnits;

  c->prevDeltas=make_real_array(nto,nfrom);
  for(i=0;i<nto;i++)
    for(j=0;j<nfrom;j++)
      c->prevDeltas[i][j]=0.0;
}

void freeze_weights(Connections *c)
{
  Group *to,*from;
  int i,j;

  to=c->to;
  from=c->from;
  for(i=0;i<to->numUnits;i++)
    for(j=0;j<from->numUnits;j++)
      {
	c->frozen[i][j]=1;
      }
}  

void unfreeze_weights(Connections *c)
{
  Group *to,*from;
  int i,j;

  to=c->to;
  from=c->from;
  for(i=0;i<to->numUnits;i++)
    for(j=0;j<from->numUnits;j++)
      {
	c->frozen[i][j]=0;
      }
}  

int load_binary_weights(Net *net,char *fn)
{
  FILE *f;
  char newfn[500];
  long size;
  char cmd[500];
  Connections *c;
  Real *buf;
  int is_tmpfile,i,j,offset,k;

  f=mikenet_open_for_reading(fn,newfn,&is_tmpfile);
  fseek(f,0,SEEK_END);
  size=ftell(f);
  fseek(f,0,SEEK_SET);

  buf=(Real *)mh_malloc(size);
  fread(buf,1,size,f);
  fclose(f);
  if (is_tmpfile)
    {
      sprintf(cmd,"rm -f %s",newfn);
      system(cmd);
    }

  offset=0;
  for(k=0;k<net->numConnections;k++)
    {
      c=net->connections[k];
      for(i=0;i<c->to->numUnits;i++)
	{
	  for(j=0;j<c->from->numUnits;j++)
	    {
	      c->weights[i][j]=buf[offset++];
	    }
	}
    }
  free(buf);

  return 0;
}

int save_binary_weights(Net *net,char *fn)
{
  FILE *f;
  long size;
  char cmd[500];
  Connections *c;
  Real *buf;
  int i,j,offset,k;

  f=fopen(fn,"w");
  size=0;
  for(i=0;i<net->numConnections;i++)
    {
      c=net->connections[i];
      size += c->to->numUnits * c->from->numUnits * sizeof(Real);
    }

  buf=(Real *)mh_malloc(size);

  offset=0;
  for(k=0;k<net->numConnections;k++)
    {
      c=net->connections[k];
      for(i=0;i<c->to->numUnits;i++)
	{
	  for(j=0;j<c->from->numUnits;j++)
	    {
	      buf[offset++]=c->weights[i][j];
	    }
	}
    }
  fwrite(buf,1,size,f);
  free(buf);
  fclose(f);

#ifdef unix
  if (compress_weight_file)
    {
      sprintf(cmd,"%s %s",default_compressor,fn);
      system(cmd);
    }
#endif

  return 0;
}


void load_taos(FILE *f,Net *net)
{
  Group *gto;
  char line[400];
  char *to,*ptr;
  int i,rc;
  float val;

  to=strtok(NULL," \t\n");
  gto=find_group_by_name(to);
  if (gto==NULL)
    {
      Error1("Error loading state file: group %s not in network",to);
      return;
    }
  for(i=0;i<gto->numUnits;i++)
    {
      fgets(line,255,f);
      ptr=strtok(line," \t\n");
      rc=sscanf(ptr,"%f",&val);
      if (rc!=1)
	{
	  Error0("Error: read failure in load_state; error reading TAOS");
	  return;
	}
      gto->taos[i]=val;
    }
}

void load_delays(FILE *f,Net *net)
{
  Group *gto;
  char line[400];
  char *to,*ptr;
  int i,rc;
  int val;

  to=strtok(NULL," \t\n");
  gto=find_group_by_name(to);
  if (gto==NULL)
    {
      Error1("Error loading state file: group %s not in network",to);
      return;
    }
  for(i=0;i<gto->numUnits;i++)
    {
      fgets(line,255,f);
      ptr=strtok(line," \t\n");
      rc=sscanf(ptr,"%d",&val);
      if (rc!=1)
	{
	  Error0("Error: read failure in load_state; error reading DELAYS");
	  return;
	}
      gto->delays[i]=val;
    }
}

int 
load_state(Net *net,char *fn)
{
  FILE *f;
  int i;
  char *from,*to,*p;
  Connections *c;
  char orgline[4000];
  char line[255];
  Group *gfrom,*gto;
  char cmd[255],newfn[255];
  int is_tmpfile=0,full;
  long newseed;

  f=mikenet_open_for_reading(fn,newfn,&is_tmpfile);
  
  fgets(line,255,f);
  while(!feof(f))
    {
      strcpy(orgline,line);
      from=strtok(line," \t\n");
      /* if its random number seed, read it in and then
	 eat the next line */
      if (strcmp(from,"SimulatorSeed")==0)
	{
	  p=strtok(NULL," \t\n");
	  newseed=atol(p);
	  mikenet_set_seed(newseed);
	}
      else if (strcmp(from,"TAOS")==0)
	{
	  load_taos(f,net);
	}
      else if (strcmp(from,"DELAYS")==0)
	{
	  load_delays(f,net);
	}
      else
	{
	  p=strtok(NULL," \t\n");
	  if (strcmp(p,"->")==0)
	    full=1;
	  else full=0;
	  to=strtok(NULL," \t\n");
	  if (from==NULL || to==NULL)
	    {
	      Error2("error loading weight file, from group=%20s, to group=%20s",
		     from,to);
	      return 0;
	    }
	  gfrom=find_group_by_name(from);
	  gto=find_group_by_name(to);
	  if (gfrom==NULL)
	    {
	      Error2("error loading weight file: group %s not in network, line %s",from,orgline);
	      return 0;
	    }
	  if (gto==NULL)
	    {
	      Error2("error loading weight file: group %s not in network, line %s",gto,orgline);
	      return 0;
	    }
	  c=NULL;
	  /* find connection */
	  for(i=0;i<net->numConnections;i++)
	    {
	      if (net->connections[i]->from==gfrom &&
		  net->connections[i]->to==gto)
		{
		  c=net->connections[i];
		  break;
		}
	    }
	  if (c==NULL)
	    {
	      Error2("group %s and %s aren't connected",
		     gfrom->name,gto->name);
	      return 0;
	    }
	  load_full_connections(c,f);
	}
      fgets(line,255,f);
    }
  fclose(f);	      
  
  if (is_tmpfile)
    {
      sprintf(cmd,"rm -f %s",newfn);
      system(cmd);
    }
  return 0;

}

void save_full_connection(Connections *c,FILE *f)
{
  int i,j;
  fprintf(f,"%s -> %s\n",
	  c->from->name,c->to->name);
  for(i=0;i<c->to->numUnits;i++)
    {
      for(j=0;j<c->from->numUnits;j++)
	{
	  fprintf(f,"%f",c->weights[i][j]);
	  if (c->frozen[i][j])
	    fprintf(f," f ");
	  if (c->dbdWeight != NULL)
	    fprintf(f," dbd %f ",c->dbdWeight[i][j]);
	  if (c->prevDeltas != NULL)
	    fprintf(f," PD %f ",c->prevDeltas[i][j]);
	  fprintf(f,"\n");
	}
    }
}


void load_full_connections(Connections *c,FILE *f)
{
  Group *gto, *gfrom;
  int i,j;
  char *ptr;
  char line[1000];
  int rc;
  float val;
  
  gto=c->to;
  gfrom=c->from;
  /* now load in the weights */
  for(i=0;i<gto->numUnits;i++)
    {
      for(j=0;j<gfrom->numUnits;j++)
	{
	  fgets(line,255,f);
	  ptr=strtok(line," \t\n");
	  rc=sscanf(ptr,"%f",&val);
	  if (rc!=1)
	    {
	      Error0("Error: read failure in load_state");
	      return;
	    }
	  /* don't change frozen weight */
	  if (c->frozen[i][j]==0)
	    c->weights[i][j]=val;
	  ptr=strtok(NULL," \t\n");  /* eat through arguments */
	  while(ptr)
	    {
	      if (strcmp(ptr,"f")==0)
		c->frozen[i][j]=1;
	      else if (strcmp(ptr,"dbd")==0)
		{
		  if (c->dbdWeight==NULL)
		    init_dbdWeight(c);
		  /* eat argument */
		  ptr=strtok(NULL," \t\n");
		  if (ptr==NULL)
		    {
		      Error0("Error in load_state: bad dbd value\n");
		      return;
		    }
		  rc=sscanf(ptr,"%f",&c->dbdWeight[i][j]);
		  if (rc!=1)
		    Error0("error reading dbd weight\n");
		}
	      else if (strcmp(ptr,"PD")==0)
		{
		  if (c->prevDeltas==NULL)
		    init_prevDeltas(c);
		  /* eat argument */
		  ptr=strtok(NULL," \t\n");
		  if (ptr==NULL)
		    {
		      Error0("Error in load_state: bad prevDelta value\n");
		      return;
		    }
		  rc=sscanf(ptr,"%f",&c->prevDeltas[i][j]);
		  if (rc!=1)
		    Error0("error reading prevDelta value\n");
		}
	      ptr=strtok(NULL," \t\n");
	    }
	}
    }
}

int save_state(Net *net,char *fn)
{
  FILE *f;
  int i,x;
  Connections *c;
  char cmd[255];

  f=fopen(fn,"w");
  if (f==NULL)
    {
      Error1("Can't open file %s for weight writing",fn);
    }
  
  fprintf(f,"SimulatorSeed %ld\n",mikenet_get_seed());
  for(x=0;x<net->numGroups;x++)
    {
      fprintf(f,"TAOS %s\n",net->groups[x]->name);
      for(i=0;i<net->groups[x]->numUnits;i++)
	{
	  fprintf(f,"%f\n",net->groups[x]->taos[i]);
	}
      fprintf(f,"DELAYS %s\n",net->groups[x]->name);
      for(i=0;i<net->groups[x]->numUnits;i++)
	{
	  fprintf(f,"%d\n",net->groups[x]->delays[i]);
	}
    }

  for(x=0;x<net->numConnections;x++)
    {
      c=net->connections[x];
      save_full_connection(c,f);
    }
  fclose(f);

#ifdef unix
  if (compress_weight_file)
    {
      sprintf(cmd,"%s %s",default_compressor,fn);
      system(cmd);
    }
#endif

  return 1;
}

void store_weights(Connections *c)
{
  int i,j,nto,nfrom;
  nto=c->to->numUnits;
  nfrom=c->from->numUnits;
  if (c->backupWeights==NULL)
    {
      c->backupWeights=make_real_array(nto,nfrom);
    }
  for(i=0;i<nto;i++)
    for(j=0;j<nfrom;j++)
      c->backupWeights[i][j]=
	c->weights[i][j];
}

void restore_weights(Connections *c)
{
  int i,j,nto,nfrom;
  if (c->backupWeights==NULL)
    {
      Error0("Attempt to restore weights before storing them");
    }
  nto=c->to->numUnits;
  nfrom=c->from->numUnits;
  for(i=0;i<nto;i++)
    for(j=0;j<nfrom;j++)
      c->weights[i][j]=
	c->backupWeights[i][j];
}

void store_all_weights(Net *net)
{
  int i;
  for(i=0;i<net->numConnections;i++)
    store_weights(net->connections[i]);
}

void restore_all_weights(Net *net)
{
  int i;
  for(i=0;i<net->numConnections;i++)
    restore_weights(net->connections[i]);
}

void decay_weights(Connections *c,Real v)
{
  int i,j,nto,nfrom;
  nto=c->to->numUnits;
  nfrom=c->from->numUnits;
  for(i=0;i<nto;i++)
    for(j=0;j<nfrom;j++)
      {
	if (c->frozen[i][j]==0)
	  c->weights[i][j] -= 
	    c->weights[i][j] * v;
      }
}

void weigand_decay_weights(Connections *c,Real v)
{
  int i,j,nto,nfrom;
  Real w,b;
  nto=c->to->numUnits;
  nfrom=c->from->numUnits;
  for(i=0;i<nto;i++)
    for(j=0;j<nfrom;j++)
      {
	w=c->weights[i][j];
	b= 1.0 + w * w;

	if (c->frozen[i][j]==0)
	  c->weights[i][j] -= 
	    v * (w / (b * b));
      }
}

void decay_all_weights(Net *net,Real v)
{
  int i;
  for(i=0;i<net->numConnections;i++)
    decay_weights(net->connections[i],v);
}


void weigand_decay_all_weights(Net *net,Real v)
{
  int i;
  for(i=0;i<net->numConnections;i++)
    weigand_decay_weights(net->connections[i],v);
}


