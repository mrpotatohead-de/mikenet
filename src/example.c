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
#include <math.h>
#include <stdlib.h>
#include "const.h"
#include "net.h"
#include "random.h"
#include "tools.h"
#include "example.h"

static int count=0;

Real default_exampleOnValue=1.0;
Real default_exampleOffValue=0.0;


void (*ex_clamps_premethod)(Example *ex,Group *g,int t) =
     default_ex_clamps_premethod;

void (*ex_clamps_postmethod)(Example *ex,Group *g,int t) =
     default_ex_clamps_postmethod;


void (*ex_targets_premethod)(Example *ex,Group *g,int t) =
     default_ex_targets_premethod;

void (*ex_targets_postmethod)(Example *ex,Group *g,int t) =
     default_ex_targets_postmethod;


Example* get_sequential_example(set)
ExampleSet *set;
{
  int v;
  if (set->currentExample==set->numExamples)
    set->currentExample=0;
  v=set->currentExample;
  set->currentExample++;
  set->histogram[v]++;
  return &set->examples[v];
}

int find_sparse_value(SparseExample *ex, int unit)
{
  int i;
  for(i=0;i<ex->numIndices;i++)
    {
      if (ex->indices[i]==unit)
	return 1;
    }
  return 0;
}

void init_full_example_data(ExampleData *edata,int numUnits)
{
  edata->type=EXPANDED_EXAMPLE;
  edata->values.expanded.value = 
    (Real *)mh_calloc(sizeof(Real),numUnits);
}

void init_sparse_example_data(ExampleData *edata,int numUnits)
{
  edata->type=SPARSE_EXAMPLE;
  edata->values.sparse.onvalue=default_exampleOnValue;
  edata->values.sparse.offvalue=default_exampleOffValue;
  edata->values.sparse.numIndices=0;
  edata->values.sparse.indices=NULL;
}


int has_value(ExampleData ***array,int group,int time,int unit)
{
  if (array==NULL)
    return 0;
  if (array[group]==NULL)
    return 0;
  if (array[group][time]==NULL)
    return 0;
  if (array[group][time]->type==EXPANDED_EXAMPLE)
    return VAL(array[group][time]->values.expanded.value[unit]);
  else  /* sparse networks always have a value */
    {
      return 1;
    }
}

Real get_value(ExampleData ***array,int group,int time,int unit)
{
  if (array==NULL)
    return -502;
  if (array[group]==NULL)
    return -502;
  if (array[group][time]==NULL)
    return -502;
  if (array[group][time]->type==EXPANDED_EXAMPLE)
    return (array[group][time]->values.expanded.value[unit]);
  else 
    {
      return find_sparse_value(&array[group][time]->values.sparse,unit);
    }
}



Example* get_random_example(set)
ExampleSet *set;
{
  int idx;
  Real dice,prob;
  while(1)
    {
      idx=(int)((mikenet_random() * (Real)set->numExamples));
      prob=set->examples[idx].prob;
      dice=mikenet_random();
      if (dice < prob) 
        {
	  set->histogram[idx]++;
	  return &set->examples[idx];
        }
    }
}


void ex_read_tag(Example *ex,FILE *f)
{
  char tag[1001];
  fgets(tag,1000,f);
  ex->name=(char *)mh_malloc(strlen(tag)+1);
  strcpy(ex->name,tag);
}

void ex_read_prob(Example *ex,FILE *f)
{
  float prob;
  char word[100];
  int rc;
  fscanf(f,"%s",word);
  rc=sscanf(word,"%f",&prob);
  if (rc !=1)
    Error1("Cannot read example prob from %s",word);
  ex->prob=prob;
}

void ex_read_expanded_exdata(Example *ex,ExpandedExample *edata,
			     Group *g,FILE *f)
{
  int i;
  int rc;
  char word[255];
  float v;

  for(i=0;i<g->numUnits;i++)
    {
      fscanf(f,"%s",word);
      if (strcmp(word,"NaN")==0 ||
	  strcmp(word,"d")==0)
	{
	  v=-505;
	}
      else
	{
	  rc=sscanf(word,"%f",&v);
	  if (rc != 1) Error1("load_examples: failure to read float: %s",
			      word);
	}
      edata->value[i]=v;
    }
}

void ex_read_sparse_exdata(Example *ex,SparseExample *edata,
			     Group *g,FILE *f)
{
  char word[255];
  float x;
  int rc;
  int *v;
  int n=0;

  v=(int *)mh_calloc(sizeof(int),g->numUnits);
  edata->indices=v;

  fscanf(f,"%s",word);
  while(strcmp(word,",")!=0)
    {
      if (strcmp(word,"ONVAL")==0)
	{
	  fscanf(f,"%s",word);
	  rc=sscanf(word,"%f",&x);
	  if (rc!=1) Error1("Error reading ONVAL, example %d",count);
	  edata->onvalue=x;
	}
      else if (strcmp(word,"OFFVAL")==0)
	{
	  fscanf(f,"%s",word);
	  rc=sscanf(word,"%f",&x);
	  if (rc!=1) Error1("Error reading OFFVAL, example %d",count);
	  edata->offvalue=x;
	}
      else if ((rc=sscanf(word,"%f",&x))==1)
	{
	  edata->indices[n++]=x;
	}
      else
	Error2("Unknown argument, example %d word %s",count,
	       word);
      fscanf(f,"%s",word);
    }
  edata->numIndices=n;
  if (n>0)
    edata->indices=(int *)mh_realloc(edata->indices,
				     sizeof(int) * n);
  else 
    {
      free(edata->indices);
      edata->indices=NULL;
    }
}


void ex_read_exdata(Example *ex,ExampleData *edata,
		    Group *g,FILE *f)
{
  char word[255];
  fscanf(f,"%s",word);
  if (strcmp(word,"FULL")==0 ||
      strncmp(word,"EXPANDED",6)==0)
    {
      init_full_example_data(edata,g->numUnits);
      ex_read_expanded_exdata(ex,&edata->values.expanded,g,f);
    }
  else if (strcmp(word,"SPARSE")==0)
    {
      init_sparse_example_data(edata,g->numUnits);
      ex_read_sparse_exdata(ex,&edata->values.sparse,g,f);
    }
  else 
    Error1("load_examples: illegal verb %s\n",word);
}

void ex_read_spec(Example *ex,FILE *f,SpecificationType type,
		  ExampleClampType clampType)
{
  char word[500];
  Group *g;
  int rc,start_t,end_t,t;
  ExampleData *edata;

  fscanf(f,"%s",word);
  g=find_group_by_name(word);
  if (g==NULL)
    Error1("load_examples: group %s not found\n",word);
  
  fscanf(f,"%s",word);
  if (strcmp(word,"ALL")==0)
    {
      start_t=0;
      end_t=ex->time-1;
    }
  else if ((rc=sscanf(word,"%d-%d",&start_t,&end_t))==2)
    { /* we're happy here -- do nothing  */
    }
  else if (((rc=sscanf(word,"%d-ALL",&start_t))==1) &&
	   strstr(word,"ALL")!=NULL)
    { 
      /* we need to check strstr of ALL because a single
	 number will match "%d-ALL". */
      end_t=ex->time-1;
    }
  else if ((rc=sscanf(word,"%d",&start_t))==1)
    {
      end_t=start_t;
    }
  else 
    Error1("load_examples: invalid time specification %s",word);

  edata=(ExampleData *)mh_malloc(sizeof(ExampleData));
  edata->clampType=clampType;

  if (start_t >= ex->time)
    Error1("load_examples: Example has invalid start time (%d)",start_t);
  
  if (end_t >= ex->time)
    Error1("load_examples: Example has invalid end time (%d)",end_t);
  
  
  if (type==CLAMP)
    {
      if (ex->inputs[g->index]==NULL)
	ex->inputs[g->index]=(ExampleData **)mh_calloc(sizeof(ExampleData *),
						       ex->time);
      for(t=start_t;t<=end_t;t++)
	ex->inputs[g->index][t]=edata;
    }
  else
    {
      if (ex->targets[g->index]==NULL)
	ex->targets[g->index]=(ExampleData **)mh_calloc(sizeof(ExampleData *),
							ex->time);
      for(t=start_t;t<=end_t;t++)
	ex->targets[g->index][t]=edata;
    }

  ex_read_exdata(ex,edata,g,f);
  
}


ExampleSet *
load_examples(char *fn,int maxt)
{
  FILE *f;
  char cmd[255];
  int maxExamples;
  ExampleSet * set;
  Example *examples;
  Example *ex;
  char verb[255];
  int is_tmpfile=0;
  char newfn[255];

  f=mikenet_open_for_reading(fn,newfn,&is_tmpfile);

  if (f==NULL) 
    return NULL;

  count=0;
  maxExamples=3000;
  examples=(Example *)mh_calloc(sizeof(Example),maxExamples);

  fscanf(f,"%s",verb);
  while(!feof(f))
    {
      if (count >= maxExamples)
        {
          maxExamples += 1000;
          examples=(Example *)mh_realloc(examples,
					 sizeof(Example)*maxExamples);
        }
      ex=&examples[count];
      ex->time=maxt;
      examples[count].userData=NULL;
      examples[count].prob=1.0;  /* default value */
      examples[count].name=NULL;
      examples[count].index=count;

      ex->targets=(ExampleData ***)mh_calloc(globalNumGroups,
					     sizeof(ExampleData **));
      ex->inputs=(ExampleData ***)mh_calloc(globalNumGroups,
					    sizeof(ExampleData **));

      /* munch until example is done (signaled by semicolon) */
      while(strcmp(verb,";")!=0)
	{
	  if (strcmp(verb,"TAG")==0)
	    ex_read_tag(ex,f);
	  else if (strcmp(verb,"PROB")==0)
	    ex_read_prob(ex,f);
	  else if ((strncmp(verb,"CLAMP",5)==0)  ||
		   (strncmp(verb,"INPUT",5)==0))
	    {
	      ex_read_spec(ex,f,CLAMP,CLAMP_HARD);
	    }
	  else if ((strncmp(verb,"SOFTCLAMP",9)==0)  ||
		   (strncmp(verb,"SOFT_CLAMP",10)==0))
	    {
	      ex_read_spec(ex,f,CLAMP,CLAMP_SOFT);
	    }
	  else if (strncmp(verb,"TARGET",6)==0)
	    {
	      ex_read_spec(ex,f,TARGET,CLAMP_HARD);
	    }
	  else
	    Error1("Unknown argument in example file: %s",verb);
	  fscanf(f,"%s",verb);
	}
      fscanf(f,"%s",verb);
      count++;
    }
  fclose(f);

#ifdef unix
  if (is_tmpfile)
    {
      sprintf(cmd,"rm -f %s",newfn);
      system(cmd);
    }
#endif  

  set=(ExampleSet *)mh_malloc(sizeof(ExampleSet));
  set->numExamples=count;
  set->examples=examples;
  set->name=(char *)mh_malloc(strlen(fn)+1);
  strcpy(set->name,fn);
  set->histogram=(int *)mh_calloc(sizeof(int),count);
  set->currentExample=0;

  return set; /* ok */
}


void
zap_example(Example *e)
{
  int i,j,k,time=e->time;
  for(i=0;i<globalNumGroups;i++)
    {
      for(j=0;j<time;j++)
	{
	  if (e->inputs[i][j]->type==EXPANDED_EXAMPLE)
	    {
	      for(k=0;k<groups[i]->numUnits;k++)
		{
		  e->targets[i][j]->values.expanded.value[k]= -501;
		  e->inputs[i][j]->values.expanded.value[k]= -501;
		}
	    }
	}
    }
}

void free_example(Example *ex)
{
  int i,j;

  for(i=0;i<globalNumGroups;i++)
    {
      for(j=0;j<ex->time;j++)
	{
	  free(ex->targets[i][j]->values.expanded.value);
	  free(ex->targets[i][j]);
	  free(ex->inputs[i][j]->values.expanded.value);
	  free(ex->inputs[i][j]);
	}
      free(ex->targets[i]);
      free(ex->inputs[i]);
    }
  free(ex->name);
  free(ex->targets);
  free(ex->inputs);
  free(ex);
}

Example *
create_example(int time)
{
  Example *e;
  ExampleData *edata;
  int i,j;
  e=(Example *)mh_calloc(1,sizeof(Example));
  e->time=time;
  e->name=(char *)mh_malloc(255);
  e->targets=(ExampleData ***)mh_calloc(globalNumGroups,
					sizeof(ExampleData **));
  e->inputs=(ExampleData ***)mh_calloc(globalNumGroups,
				       sizeof(ExampleData **));
  for(i=0;i<globalNumGroups;i++)
    {
      e->targets[i]=(ExampleData **)mh_calloc(time,
					      sizeof(ExampleData *));
      e->inputs[i]=(ExampleData **)mh_calloc(time,
					     sizeof(ExampleData *));
      for(j=0;j<time;j++)
	{
	  e->targets[i][j]=(ExampleData *)mh_calloc(1,
						    sizeof(ExampleData));
	  edata = e->targets[i][j];
	  init_full_example_data(edata,groups[i]->numUnits);
	  e->inputs[i][j]=(ExampleData *)mh_calloc(1,
						   sizeof(ExampleData));
	  edata = e->inputs[i][j];
	  init_full_example_data(edata,groups[i]->numUnits);
	}
    }
  zap_example(e);
  return e;
}


void default_ex_clamps_premethod(Example *ex,Group *g,int t)
{
  /* do nothing */
}

void default_ex_clamps_postmethod(Example *ex,Group *g,int t)
{
  /* do nothing */
}

void default_ex_targets_premethod(Example *ex,Group *g,int t)
{
  /* do nothing */
}

void default_ex_targets_postmethod(Example *ex,Group *g,int t)
{
  /* do nothing */
}

void apply_example_clamps(Example *ex,Group *g,int t)
{
  int i;
  SparseExample *sparse;
  ExpandedExample *expanded;

  (*ex_clamps_premethod)(ex,g,t);




  if (ex->inputs[g->index]==NULL)
    {
      for(i=0;i<g->numUnits;i++)
	g->exampleData[i]=-502;
    }
  else if (ex->inputs[g->index][t]==NULL)
    {
      for(i=0;i<g->numUnits;i++)
	g->exampleData[i]=-502;
    }
  else 
    {
      g->clampType=ex->inputs[g->index][t]->clampType;
      if (ex->inputs[g->index][t]->type==EXPANDED_EXAMPLE)
	{
	  expanded=&ex->inputs[g->index][t]->values.expanded;
	  
	  for(i=0;i<g->numUnits;i++)
	    g->exampleData[i]=expanded->value[i];
	}
      else if (ex->inputs[g->index][t]->type==SPARSE_EXAMPLE)
	{
	  sparse = &ex->inputs[g->index][t]->values.sparse;
	  for(i=0;i<g->numUnits;i++)
	    g->exampleData[i]= sparse->offvalue;
	  for(i=0;i<sparse->numIndices;i++)
	    g->exampleData[sparse->indices[i]]=sparse->onvalue;
	}
      else 
	Error0("Error in apply_example_clamps: unknown example type");
    }

  (*ex_clamps_postmethod)(ex,g,t);
}

void apply_example_targets(Example *ex,Group *g,int t)
{
  int i;
  SparseExample *sparse;
  ExpandedExample *expanded;

  (*ex_targets_premethod)(ex,g,t);

  if (ex->targets[g->index]==NULL)
    {
      for(i=0;i<g->numUnits;i++)
	g->exampleData[i]=-502;
    }
  else if (ex->targets[g->index][t]==NULL)
    {
      for(i=0;i<g->numUnits;i++)
	g->exampleData[i]=-502;
    }
  else if (ex->targets[g->index][t]->type==EXPANDED_EXAMPLE)
    {
      expanded=&ex->targets[g->index][t]->values.expanded;
      for(i=0;i<g->numUnits;i++)
	g->exampleData[i]=expanded->value[i];
    }
  else if (ex->targets[g->index][t]->type==SPARSE_EXAMPLE)
    {
      sparse = &ex->targets[g->index][t]->values.sparse;
      for(i=0;i<g->numUnits;i++)
	g->exampleData[i]= sparse->offvalue;
      for(i=0;i<sparse->numIndices;i++)
	g->exampleData[sparse->indices[i]]=sparse->onvalue;
    }
  else 
    Error0("Error in apply_example_targets: unknown example type");

  (*ex_targets_postmethod)(ex,g,t);

}

