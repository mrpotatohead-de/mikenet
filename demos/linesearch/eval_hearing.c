/* to compile this, your MIKENET_DIR environment variable
   must be set to the appropriate directory.  putting this
   at the bottom of your .cshrc file will do the trick:

   setenv MIKENET_DIR ~mharm/mikenet/default/


   This demo program solves the xor problem.  

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef unix
#include <unistd.h>
#endif

#include <mikenet/simulator.h>
#include "model.h"
#include "hearing.h"
#include "reading.h"

#define REP 100
#define ITER 10000000


void get_name(char *tag, char *name)
{
  char *p;
  p=strstr(tag,"Word:");
  p+= 5;
  p=strtok(p," \t\n");
  strcpy(name,p);
}





int main(int argc,char *argv[])
{
  Real sse,e;
  int tai=0,on;
  FILE *f;
  char feature[2000][30];
  char name[255];
  int time=5;
  int wrongs=0,wrong=0,wrongtot;
  char weightFile[4000];
  int reset=1;
  int j;
  ExampleSet *examples;
  Example *ex;
  int i,count;
  Real error;
  Real epsilon,range;
  int runfor=TICKS;


  f=fopen("key","r");
  if (f==NULL)
    {
      fprintf(stderr,"can't open key file\n");
      exit(-1);
    }

  i=0;
  fscanf(f,"%s",feature[i++]);
  while(!feof(f))
    fscanf(f,"%s",feature[i++]);
  fclose(f);

  /* don't buffer output */
  setbuf(stdout,NULL);

  /* set random number seed to process id
     (only unix machines have getpid call) */
#ifdef unix
  mikenet_set_seed(getpid()); 
#endif

  /* a default learning rate */
  epsilon=0.01;
  
  /* default weight range */
  range=0.01;

  /* what are the command line arguments? */
  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  mikenet_set_seed(atol(argv[i+1]));
	  i++;
	}
      else if (strncmp(argv[i],"-epsilon",5)==0)
	{
	  epsilon=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-noreset",5)==0)
	{
	  reset=0;
	}
      else if (strcmp(argv[i],"-tai")==0)
	{
	  tai=1;
	}
      else if (strncmp(argv[i],"-runfor",5)==0)
	{
	  runfor=atoi(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-time",5)==0)
	{
	  time=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-range")==0)
	{
	  range=atof(argv[i+1]);
	  i++;
	}
      else if ((strncmp(argv[i],"-weight",5)==0) ||
	       (strncmp(argv[i],"-load",5)==0))
	{
	  strcpy(weightFile,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-errorRadius")==0)
	{
	  default_errorRadius=atof(argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown argument: %s\n",argv[i]);
	  exit(-1);
	}
    }
  default_resetActivation=reset;
  build_hearing_model();

  hearing->runfor=runfor;
  hearing->integrationConstant=((float)time/(float)hearing->runfor);

  sp->runfor=runfor;
  sp->integrationConstant=((float)time/(float)hearing->runfor);

  ps->runfor=runfor;
  ps->integrationConstant=((float)time/(float)hearing->runfor);

  examples=load_examples("ps.pat",TICKS);

  load_weights(hearing,weightFile);

  error=0.0;
  count=1;
  wrongs=0;
  wrongtot=0;
  /* loop for ITER number of times */
  for(i=0;i<examples->numExamples;i++)
    {
      /* get j'th example from exampleset */
      ex=&examples->examples[i];
      get_name(ex->name,name);
      printf("%s\t",name);
      /* do forward propagation */
      crbp_forward(ps,ex);
      wrong=0;
      sse=0.0;
      for(j=0;j<semantics->numUnits;j++)
	{
	  e = unit_error(semantics,ex,runfor-1,j);
	  sse += e * e;
	  on=0;
	  if (ex->targets[semantics->index][hearing->runfor-1][j] >= 0.5)
	    {  /* should be on */
	      on=1;
	      if (semantics->outputs[hearing->runfor-1][j] >= 0.5)
		printf("+%s ",feature[j]);
	      else 
		{
		  printf("[-%s(%.2f)] ",feature[j],
			 semantics->outputs[hearing->runfor-1][j]);
		  wrong++;
		}
	    }
	  else
	    {
	      /* target should be off */
	      if (semantics->outputs[hearing->runfor-1][j] < 0.5)
		{
		  /* print nothing; should be off and is */
		}
	      else
		{
		  /* should be off, and is on */
		  printf("[+%s(%.2f)] ",feature[j],
			 semantics->outputs[hearing->runfor-1][j]);
		  wrong++;
		}
	    }
	}
      if (wrong >0)
	wrongs++;
      wrongtot+= wrong;
      printf("\n");
    }
  printf("%d wrong, %d features\n",wrongs,wrongtot);
  return 0;
}
