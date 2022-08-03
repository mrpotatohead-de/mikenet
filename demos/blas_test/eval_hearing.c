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
#include "simconfig.h"
#include "model.h"
#include "hearing.h"
#include "reading.h"

#define REP 100
#define ITER 10000000

char feature[3000][100];

float compute_sem_error(Example *ex)
{
  float e=0.0,e1;
  int i,j;

  for(i=0;i<semantics->numUnits;i++)
    {
      e1 = fabs(semantics->outputs[SAMPLES-1][i] -
                get_value(ex->targets,semantics->index,SAMPLES-1,i));
      e += e1*e1;
    }
  return e;
}


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
  float clamp=0.01;
  float o;
  float v;
  Real sse,e;
  int tai=0,on;
  FILE *f;
  char name[255];
  float time=SECONDS;
  int wrongs=0,wrong=0,wrongtot;
  char weightFile[4000];
  int reset=1;
  int j;
  ExampleSet *examples;
  Example *ex;
  int i,count;
  Real error;
  Real epsilon,range;
  int runfor=SAMPLES,junk;

#ifdef BLAS
  default_useBlas=1;
  fprintf(stderr,"Using blas!\n");
#else
  default_useBlas=0;
  fprintf(stderr,"NOT using blas!\n");
#endif

  announce_version();



  strcpy(default_compressor,"/usr/bin/gzip");
  strcpy(default_decompressor,"/usr/bin/gunzip");

  /* don't buffer output */
  setbuf(stdout,NULL);

  /* set random number seed to process id
     (only unix machines have getpid call) */

  mikenet_set_seed(555); 

  /* a default learning rate */
  epsilon=0.001;
  
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
      else if (strncmp(argv[i],"-clamp",5)==0)
	{
	  clamp=atof(argv[i+1]);
	  i++;
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

  default_softClampThresh=clamp;
  default_errorRamp=RAMP_ERROR;
  
  build_hearing_model(runfor,0.1,0.1,-3.0,-3.0);

  hearing->integrationConstant=(float)time/(float)hearing->runfor;

  examples=load_examples("ps.pat",runfor);

  error=0.0;
  count=1;
  wrongs=0;
  wrongtot=0;

  precompute_topology(hearing,phonology);

  /* loop for ITER number of times */
  for(i=0;i<=150;i++)
    {
      /* get j'th example from exampleset */
      ex=&examples->examples[(i*25)%(examples->numExamples-1)];

      crbp_forward(hearing,ex);
      sse = compute_sem_error(ex);

      /*  crbp_compute_gradients(hearing,ex);
	  crbp_apply_deltas(hearing);   */

      if (i % 10 ==0)
	printf("%d sse %f\n",i,sse);
    }
  return 0;
}
