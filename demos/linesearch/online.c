
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
#include "search.h"

#define REP 100
#define ITER 10000000







int main(int argc,char *argv[])
{
  int j;
  float epsilon,err;
  float hN=0;
  int nN=0;
  int seed,start=1,i;
  Example *ex;
  ExampleSet *hearing_examples;

  setbuf(stdout,NULL);
  
  /* a default learning rate */
  epsilon=0.1;
  
  /* what are the command line arguments? */
  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  seed=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-start")==0)
	{
	  start=atoi(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-epsilon",5)==0)
	{
	  epsilon=atof(argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown argument: %s\n",argv[i]);
	  exit(1);
	}
    }
  default_epsilon=epsilon;


  mikenet_set_seed(seed);

  build_hearing_model();

  /* load in our example set */
  hearing_examples=load_examples("ps.pat",TICKS); 

  err=0.0;
  /* loop for ITER number of times */
  for(i=start;i<=5000000;i++)
    {
      ex=get_random_example(hearing_examples);
      bptt_forward(ps,ex);
      
      for(j=0;j<psh->numUnits;j++)
	{
	  hN += fabs(0.5 - psh->outputs[2][j]);
	  nN++;
	}

      bptt_compute_gradients(ps,ex);
      err += compute_error(ps,ex);
      bptt_apply_deltas(ps);
      if (i % 500 ==0)
	{
	  printf("iter %d mean err %f |h|=%f\n",i,err/500.0,
		 hN/(float)nN);
	  hN=0.0;
	  nN=0;
	  err=0;
	}
    }
  return 0;
}
