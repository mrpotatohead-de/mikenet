
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



int main(int argc,char *argv[])
{
  float err;
  Example *ex;
  int do_cg=0;
  float epsilon;
  int seed=555,first=1;
  float e,e0,lr,lrPrev;
  int start=1,i,j;
  char  loadFile[255];
  ExampleSet *hearing_examples;

  setbuf(stdout,NULL);

  loadFile[0]=0;

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
      else if (strncmp(argv[i],"-load",5)==0)
	{
	  strcpy(loadFile,argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown argument: %s\n",argv[i]);
	  exit(1);
	}
    }

  default_epsilon=0.1;


  mikenet_set_seed(seed);

  build_hearing_model();

  /* load in our example set */
  hearing_examples=load_examples("ps.pat",TICKS);

  err=0.0;
  for(i=0;i<10000;i++)
    {
      ex=get_random_example(hearing_examples);
      bptt_forward(ps,ex);
      err+=compute_error(ps,ex);
      if (i % 500==0)
	{
	  printf("%d %f\n",i,err/500.0);
	  err=0.0;
	}
    }
  save_weights(ps,"online.weights");
  do_cg=1;

  /* loop for ITER number of times */
  for(i=start;i<=1000;i++)
    {
      store_all_weights(ps);
      e=g(ps,hearing_examples,0,hearing_examples->numExamples);

      if(do_cg)
	{
	  if (first)
	    init_cg(ps);
	  else
	    cg(ps);
	  first=0;
	}

      e0=e;
      printf("%d e0: %f\n",i,e);
      lr = 0.2/hearing_examples->numExamples;
      lrPrev=lr/10;
      for(j=1;j<6;j++)
	{
	  e=sample(ps,hearing_examples,0,hearing_examples->numExamples,
		   lr);
	  printf("\t\t%d %f %f\n",j,e,lr);
	  if (e>e0)
	    {
	      test_step(ps,lrPrev);
	      break;
	    }
	  e0=e;
	  lrPrev=lr;
	  lr *= 1.5;
	}
      zero_gradients(ps);
    }
  return 0;
}
