#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mikenet/simulator.h>

#include "model.h"

#define REP 500

int iter=1;

Real lr;

int count_connections(Net *net)
{
  int count=0,i,j,k;

  for(i=0;i<net->numConnections;i++)
    count += net->connections[i]->from->numUnits * 
      net->connections[i]->to->numUnits;
  return count;
}

int train(Net *net,ExampleSet *examples,int to)
{
  Example *ex;
  Real error=0;

  int count=1;
  
  for(;iter<=to;iter++)
    {
      ex=get_random_example(examples);
      crbp_forward(net,ex);
      crbp_compute_gradients(net,ex);
      error+=compute_error(net,ex);
      bptt_apply_deltas(net);
      
      /* is it time to write status? */
      if (count==REP)
	{
	  error = error/(float)count;
	  count=1;
	  /* print a message about average error so far */
	  printf("%d\t%f\tlr %f\n",iter,error,lr);
	  error=0.0;
	}
      else count++;
    }
  return 0;
}

int main(int argc,char *argv[])
{
  int i,sum;

  setbuf(stdout,NULL);
  /* set random number seed to process id */
  mikenet_set_seed(666);

  announce_version();

  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  mikenet_set_seed(atol(argv[i+1]));
	  i++;
	}
      else if (strncmp(argv[i],"-iter",5)==0)
	{
	  iter=atoi(argv[i+1]);
	  i++;
	}
    }
  
  
  /* build a network, with TIME number of time ticks */
  build_model();

  sum=count_connections(reading);
  printf("connections: %d\n",sum);

  lr=default_epsilon;

  /* load in our example set */
  reading_examples=load_examples("benchmark.pat",TIME);
  train(reading,reading_examples,10000);
  return 0;
}

