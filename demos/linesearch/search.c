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


/* do a forward propagation of net using examples */
Real f(Net *net,ExampleSet *examples,int from,int to)
{
  int i,j;
  float meanH=0;
  int nH=0;
  float e=0.0;
  Example  *ex;
  for(i=from;i<to;i++)
    {
      ex=&examples->examples[i];
      bptt_forward(net,ex);
      for(j=0;j<psh->numUnits;j++)
	{
	  meanH += fabs(0.5-psh->outputs[2][j]);
	  nH++;
	}
      e+= compute_error(net,ex);
    }
  printf("-------------   mean H %f -----------\n",meanH/(float)nH);
  return e;
}

/* do a forward and backward propagation of net using examples */
Real g(Net *net,ExampleSet *examples,int from,int to)
{
  int i;
  float e=0.0;
  Example  *ex;
  for(i=from;i<to;i++)
    {
      ex=&examples->examples[i];
      bptt_forward(net,ex);
      bptt_compute_gradients(net,ex);
      e+= compute_error(net,ex);
    }
  return e;
}


/* what is the error if we move the net in the direction of lr */
/* assumes weights have been stored already, and gradient is computed  */
float sample(Net *net,ExampleSet *examples,int from,int to,float lr)
{
  /* move in the direction */
  test_step(net,lr);
  return f(net,examples,from,to);
}

