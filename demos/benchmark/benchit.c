#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mikenet/simulator.h>


int main(int argc,char *argv[])
{
  int i,sum;
  float cps;
  int iters=1000;
  char line[255];

  setbuf(stdout,NULL);

  /* default_activationType=FAST_LOGISTIC_ACTIVATION; */

  if (argc>1)
    {
      iters=atoi(argv[1]);
    }

  printf("running for %d iters..\n",iters);
  cps=benchmark(iters);

  printf("%.2f million cps\n",(float)(cps/1000000.0));
  
}

