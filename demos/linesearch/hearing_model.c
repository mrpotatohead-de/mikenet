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
#include "simconfig.h"

#ifdef unix
#include <unistd.h>
#endif

#include <mikenet/simulator.h>
#include "model.h"
#include "hearing.h"

Net *hearing,*sp,*ps;
Group *phonology,*psh,*semantics,*sem_cleanup,*pho_cleanup,*bias;
Connections *c[100],*bias_semantics,*bias_phono,*sem_hid,*hid_pho;
int connection_count=0;

void build_hearing_model()
{
  int i,x;
  float range;

  range = RANDOM_NUMBER_RANGE;

  default_activationType=LOGISTIC_ACTIVATION; 
  default_errorComputation=CROSS_ENTROPY_ERROR;
  default_errorRadius=0.01;

  /* build a network, with TICKS number of time ticks */
  ps=create_net(TICKS);
  phonology=init_group("Phono",PHO_FEATURES * PHO_SLOTS,TICKS);
  psh=init_group("hidden",100,TICKS);
  semantics=init_group("Semantics",SEM_FEATURES,TICKS);
  bias=init_bias(1.0,TICKS);

  /* now add our groups to the network object */
  bind_group_to_net(ps,phonology);
  bind_group_to_net(ps,psh);
  bind_group_to_net(ps,semantics);
  bind_group_to_net(ps,bias);

  x=0;
  /* now connect our groups */
  c[x]=connect_groups(phonology,psh);
  bind_connection_to_net(ps,c[x]);
  x++;

  c[x]=connect_groups(psh,semantics);
  bind_connection_to_net(ps,c[x]);
  x++;


  c[x]=connect_groups(bias,semantics);
  bind_connection_to_net(ps,c[x]);
  bias_semantics=c[x];
  x++;

  c[x]=connect_groups(bias,psh);
  bind_connection_to_net(ps,c[x]);
  x++;

  connection_count=x;

  for(i=0;i<x;i++)
    {
      randomize_connections(c[i],1.0);
    }

  for(i=0;i<semantics->numUnits;i++)
    bias_semantics->weights[i][0] = -5.0;


  return;
}
