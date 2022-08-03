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

Net *hearing,*sp,*ps,*sem,*phono;
Group *phonology,*psh,*semantics,*sem_cleanup,*pho_cleanup,*bias,*sph;
Connections *c[100],*bias_semantics,*bias_phono,*hid_pho,*pho_to_sem;
Connections *bias_sem_cleanup,*bias_psh,*bias_sph;
Connections *bias_pho_cleanup;
Connections *psh_to_pho,*pho_cleanup_to_pho,*sem_to_psh,*psh_to_sem;
Connections *pho_to_psh,*sem_cleanup_to_sem,*pho_to_pho;
int connection_count=0;

float random_number_range=RANDOM_NUMBER_RANGE;


void build_hearing_model(int samples,float tohid,float fromhid,
			 float negpsh,float negsem)
{

  int i,x,j;
  float range;

  range = random_number_range;

  default_tai=1;

  default_activationType=LOGISTIC_ACTIVATION; 
  default_errorComputation=CROSS_ENTROPY_ERROR;
  default_errorRadius=0.01;

  
  /* build a network, with samples number of time samples */
  hearing=create_net(samples);
  hearing->integrationConstant=(float)SECONDS/(float)samples;

  ps=create_net(samples);
  ps->integrationConstant=(float)SECONDS/(float)samples;

  sp=create_net(samples);
  sp->integrationConstant=(float)SECONDS/(float)samples;

  phonology=init_group("Phono",PHO_FEATURES * PHO_SLOTS,samples);
  semantics=init_group("Semantics",SEM_FEATURES,samples);

  /* now add our groups to the network object */
  bind_group_to_net(hearing,phonology);
  bind_group_to_net(hearing,semantics);
  

  bind_group_to_net(ps,phonology);
  bind_group_to_net(ps,semantics);

  x=0;
  /* now connect our groups */
  c[x]=connect_groups(phonology,semantics);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  x++;


  connection_count=x;

  randomize_connections(c[0],0.01);

  return;
}


