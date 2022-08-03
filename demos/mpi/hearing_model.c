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
Connections *psh_to_pho,*pho_cleanup_to_pho,*sem_to_psh,*psh_to_sem;
Connections *pho_to_psh,*sem_cleanup_to_sem,*pho_to_pho;
int connection_count=0;

float random_number_range=RANDOM_NUMBER_RANGE;


void build_hearing_model(int samples)
{

  int i,x;
  float range;

  range = random_number_range;

  default_errorRamp=RAMP_ERROR;

  default_tai=1;

  default_activationType=LOGISTIC_ACTIVATION; 
  default_errorComputation=CROSS_ENTROPY_ERROR;
  default_errorRadius=0.01;

  default_scaling=SCALE_PROB;

  
  /* build a network, with samples number of time samples */
  hearing=create_net(samples);
  sp=create_net(samples);
  ps=create_net(samples);
  sem=create_net(samples);
  phono=create_net(samples);
  hearing->integrationConstant=(float)SECONDS/(float)samples;
  sp->integrationConstant=(float)SECONDS/(float)samples;
  ps->integrationConstant=(float)SECONDS/(float)samples;
  sem->integrationConstant=(float)SECONDS/(float)samples;
  phono->integrationConstant=(float)SECONDS/(float)samples;

  phonology=init_group("Phono",PHO_FEATURES * PHO_SLOTS,samples);
  psh=init_group("psh",300,samples);
  sph=init_group("sph",300,samples);
  semantics=init_group("Semantics",SEM_FEATURES,samples);
  sem_cleanup=init_group("SemCleanup",100,samples);
  pho_cleanup=init_group("PhoCleanup",50,samples);
  bias=init_bias(1.0,samples);

  /* now add our groups to the network object */
  bind_group_to_net(hearing,phonology);
  bind_group_to_net(hearing,psh);
  bind_group_to_net(hearing,sph);
  bind_group_to_net(hearing,semantics);
  bind_group_to_net(hearing,sem_cleanup);
  bind_group_to_net(hearing,pho_cleanup);
  bind_group_to_net(hearing,bias);

  bind_group_to_net(ps,phonology);
  bind_group_to_net(ps,psh);
  bind_group_to_net(ps,semantics);
  bind_group_to_net(ps,sem_cleanup);
  bind_group_to_net(ps,bias);

  bind_group_to_net(phono,bias);
  bind_group_to_net(phono,phonology);
  bind_group_to_net(phono,pho_cleanup);

  bind_group_to_net(sem,bias);
  bind_group_to_net(sem,semantics);
  bind_group_to_net(sem,sem_cleanup);

  bind_group_to_net(sp,phonology);
  bind_group_to_net(sp,sph);
  bind_group_to_net(sp,semantics);
  bind_group_to_net(sp,pho_cleanup);
  bind_group_to_net(sp,bias);


  x=0;
  /* now connect our groups */
  c[x]=connect_groups(phonology,psh);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  pho_to_psh=c[x];
  x++;

  c[x]=connect_groups(psh,semantics);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  pho_to_sem=c[x];
  psh_to_sem=c[x];
  x++;

  c[x]=connect_groups(semantics,sph);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  sem_to_psh=c[x];
  x++;

  c[x]=connect_groups(semantics,sem_cleanup);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  bind_connection_to_net(sem,c[x]);
  x++;

  c[x]=connect_groups(sem_cleanup,semantics);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  bind_connection_to_net(sem,c[x]);
  sem_cleanup_to_sem=c[x];
  x++;

  c[x]=connect_groups(bias,semantics);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  bind_connection_to_net(sem,c[x]);
  bias_semantics=c[x];
  x++;

  c[x]=connect_groups(bias,sem_cleanup);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  bind_connection_to_net(sem,c[x]);
  x++;

  c[x]=connect_groups(bias,psh);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(ps,c[x]);
  x++;

  c[x]=connect_groups(bias,sph);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  x++;

  c[x]=connect_groups(phonology,pho_cleanup);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  bind_connection_to_net(phono,c[x]);
  x++;

  c[x]=connect_groups(pho_cleanup,phonology);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  bind_connection_to_net(phono,c[x]);
  pho_cleanup_to_pho=c[x];
  x++;

  c[x]=connect_groups(sph,phonology);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  psh_to_pho=c[x];
  x++;

  c[x]=connect_groups(bias,phonology);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  bind_connection_to_net(phono,c[x]); 
  bias_phono=c[x];
  x++;

  c[x]=connect_groups(bias,pho_cleanup);
  bind_connection_to_net(hearing,c[x]);
  bind_connection_to_net(sp,c[x]);
  bind_connection_to_net(phono,c[x]);
  x++;


  connection_count=x;


  for(i=0;i<hearing->numConnections;i++)
    {
      c[i]=hearing->connections[i];
      randomize_connections(c[i],1.0);
    }

  for(i=0;i<semantics->numUnits;i++)
    bias_semantics->weights[i][0]=-3.0;

  for(i=0;i<phonology->numUnits;i++)
    bias_phono->weights[i][0]=-3.0;

  return;
}
