
extern Net *hearing,*bias_sem,*sem,*phono,*ps,*sp;
extern Group *phonology,*psh,*semantics,*bias,*sem_cleanup,*pho_cleanup;
extern Group *sph;
extern int connection_count;
void build_hearing_model(int samples,float tohid,float fromhid,float negpsh,float negsem);
extern float random_number_range;

