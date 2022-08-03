
extern Net *hearing,*sp,*ps,*sem,*phono;
extern Group *phonology,*psh,*semantics,*sem_cleanup,*pho_cleanup,*sph,*bias;
extern Connections * c[100],*pho_to_sem,*psh_to_pho,*pho_cleanup_to_pho;
extern Connections *bias_phono,*sem_to_psh,*psh_to_sem,*bias_semantics;
extern int connection_count;
void build_hearing_model(int samples);
extern float random_number_range;

