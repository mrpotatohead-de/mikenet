#define PHO_FEATURES 18
#define PHO_SLOTS 4
#define SEM_FEATURES 683
#define MAX_FREQ 2645809
#define ORTHO_FEATURES 81

#define PROB_MAX 300000

#define FREQ(v) ((float)(log((v+1)/5.0)/log(PROB_MAX/5.0)))
