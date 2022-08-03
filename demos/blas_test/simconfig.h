#define SAMPLES 20
#define TICKS SAMPLES
#define SECONDS 4

#define TARGET_ON 3
#define TARGET_OFF  SAMPLES-1

#define AUTO_TARGET_ON 8
#define AUTO_TARGET_OFF  SAMPLES-1

#define AUTO_CLAMP_ON 0
#define AUTO_CLAMP_OFF 7

#define READ_PHO_TARGET_ON 2
#define READ_SEM_TARGET_ON 2


#define PHO_FEATURES 18
#define PHO_SLOTS 11
#define SEM_FEATURES 2050
#define MAX_FREQ 2645809
#define ORTHO_FEATURES 115+22

#define PROB_MAX 30000.0

#define MIN_PROB 0.01
/* #define MIN_PROB 0.01  */

#define FREQ(v) (((float)v)/5000.0) 


/* #define FREQ(v) ((sqrt(v)/sqrt(5000))) */

/* #define FREQ(v) (log(v+1)/log(373123)) */




