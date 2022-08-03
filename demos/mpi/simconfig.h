#define SAMPLES 8
#define TICKS SAMPLES
#define SECONDS 4

#define TARGET_ON 6
#define TARGET_OFF  SAMPLES-1

#define AUTO_TARGET_ON 5
#define AUTO_TARGET_OFF  SAMPLES-1

#define AUTO_CLAMP_ON 0
#define AUTO_CLAMP_OFF 4


#define PHO_FEATURES 25
#define PHO_SLOTS 8
#define SEM_FEATURES 2446
#define MAX_FREQ 2645809
#define ORTHO_FEATURES 111+22

#define PROB_MAX 30000.0

#define MIN_PROB 0.05

/* #define FREQ(v) (log(v+1)/log(PROB_MAX)) */
#define FREQ(v) (sqrt(v)/sqrt(500.0))





