
#define AP_INT_MAX_W $AP_INT_MAX_W$
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2tensorstream.hpp"
#include <vector>

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

$DOCOMPUTE$

$DATAOUTSTREAM$

$SAVEASCNPY$

}

