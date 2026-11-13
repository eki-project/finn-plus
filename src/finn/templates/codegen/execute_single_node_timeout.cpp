
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

unsigned timeout = 0;
while(timeout < $TIMEOUT_VALUE$){

$DOCOMPUTE$

if($TIMEOUT_CONDITION$){
timeout++;
}

else{
$TIMEOUT_READ_STREAM$
timeout = 0;
}
}

$DATAOUTSTREAM$

$SAVEASCNPY$

}

