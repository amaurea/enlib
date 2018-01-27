#include <actpol/dirfile.h>
#include <actpol/getdata.h>
void ** read_channels_omp(int nchannel, int nthread, char * typechars, ACTpolDirfile *dirfile, char ** channelnames, int * nsamples);
