#include <actpol/dirfile.h>
#include <actpol/getdata.h>
void ** read_channels_omp(int nchannel, int nthread, char * typechars, ACTpolDirfile *dirfile, char ** channelnames, int * nsamples);
void read_channels_into_omp(int nchannel, int nthread, char typechar, int nbyte, ACTpolDirfile *dirfile, char ** channelnames, void ** data);
