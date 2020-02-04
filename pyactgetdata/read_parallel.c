#include <stdlib.h>
#include <string.h>
#include <actgetdata/dirfile.h>
#include <omp.h>

void ** read_channels_omp(int nchannel, int nthread, char * typechars, ACTpolDirfile *dirfile, char ** channelnames, int * nsamples)
{
	void ** data = malloc(nchannel*sizeof(void*));
	#pragma omp parallel for num_threads(nthread)
	for(int i = 0; i < nchannel; i++)
		data[i] = ACTpolDirfile_read_channel(typechars[i], dirfile, channelnames[i], &nsamples[i]);
	return data;
}

void read_channels_into_omp(int nchannel, int nthread, char typechar, int nbyte, ACTpolDirfile *dirfile, char ** channelnames, void ** data)
{
	int nsamp;
	void * row;
	#pragma omp parallel for num_threads(nthread) private(row)
	for(int i = 0; i < nchannel; i++) {
		row = ACTpolDirfile_read_channel(typechar, dirfile, channelnames[i], &nsamp);
		memcpy(data[i], row, nbyte);
		free(row);
	}
}
