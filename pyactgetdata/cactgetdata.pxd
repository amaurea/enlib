from libc.stdint cimport int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
cdef extern from "cactgetdata.h":
	ctypedef struct FormatType:
		pass
	ctypedef struct ACTpolDirfile:
		FormatType * format
	
	ACTpolDirfile * ACTpolDirfile_open(char *filename)
	void ACTpolDirfile_close(ACTpolDirfile *dirfile)
	
	bint ACTpolDirfile_has_channel(ACTpolDirfile *dirfile, char *channel)
	
	void * ACTpolDirfile_read_channel(char typechar, ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	void ** read_channels_omp(int nchannel, int nthread, char * typechars, ACTpolDirfile *dirfile, char ** channelnames, int * nsamples)
	void read_channels_into_omp(int nchannel, int nthread, char typechar, int nbyte, ACTpolDirfile *dirfile, char ** channelnames, void ** data)

	int GetNEntry(FormatType * F)
	int GetEntryInfo(FormatType * F, int ind, char ** category, char ** name, char * field_code)
