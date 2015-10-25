from libc.stdint cimport int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
cdef extern from "cactgetdata.h":
	ctypedef struct RawEntryType:
		char * field
		char * file
		char type
		int size
		int samples_per_frame
	ctypedef struct PolynomEntryType:
		char * field
	ctypedef struct LincomEntryType:
		char * field
	ctypedef struct LinterpEntryType:
		char * field
	ctypedef struct MultiplyEntryType:
		char * field
	ctypedef struct MplexEntryType:
		char * field
	ctypedef struct BitEntryType:
		char * field

	ctypedef struct FormatType:
		char * FileDirName
		int frame_offset
		RawEntryType first_field
		RawEntryType * rawEntries
		int n_raw
		PolynomEntryType *polynomEntries
		int n_polynom
		LincomEntryType *lincomEntries
		int n_lincom
		LinterpEntryType *linterpEntries
		int n_linterp
		MultiplyEntryType *multiplyEntries
		int n_multiply
		MplexEntryType *mplexEntries
		int n_mplex
		BitEntryType *bitEntries
		int n_bit
	ctypedef struct ACTpolDirfile:
		FormatType * format
	
	ACTpolDirfile * ACTpolDirfile_open(char *filename)
	void ACTpolDirfile_close(ACTpolDirfile *dirfile)
	
	bint ACTpolDirfile_has_channel(ACTpolDirfile *dirfile, char *channel)
	
	int16_t * ACTpolDirfile_read_int16_channel(ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	
	uint16_t * ACTpolDirfile_read_uint16_channel(ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	
	int32_t * ACTpolDirfile_read_int32_channel(ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	
	uint32_t * ACTpolDirfile_read_uint32_channel(ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	
	float * ACTpolDirfile_read_float_channel(ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	
	double * ACTpolDirfile_read_double_channel(ACTpolDirfile *dirfile, char *channelname, int *nsamples)
	
	uint32_t ACTpolDirfile_read_uint32_sample(ACTpolDirfile *dirfile, char *channelname, int index)
