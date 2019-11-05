typedef struct {
	int year, month, day;
	double mjd;
	char pstat;
	double pmx, pmx_err, pmy, pmy_err;
	char utstat;
	double dUT, dUT_err, lod, lod_err;
	char nutstat;
	double dPSI, dPSI_err, dEps, dEps_err;
} IERSInfo;

IERSInfo * iers_lookup(double mjd);
IERSInfo * iers_get(int i);
int iers_n;
