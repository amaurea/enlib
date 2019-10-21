double displace_map_blocks_avx_omp(
		// The input and output maps. Each is [ny,nx] in CAR
		float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra,
		// The x,y,z position of the earth relative to the sun in celestial coordinates
		double * earth_pos,
		// The sun-distance of the object (which determines the parallax displacement), and the orbital offset
		double r, double dy, double dx);

void solve_plain(float * frhs, float * kmap, float * osigma, int ny, int nx, float klim);
void update_total_plain(float * sigma, float * sigma_max, float * param_max, int * hit_tot, float * kmap, int ny, int nx, float r, float vy, float vx);

#if 0
double displace_map2( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
double displace_map3( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
double displace_map4( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
double displace_map5( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
double displace_map6( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
double displace_map7( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
double displace_map8( float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx);
#endif
