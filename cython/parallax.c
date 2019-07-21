#include <math.h>
#include "immintrin.h"

inline void displace_pos(double dec, double ra, double * earth_pos, double r, double dy, double dx, double * odec, double * ora)
{
	double cdec = cos(dec);
	ra += dx/cdec; dec += dy;
	double x = cdec*cos(ra)*r - earth_pos[0];
	double y = cdec*sin(ra)*r - earth_pos[1];
	double z = sin(dec)    *r - earth_pos[2];
	double r_hor = sqrt(x*x+y*y);
	*odec = atan2(z,r_hor);
	*ora  = atan2(y,x);
}

// Conceptually what we want is
// raw_pos = omap.posmap()
// obs_pos = raw_pos + dt*vel + parallax
// obs_pix = sky2pix(obs_pos)
// omap    = interpol(imap, obs_pix)
// but doing it directly like that was too slow, so here we specialize to CAR
// projection. Additionally we will do the parallax computation in blocks of pixels,
// since that is a relatively heavy computation. Well, let's implement the straightforward
// version first.

// Like displace_map7, but with openmp. Only a 23% gain on my laptop. It's probably memory bound.
// But this brings the speedup up to a total factor of 480 relative to the straightforward
// enmap+scipy implementation.
#define bs 8
#define bgroup 2
void displace_map_blocks_avx_omp(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	int btot = bs*bgroup;
	int nphi = abs((int)round(2*M_PI/dra));
	// Loop over groups of 8x8 blocks
	#pragma omp parallel for
	for(int gy1 = 0; gy1 < ny; gy1+=btot) {
		int gy2 = gy1 + btot;
		int refy = gy1 + btot/2;
		double ref_dec = dec0 + refy*ddec;
		for(int gx1 = 0; gx1 < nx; gx1+=btot) {
			int gx2 = gx1 + btot;
			int refx = gx1 + btot/2;
			double ref_ra = ra0 + refx*dra;
			// For each such group compute a single, common displacement at the
			// reference location
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			// Given this offset we will now loop over individual 8x8 blocks and
			// displace them. There are two cases: Full blocks, for which we will
			// use avx, and partial blocks, where we will use normal C.
			for(int y1 = gy1; y1 < gy2; y1 += bs) {
				int y2 = y1 + bs; if(y2 > ny) y2 = ny;
				int fully = (y2-y1 == bs && y1 + yoffi >= 0 && y2 + yoffi < ny-1);
				for(int x1 = gx1; x1 < gx2; x1 += bs) {
					int x2 = x1 + bs; if(x2 > nx) x2 = nx;
					int fullx = (x2-x1 == bs && x1 + xoffi >= 0 && x2 + xoffi < nx-1);
					if(fully && fullx) {
						// Full block
						int ypix = y1+yoffi, xpix = x1+xoffi;
						__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot, old;
						xr1 = _mm256_set1_ps(1-xrel);
						xr2 = _mm256_set1_ps(xrel);
						yr1 = _mm256_set1_ps(1-yrel);
						yr2 = _mm256_set1_ps(yrel);
						v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
						v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
						v1  = _mm256_fmadd_ps(v11, xr1, _mm256_mul_ps(v12,xr2));
						for(int y = y1; y < y2; y+=2) {
							// even rows
							v21  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix]);
							v22  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix+1]);
							v2   = _mm256_fmadd_ps(v21, xr1, _mm256_mul_ps(v22,xr2));
							vtot = _mm256_fmadd_ps(v1,  yr1, _mm256_mul_ps(v2,yr2));
							old  = _mm256_loadu_ps(&omap[y*nx+x1]);
							vtot = _mm256_add_ps(vtot, old);
							_mm256_storeu_ps(&omap[y*nx+x1], vtot);
							// odd rows
							v11  = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix]);
							v12  = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix+1]);
							v1   = _mm256_fmadd_ps(v11, xr1, _mm256_mul_ps(v12,xr2));
							vtot = _mm256_fmadd_ps(v2,  yr1, _mm256_mul_ps(v1,yr2));
							old  = _mm256_loadu_ps(&omap[(y+1)*nx+x1]);
							vtot = _mm256_add_ps(vtot, old);
							_mm256_storeu_ps(&omap[(y+1)*nx+x1], vtot);
						}
					} else {
						// Partial block
						for(int y = y1; y < y2; y++) {
							int ypix = y + yoffi;
							for(int x = x1; x < x2; x++) {
								if(ypix < 0 || ypix > ny-2)
									continue;
								else {
									// Support wrapping horizontally. Would be more general to use
									// mod here, but that was quite expensive (5-10% speed loss) even
									// though partial blocks are much less common than full blocks.
									// The current version is practically free.
									int xpix = x+xoffi, xpix2 = x+xoffi+1;
									if(xpix < 0) { xpix += nphi; xpix2 += nphi; }
									else if(xpix == nphi-1) { xpix2 = 0; }
									else if(xpix >= nphi) { xpix -= nphi; xpix2 -= nphi; }
									if(xpix > nx || xpix2 > nx)
										continue;
									else {
										float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix2] * xrel;
										float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix2] * xrel;
										omap[y*nx+x] += v1 * (1-yrel) + v2 * yrel;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#if 0

// straightforward implementation
void displace_map1(
		// The input and output maps. Each is [ny,nx] in CAR
		float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra,
		// The x,y,z position of the earth relative to the sun in celestial coordinates
		double * earth_pos,
		// The sun-distance of the object (which determines the parallax displacement), and the orbital offset
		double r, double dy, double dx)
{
	double ra, dec, ora, odec;
	//#pragma omp parallel for private(ra,dec,ora,odec)
	for(int y = 0; y < ny; y++) {
		dec = dec0 + y*ddec;
		for(int x = 0; x < nx; x++) {
			ra = ra0 + x*dra;
			displace_pos(dec, ra, earth_pos, r, dy, dx, &odec, &ora);
			float oy = (odec - dec0)/ddec;
			float ox = (ora  - ra0 )/dra;
			// Bilinear interpolation
			int   ypix = (int)oy,   xpix = (int)ox;
			float yrel = oy - ypix, xrel = ox - xpix;
			if(ypix < 0 || ypix > ny-2 || xpix < 0 || xpix > nx-2) omap[y*nx+x] = 0;
			else {
				float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix+1] * xrel;
				float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix+1] * xrel;
				omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
			}
		}
	}
}

// reuse offsets in blocks
void displace_map2(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	const int bs = 8;
	for(int y1 = 0; y1 < ny; y1+=bs) {
		int y2   = y1+bs; if(y2>ny) y2 = ny;
		int refy = y1+bs/2;
		double ref_dec = dec0 + refy*ddec;
		for(int x1 = 0; x1 < nx; x1+=bs) {
			int x2   = x1+bs; if(x2>nx) x2 = nx;
			int refx = x1+bs/2;
			double ref_ra = ra0 + refx*dra;
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			for(int y = y1; y < y2; y++) {
				int ypix = y + yoffi;
				for(int x = x1; x < x2; x++) {
					int xpix = x + xoffi;
					if(ypix < 0 || ypix > ny-2 || xpix < 0 || xpix > nx-2)
						omap[y*nx+x] = 0;
					else {
						float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix+1] * xrel;
						float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix+1] * xrel;
						omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
					}
				}
			}
		}
	}
}

#if 0
// BUG: Assumes that if row i is aligned, then row (i+1) will be aligned too.
// explicit avx
// Handle the edges separately, so we can use full 8x8 blocks with no bounds checking in
// the interior, both for reads and writes. The writes will be aligned, but the reads
// won't be in general due to the variable displacement.
#define bs 8
void displace_map3(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	for(int y1 = 0; y1 < ny; y1+=bs) {
		int y2   = y1+bs;
		int refy = y1+bs/2;
		double ref_dec = dec0 + refy*ddec;
		// Find offset of first aligned element in omap
		int align = ((uintptr_t)(&omap[y1*ny]) & 0x1f)/4;
		//fprintf(stderr, "align: %2d %p %p\n", align, &omap[y1*ny], &omap[y1*ny-align]);
		for(int x1 = -align; x1 < nx; x1+=bs) {
			int x2   = x1+bs;
			int refx = x1+bs/2;
			double ref_ra = ra0 + refx*dra;
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			// Are we at an edge? If so use the displace_map2 strategy
			if(x1 < 0 || y1 + yoffi < 0 || y2 + yoffi >= ny-1 || x1 + xoffi < 0 || x2 + xoffi >= nx-1) {
				for(int y = y1; y < y2; y++) {
					int ypix = y + yoffi;
					for(int x = x1; x < x2; x++) {
						if(x < 0) continue;
						int xpix = x + xoffi;
						if(ypix < 0 || ypix > ny-2 || xpix < 0 || xpix > nx-2)
							omap[y*nx+x] = 0;
						else {
							float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix+1] * xrel;
							float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix+1] * xrel;
							omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
						}
					}
				}
			}
			else {
				// we're in the interior
				int ypix = y1+yoffi, xpix = x1+xoffi;
				__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot;
				xr1 = _mm256_set1_ps(1-xrel);
				xr2 = _mm256_set1_ps(xrel);
				yr1 = _mm256_set1_ps(1-yrel);
				yr2 = _mm256_set1_ps(yrel);
				v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
				v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
				v1  = _mm256_add_ps(_mm256_mul_ps(v11,xr1), _mm256_mul_ps(v12,xr2));
				for(int y = y1; y < y2; y+=2) {
					// even rows
					v21  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix]);
					v22  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix+1]);
					v2   = _mm256_add_ps(_mm256_mul_ps(v21,xr1), _mm256_mul_ps(v22,xr2));
					vtot = _mm256_add_ps(_mm256_mul_ps(v1,yr1),  _mm256_mul_ps(v2,yr2));
					_mm256_store_ps(&omap[y*nx+x1], vtot);
					// odd rows
					v11 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix]);
					v12 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix+1]);
					v1  = _mm256_add_ps(_mm256_mul_ps(v11,xr1), _mm256_mul_ps(v12,xr2));
					vtot = _mm256_add_ps(_mm256_mul_ps(v2,yr1), _mm256_mul_ps(v1,yr2));
					_mm256_store_ps(&omap[(y+1)*nx+x1], vtot);
				}
			}
		}
	}
}
#endif

// explicit avx
// Handle the edges separately, so we can use full 8x8 blocks with no bounds checking in
// the interior, both for reads and writes. The writes will be aligned, but the reads
// won't be in general due to the variable displacement.
#define bs 8
void displace_map3(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	for(int y1 = 0; y1 < ny; y1+=bs) {
		int y2   = y1+bs;
		int y2cap= y2 < ny ? y2 : ny;
		int refy = y1+bs/2;
		double ref_dec = dec0 + refy*ddec;
		for(int x1 = 0; x1 < nx; x1+=bs) {
			int x2   = x1+bs;
			int x2cap= x2 < nx ? x2 : nx;
			int refx = x1+bs/2;
			double ref_ra = ra0 + refx*dra;
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			// Are we at an edge? If so use the displace_map2 strategy
			if(x1 < 0 || y1 + yoffi < 0 || y2 + yoffi >= ny-1 || x1 + xoffi < 0 || x2 + xoffi >= nx-1) {
				for(int y = y1; y < y2cap; y++) {
					int ypix = y + yoffi;
					for(int x = x1; x < x2cap; x++) {
						int xpix = x + xoffi;
						if(ypix < 0 || ypix > ny-2 || xpix < 0 || xpix > nx-2)
							omap[y*nx+x] = 0;
						else {
							float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix+1] * xrel;
							float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix+1] * xrel;
							omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
						}
					}
				}
			}
			else {
				// we're in the interior
				int ypix = y1+yoffi, xpix = x1+xoffi;
				__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot;
				xr1 = _mm256_set1_ps(1-xrel);
				xr2 = _mm256_set1_ps(xrel);
				yr1 = _mm256_set1_ps(1-yrel);
				yr2 = _mm256_set1_ps(yrel);
				v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
				v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
				v1  = _mm256_add_ps(_mm256_mul_ps(v11,xr1), _mm256_mul_ps(v12,xr2));
				for(int y = y1; y < y2; y+=2) {
					// even rows
					v21  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix]);
					v22  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix+1]);
					v2   = _mm256_add_ps(_mm256_mul_ps(v21,xr1), _mm256_mul_ps(v22,xr2));
					vtot = _mm256_add_ps(_mm256_mul_ps(v1,yr1),  _mm256_mul_ps(v2,yr2));
					_mm256_storeu_ps(&omap[y*nx+x1], vtot);
					// odd rows
					v11 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix]);
					v12 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix+1]);
					v1  = _mm256_add_ps(_mm256_mul_ps(v11,xr1), _mm256_mul_ps(v12,xr2));
					vtot = _mm256_add_ps(_mm256_mul_ps(v2,yr1), _mm256_mul_ps(v1,yr2));
					_mm256_storeu_ps(&omap[(y+1)*nx+x1], vtot);
				}
			}
		}
	}
}

#define bgroup 2
// AVX with larger blocks. Blocks larger than 2 don't help
void displace_map4(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	int btot = bs*bgroup;
	// Loop over groups of 8x8 blocks
	for(int gy1 = 0; gy1 < ny; gy1+=btot) {
		int gy2 = gy1 + btot;
		int refy = gy1 + btot/2;
		double ref_dec = dec0 + refy*ddec;
		for(int gx1 = 0; gx1 < nx; gx1+=btot) {
			int gx2 = gx1 + btot;
			int refx = gx1 + btot/2;
			double ref_ra = ra0 + refx*dra;
			// For each such group compute a single, common displacement at the
			// reference location
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			// Given this offset we will now loop over individual 8x8 blocks and
			// displace them. There are two cases: Full blocks, for which we will
			// use avx, and partial blocks, where we will use normal C.
			for(int y1 = gy1; y1 < gy2; y1 += bs) {
				int y2 = y1 + bs; if(y2 > ny) y2 = ny;
				int fully = (y2-y1 == bs && y1 + yoffi >= 0 && y2 + yoffi < ny-1);
				for(int x1 = gx1; x1 < gx2; x1 += bs) {
					int x2 = x1 + bs; if(x2 > nx) x2 = nx;
					int fullx = (x2-x1 == bs && x1 + xoffi >= 0 && x2 + xoffi < nx-1);
					if(fully && fullx) {
						// Full block
						int ypix = y1+yoffi, xpix = x1+xoffi;
						__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot;
						xr1 = _mm256_set1_ps(1-xrel);
						xr2 = _mm256_set1_ps(xrel);
						yr1 = _mm256_set1_ps(1-yrel);
						yr2 = _mm256_set1_ps(yrel);
						v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
						v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
						v1  = _mm256_add_ps(_mm256_mul_ps(v11,xr1), _mm256_mul_ps(v12,xr2));
						for(int y = y1; y < y2; y+=2) {
							// even rows
							v21  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix]);
							v22  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix+1]);
							v2   = _mm256_add_ps(_mm256_mul_ps(v21,xr1), _mm256_mul_ps(v22,xr2));
							vtot = _mm256_add_ps(_mm256_mul_ps(v1,yr1),  _mm256_mul_ps(v2,yr2));
							_mm256_storeu_ps(&omap[y*nx+x1], vtot);
							// odd rows
							v11 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix]);
							v12 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix+1]);
							v1  = _mm256_add_ps(_mm256_mul_ps(v11,xr1), _mm256_mul_ps(v12,xr2));
							vtot = _mm256_add_ps(_mm256_mul_ps(v2,yr1), _mm256_mul_ps(v1,yr2));
							_mm256_storeu_ps(&omap[(y+1)*nx+x1], vtot);
						}
					} else {
						// Partial block
						for(int y = y1; y < y2; y++) {
							int ypix = y + yoffi;
							for(int x = x1; x < x2; x++) {
								int xpix = x + xoffi;
								if(ypix < 0 || ypix > ny-2 || xpix < 0 || xpix > nx-2)
									omap[y*nx+x] = 0;
								else {
									float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*ny+xpix+1] * xrel;
									float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*ny+xpix+1] * xrel;
									omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
								}
							}
						}
					}
				}
			}
		}
	}
}

#define bgroup 2
// AVX with larger block and also fused-multiply-add. This gave a marginal speedup.
void displace_map5(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	int btot = bs*bgroup;
	int nphi = abs((int)round(2*M_PI/dra));
	// Loop over groups of 8x8 blocks
	for(int gy1 = 0; gy1 < ny; gy1+=btot) {
		int gy2 = gy1 + btot;
		int refy = gy1 + btot/2;
		double ref_dec = dec0 + refy*ddec;
		for(int gx1 = 0; gx1 < nx; gx1+=btot) {
			int gx2 = gx1 + btot;
			int refx = gx1 + btot/2;
			double ref_ra = ra0 + refx*dra;
			// For each such group compute a single, common displacement at the
			// reference location
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			// Given this offset we will now loop over individual 8x8 blocks and
			// displace them. There are two cases: Full blocks, for which we will
			// use avx, and partial blocks, where we will use normal C.
			for(int y1 = gy1; y1 < gy2; y1 += bs) {
				int y2 = y1 + bs; if(y2 > ny) y2 = ny;
				int fully = (y2-y1 == bs && y1 + yoffi >= 0 && y2 + yoffi < ny-1);
				for(int x1 = gx1; x1 < gx2; x1 += bs) {
					int x2 = x1 + bs; if(x2 > nx) x2 = nx;
					int fullx = (x2-x1 == bs && x1 + xoffi >= 0 && x2 + xoffi < nx-1);
					if(fully && fullx) {
						// Full block
						int ypix = y1+yoffi, xpix = x1+xoffi;
						__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot;
						xr1 = _mm256_set1_ps(1-xrel);
						xr2 = _mm256_set1_ps(xrel);
						yr1 = _mm256_set1_ps(1-yrel);
						yr2 = _mm256_set1_ps(yrel);
						v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
						v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
						v1  = _mm256_fmadd_ps(v11, xr1, _mm256_mul_ps(v12,xr2));
						for(int y = y1; y < y2; y+=2) {
							// even rows
							v21  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix]);
							v22  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix+1]);
							v2   = _mm256_fmadd_ps(v21, xr1, _mm256_mul_ps(v22,xr2));
							vtot = _mm256_fmadd_ps(v1,  yr1,  _mm256_mul_ps(v2,yr2));
							_mm256_storeu_ps(&omap[y*nx+x1], vtot);
							// odd rows
							v11 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix]);
							v12 = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix+1]);
							v1  = _mm256_fmadd_ps(v11, xr1, _mm256_mul_ps(v12,xr2));
							vtot = _mm256_fmadd_ps(v2,yr1, _mm256_mul_ps(v1,yr2));
							_mm256_storeu_ps(&omap[(y+1)*nx+x1], vtot);
						}
					} else {
						// Partial block
						for(int y = y1; y < y2; y++) {
							int ypix = y + yoffi;
							for(int x = x1; x < x2; x++) {
								if(ypix < 0 || ypix > ny-2)
									omap[y*nx+x] = 0;
								else {
									// Support wrapping horizontally. Would be more general to use
									// mod here, but that was quite expensive (5-10% speed loss) even
									// though partial blocks are much less common than full blocks.
									// The current version is practically free.
									int xpix = x+xoffi, xpix2 = x+xoffi+1;
									if(xpix < 0) { xpix += nphi; xpix2 += nphi; }
									else if(xpix == nphi-1) { xpix2 = 0; }
									else if(xpix >= nphi) { xpix -= nphi; xpix2 -= nphi; }
									if(xpix > nx || xpix2 > nx)
										omap[y*nx+x] = 0;
									else {
										float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix2] * xrel;
										float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix2] * xrel;
										omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// line-wise avx, with pre-computation of offsets in blocks. Incomplete, but already
// 30% slower than the block-wise one, and more complicated than it
#define gridres 16
void displace_map6(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	// Allocate our grid
	int ngy = (ny + gridres-1)/gridres;
	int ngx = (nx + gridres-1)/gridres;
	int gsize = ngy*ngx;
	int   * grid_yoff = (int*)malloc(gsize*sizeof(int));
	int   * grid_xoff = (int*)malloc(gsize*sizeof(int));
	float * grid_yrel = (float*)malloc(gsize*sizeof(float));
	float * grid_xrel = (float*)malloc(gsize*sizeof(float));
	// and build it
	for(int gy = 0; gy < ngy; gy++) {
		int refy = gy*gridres+gridres/2;
		double ref_dec = dec0 + refy*ddec;
		for(int gx = 0; gx < ngx; gx++) {
			int refx = gx*gridres+gridres/2;
			int gi   = gy*ngx+gx;
			double ref_ra = ra0 + refx*dra;
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			grid_yoff[gi] = (int)floor(yoff);
			grid_xoff[gi] = (int)floor(xoff);
			grid_yrel[gi] = yoff-grid_yoff[gi];
			grid_xrel[gi] = xoff-grid_xoff[gi];
		}
	}
	// Now loop through output rows
	for(int y = 0; y < ny; y++) {
		int align = ((uintptr_t)(&omap[y*ny]) & 0x1f)/4;
		if(align) align = 8 - align;
		int ax1 = align, ax2 = (nx-align)/bs*bs;
		// Handle the unaligned stuff
		//for(int x = 0; x < ax1; x++) {
		//	int gi = y/gridres*ngx;
		//	int   ypix = y+grid_yoff[gi], xpix = x+grid_xoff[gi];
		//	float yrel = grid_yrel[gi],   xrel = grid_xrel[gi];
		//	if(ypix < 0 || ypix >= ny-1 || xpix < 0 || xpix >= nx-1)
		//		omap[y*nx+x] = 0;
		//	else {
		//		float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix+1] * xrel;
		//		float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix+1] * xrel;
		//		omap[y*nx+x] = v1 * (1-yrel) + v2 * yrel;
		//	}
		//}
		// Same for the end, just with for(int x = ax2; x < nx; x++)
		// Then the aligned blocks
		for(int x = ax1; x < ax2; x+= bs) {
			int   gi   = y/gridres*ngx + x/gridres;
			int   ypix = y+grid_yoff[gi], xpix = x+grid_xoff[gi];
			float yrel = grid_yrel[gi],   xrel = grid_xrel[gi];
			// Are we at a boundary?
			if(ypix < 0 || ypix >= ny-1 || xpix < 0 || xpix+bs >= nx-1) {

			} else {
				__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot;
				xr1 = _mm256_set1_ps(1-xrel);
				xr2 = _mm256_set1_ps(xrel);
				yr1 = _mm256_set1_ps(1-yrel);
				yr2 = _mm256_set1_ps(yrel);
				v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
				v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
				v1  = _mm256_add_ps(_mm256_mul_ps(v11, xr1), _mm256_mul_ps(v12,xr2));
				v21 = _mm256_loadu_ps (&imap[(ypix+1)*nx+xpix]);
				v22 = _mm256_loadu_ps (&imap[(ypix+1)*nx+xpix+1]);
				v2  = _mm256_add_ps(_mm256_mul_ps(v21, xr1), _mm256_mul_ps(v22,xr2));
				vtot= _mm256_add_ps(_mm256_mul_ps(v1,  yr1), _mm256_mul_ps(v2,yr2));
				_mm256_storeu_ps(&omap[y*nx+x], vtot);
			}
		}
	}
}

// Like displace_map5, but accumulates into output
void displace_map7(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx) {
	int btot = bs*bgroup;
	int nphi = abs((int)round(2*M_PI/dra));
	// Loop over groups of 8x8 blocks
	for(int gy1 = 0; gy1 < ny; gy1+=btot) {
		int gy2 = gy1 + btot;
		int refy = gy1 + btot/2;
		double ref_dec = dec0 + refy*ddec;
		for(int gx1 = 0; gx1 < nx; gx1+=btot) {
			int gx2 = gx1 + btot;
			int refx = gx1 + btot/2;
			double ref_ra = ra0 + refx*dra;
			// For each such group compute a single, common displacement at the
			// reference location
			double odec, ora;
			displace_pos(ref_dec, ref_ra, earth_pos, r, dy, dx, &odec, &ora);
			// This is the pixel offset we will use for all pixels in this block
			float yoff = (odec - dec0)/ddec - refy;
			float xoff = (ora  - ra0 )/dra  - refx;
			int yoffi = (int)floor(yoff);
			int xoffi = (int)floor(xoff);
			float yrel = yoff-yoffi;
			float xrel = xoff-xoffi;
			// Given this offset we will now loop over individual 8x8 blocks and
			// displace them. There are two cases: Full blocks, for which we will
			// use avx, and partial blocks, where we will use normal C.
			for(int y1 = gy1; y1 < gy2; y1 += bs) {
				int y2 = y1 + bs; if(y2 > ny) y2 = ny;
				int fully = (y2-y1 == bs && y1 + yoffi >= 0 && y2 + yoffi < ny-1);
				for(int x1 = gx1; x1 < gx2; x1 += bs) {
					int x2 = x1 + bs; if(x2 > nx) x2 = nx;
					int fullx = (x2-x1 == bs && x1 + xoffi >= 0 && x2 + xoffi < nx-1);
					if(fully && fullx) {
						// Full block
						int ypix = y1+yoffi, xpix = x1+xoffi;
						__m256 xr1, xr2, yr1, yr2, v11, v12, v21, v22, v1, v2, vtot, old;
						xr1 = _mm256_set1_ps(1-xrel);
						xr2 = _mm256_set1_ps(xrel);
						yr1 = _mm256_set1_ps(1-yrel);
						yr2 = _mm256_set1_ps(yrel);
						v11 = _mm256_loadu_ps (&imap[ypix*nx+xpix]);
						v12 = _mm256_loadu_ps (&imap[ypix*nx+xpix+1]);
						v1  = _mm256_fmadd_ps(v11, xr1, _mm256_mul_ps(v12,xr2));
						for(int y = y1; y < y2; y+=2) {
							// even rows
							v21  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix]);
							v22  = _mm256_loadu_ps (&imap[(y+yoffi+1)*nx+xpix+1]);
							v2   = _mm256_fmadd_ps(v21, xr1, _mm256_mul_ps(v22,xr2));
							vtot = _mm256_fmadd_ps(v1,  yr1,  _mm256_mul_ps(v2,yr2));
							old  = _mm256_loadu_ps(&omap[y+nx+x1]);
							vtot = _mm256_add_ps(vtot, old);
							_mm256_storeu_ps(&omap[y*nx+x1], vtot);
							// odd rows
							v11  = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix]);
							v12  = _mm256_loadu_ps (&imap[(y+yoffi+2)*nx+xpix+1]);
							v1   = _mm256_fmadd_ps(v11, xr1, _mm256_mul_ps(v12,xr2));
							vtot = _mm256_fmadd_ps(v2,yr1, _mm256_mul_ps(v1,yr2));
							old  = _mm256_loadu_ps(&omap[(y+1)+nx+x1]);
							vtot = _mm256_add_ps(vtot, old);
							_mm256_storeu_ps(&omap[(y+1)*nx+x1], vtot);
						}
					} else {
						// Partial block
						for(int y = y1; y < y2; y++) {
							int ypix = y + yoffi;
							for(int x = x1; x < x2; x++) {
								if(ypix < 0 || ypix > ny-2)
									continue;
								else {
									// Support wrapping horizontally. Would be more general to use
									// mod here, but that was quite expensive (5-10% speed loss) even
									// though partial blocks are much less common than full blocks.
									// The current version is practically free.
									int xpix = x+xoffi, xpix2 = x+xoffi+1;
									if(xpix < 0) { xpix += nphi; xpix2 += nphi; }
									else if(xpix == nphi-1) { xpix2 = 0; }
									else if(xpix >= nphi) { xpix -= nphi; xpix2 -= nphi; }
									if(xpix > nx || xpix2 > nx)
										continue;
									else {
										float v1 = imap[(ypix+0)*nx+xpix] * (1-xrel) + imap[(ypix+0)*nx+xpix2] * xrel;
										float v2 = imap[(ypix+1)*nx+xpix] * (1-xrel) + imap[(ypix+1)*nx+xpix2] * xrel;
										omap[y*nx+x] += v1 * (1-yrel) + v2 * yrel;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#endif
