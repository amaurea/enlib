module pmat_core

contains

! Performance for tod[445,262000] on laptop with OMP_NUM_THREADS=4
!  dp_all   bounds tri Ofast   : 3.6131 3.6265 3.6019
!  dp_bore  bounds tri Ofast   : 3.8680 3.8867 3.8988
!  sp_all   bounds tri Ofast   : 3.3886 3.3987 3.3839
!  sp_all   bounds tri Ofast   : 3.4127 3.3815 3.4020
!  sp_all  !bounds tri Ofast   : 3.3271 3.3300 3.3494
!  sp_all   bounds grd Ofast   : 2.4928 2.4915 2.5022
!  sp_all   bounds ner Ofast   : 2.1450 2.1151 2.1395
!  sp_all   bounds tri native  : 3.5933 3.6151 3.6027 Slower??
!  sp_all   bounds tri O2      : 3.5449 3.5517 3.5351
!  sp_all   bounds tri Os      : 4.1127 4.1443 4.1305
!  sp_all   bounds tri Of omit : 3.3724 3.4091 3.3960
!  sp_all   bounds tri Of col1 : 3.3961 3.4003 3.4068
!
! Interpolation accuracy at 1e-3,1e-3,arcsec,arcsec
!  std(tri-grd): 4.76968216E-06   2.86180921E-06   1.63026237E-09   9.31578481E-10
! No point in using the 35% slower tridiagonal interpolation. In that case,
! is there a point in using the clunky ys interface?
!
! nint is 5% slower than floor

	! Implements projection from tod to map and vice versa.
	! Please forgive all the repeated code in it. If statements
	! in the loops proved too expensive, as did function calls.
	! But the logic is actually pretty simple. We reorder the
	! maps for better memory locality, and then loop over each
	! tod sample, computing its pixel and polarization information
	! on the fly.
	!
	! For my last benchmark [602,262144] tod on scinet, this one took
	!  1.05  1.20 s @ dp 16 cores gfortran
	!  1.33  1.48 s @ dp  8 cores gfortran
	!  2.67  3.18 s @ dp  4 cores gfortran
	! 10.31 11.03 s @ dp  1 core  gfortran
	!  0.96  1.11 s @ sp 16 cores gfortran
	!  1.15  1.35 s @ sp  8 cores gfortran
	!  2.31  2.60 s @ sp  4 cores gfortran
	!  8.82  9.36 s @ sp  1 core  gfortran
	!  0.83  1.30 s @ dp 16 cores ifort after adding --noopt to f2py
	!  0.66  1.12 s @ dp 16 cores ifort after removing collapse statements
	!  0.63  0.90 s @ sp 16 cores
	!  --------------------------
	!  1.77  1.07 s @ dp 16 cores ninkasi
	! Since precision does not save much on speed, probably due to the majority
	! of the logic here being in pointing, which isn't affected. But it does
	! save on memory, of course. Scaling is quite good. We get a bit more than
	! half the max possible speedup when going from 1 to 16 cores. We are slightly
	! faster than ninkasi here, but not much: 2.25 s vs. 2.84 s (26% faster)
	subroutine pmat_nearest( &
			dir,                       &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
			tmul, mmul,                &! Consts to multiply tod/map by
			tod, map,                  &! Main inputs/outpus
			bore, det_pos, det_comps,  &! Input pointing
			comps,                     &! Ignored. Supporting arbitrary component ordering was too expensive
			rbox, nbox, ys, pbox       &! Coordinate transformation
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, nbox(:), pbox(:,:), comps(:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:)
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:), tmul, mmul
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		! Work
		integer(4) :: ndet, nsamp, ncomp, nproc, di, si, id, ic
		integer(4) :: steps(size(rbox,1)), psize(2)
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1))
		real(_),    allocatable :: wmap3(:,:,:), wmap4(:,:,:,:)
		real(_)    :: xrel(3), point(4), phase(3)
		integer(4) :: xind(3), ig, pix(2), ix, iy

		nsamp   = size(tod, 1)
		ndet    = size(tod, 2)
		ncomp   = size(map, 3)
		psize   = pbox(:,2)-pbox(:,1)
		nproc   = omp_get_max_threads()
		! In C order, ys has pixel axes t,ra,dec, so nbox = [nt,nra,ndec]
		! Index mapping is therefore given by [nra*ndec,ndec,1]
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		if(dir > 0) then
			! Forward transform - no worry of clobbering, so we can use a
			! single work map
			allocate(wmap3(3,psize(2),psize(1)))
			!$omp parallel do collapse(2)
			do iy = 1, size(wmap3,3)
				do ix = 1, size(wmap3,2)
					wmap3(1:ncomp,ix,iy) = map(ix+pbox(2,1),iy+pbox(1,1),1:ncomp)
					wmap3(ncomp+1:3,ix,iy) = 0
				end do
			end do
			if(mmul .ne. 1) then
				!$omp parallel workshare
				wmap3 = wmap3 * mmul
				!$omp end parallel workshare
			end if
			if(tmul .eq. 0) then
				!$omp parallel do private(di, si, xrel, xind, ig, point, pix, phase)
				do di = 1, ndet
					do si = 1, nsamp
						! Compute interpolated pointing. This used to be done via a function
						! call, but that proved too costly.
						xrel = (bore(:,si)+det_pos(:,di)-x0)*inv_dx
						xind = floor(xrel)
						xrel = xrel - xind
						ig   = sum(xind*steps)+1
						! Pointing is done via gradient interpolation, which is one
						! step simpler than multilinear interpolation because it
						! ignores cross terms. This means that pointing will be
						! discontinuous at some corners.
						!
						! Costs of some approaches (assuming we already have xrel and ig)
						!                      mul add
						! 1. Nearest neighbor    0  0  equivalent to precomputing all pixels
						! 2. Gradient3          12 12  discontinuous
						! 2. Gradient2           8  8
						! 3. Implicit grad3     12 24  same as above, but more general storage
						! 3. Implicit grad2      8 16
						! 4. Bigradient3        12 15  continuous
						! 4. Bigradient2         8 10
						! 5. Impl. Bigrad3      12 27  as above, more general storage
						! 5. Impl. Bigrad2       9 18
						! 6. Bilinear3          56 28  differentiable in cell
						! 6. Bilinear2          24 12
						! 7. Linear 1d          -3 -8  may be infeasible. Per-detector
						! Elevation support is quite costly for the more advanced methods.
						! But without it, we need ~1000 times more interpolation points,
						! and can't handle elevation jitter.
						point = ys(:,1,ig) + xrel(1)*ys(:,2,ig) + xrel(2)*ys(:,3,ig) + xrel(3)*ys(:,4,ig)
						pix = nint(point(1:2))+1 - pbox(:,1)
						! Bounds checking. Costs 2% performance. Worth it
						pix(1) = min(psize(1),max(1,pix(1)))
						pix(2) = min(psize(2),max(1,pix(2)))
						! Compute signal polarization projection parameters.
						! Checking which components to compute takes 17% longer than just computing
						! all of them, which takes about the same time as computing one.
						phase(1) = det_comps(1,di)
						phase(2) = point(3)*det_comps(2,di) - point(4)*det_comps(3,di)
						phase(3) = point(4)*det_comps(2,di) + point(3)*det_comps(3,di)
						! If tests are not free, despite branch prediction. Adding a test
						! for dir here takes the time from 0.85 to 1.15, a 35% increase!
						tod(si,di) = sum(wmap3(:,pix(2),pix(1))*phase)
					end do
				end do
			else
				!$omp parallel do private(di, si, xrel, xind, ig, point, pix, phase)
				do di = 1, ndet
					do si = 1, nsamp
						xrel = (bore(:,si)+det_pos(:,di)-x0)*inv_dx
						xind = floor(xrel)
						xrel = xrel - xind
						ig   = sum(xind*steps)+1
						point = ys(:,1,ig) + xrel(1)*ys(:,2,ig) + xrel(2)*ys(:,3,ig) + xrel(3)*ys(:,4,ig)
						pix = nint(point(1:2))+1 - pbox(:,1)
						pix(1) = min(psize(1),max(1,pix(1)))
						pix(2) = min(psize(2),max(1,pix(2)))
						phase(1) = det_comps(1,di)
						phase(2) = point(3)*det_comps(2,di) - point(4)*det_comps(3,di)
						phase(3) = point(4)*det_comps(2,di) + point(3)*det_comps(3,di)
						tod(si,di) = tod(si,di) + sum(wmap3(:,pix(2),pix(1))*phase)
					end do
				end do
			end if
			deallocate(wmap3)
		else
			! Backwards transform. Here there is a risk of multiple
			! threads clobbering each other, so we either need separate
			! work spaces or critical sections.
			allocate(wmap4(3,psize(2),psize(1),nproc))
			!$omp parallel private(di, si, xrel, xind, ig, point, pix, phase,id)
			id = omp_get_thread_num()+1
			!$omp workshare
			wmap4 = 0
			!$omp end workshare
			!$omp do
			do di = 1, ndet
				do si = 1, nsamp
					xrel = (bore(:,si)+det_pos(:,di)-x0)*inv_dx
					xind = floor(xrel)
					xrel = xrel - xind
					ig   = sum(xind*steps)+1
					point = ys(:,1,ig) + xrel(1)*ys(:,2,ig) + xrel(2)*ys(:,3,ig) + xrel(3)*ys(:,4,ig)
					pix = nint(point(1:2))+1 - pbox(:,1)
					pix(1) = min(psize(1),max(1,pix(1)))
					pix(2) = min(psize(2),max(1,pix(2)))
					phase(1) = det_comps(1,di)
					phase(2) = point(3)*det_comps(2,di) - point(4)*det_comps(3,di)
					phase(3) = point(4)*det_comps(2,di) + point(3)*det_comps(3,di)
					wmap4(:,pix(2),pix(1),id) = wmap4(:,pix(2),pix(1),id) + tod(si,di)*phase
				end do
			end do
			!$omp end parallel
			! Copy out result. Applying mmul and tmul here costs 1%
			!$omp parallel do collapse(1) private(iy,ix,ic)
			do iy = 1, size(wmap4,3)
				do ix = 1, size(wmap4,2)
					do ic = 1, ncomp
						map(ix+pbox(2,1),iy+pbox(1,1),ic) = map(ix+pbox(2,1),iy+pbox(1,1),ic)*mmul + sum(wmap4(ic,ix,iy,:))*tmul
					end do
				end do
			end do
			deallocate(wmap4)
		end if
	end subroutine

	subroutine pmat_ninkasi( &
			dir,                       &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
			tmul, mmul,                &! Consts to multiply tod/map by
			tod, map,                  &! Main inputs/outpus
			bore, box, pbox,           &! Input pointing
			posfit, polfit             &! Ninkasi pointing model
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, pbox(:,:)
		real(_),    intent(in)    :: bore(:,:)
		real(_),    intent(in)    :: tmul, mmul, posfit(:,:,:), polfit(:,:,:), box(:,:)
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		! Work
		integer(4) :: ndet, nsamp, ncomp, nproc, di, si, id, ic, iy, ix, pix(2), psize(2)
		real(_),    allocatable :: wmap3(:,:,:), wmap4(:,:,:,:)
		real(_)    :: point(2), phase(3), tazel(3), boff(3), bscale(3)

		nsamp   = size(tod, 1)
		ndet    = size(tod, 2)
		ncomp   = size(map, 3)

		! Our model is pos[{y,x},det,samp] = posfit[{y,x},det,bi] * basis[bi,samp]
		! pol[{cos,sin},det,samp] = polfit[{cos,sin},det,bi] * polbasis[bi,samp]
		! basis[bi] = [az**4, az**3, az**2, az**1, az**0, el**2, el, t**2, t, t*az]

		psize   = pbox(:,2)-pbox(:,1)
		boff    = box(:,1)
		bscale  = 2/(box(:,2)-box(:,1))
		nproc   = omp_get_max_threads()
		if(dir > 0) then
			! Forward transform - no worry of clobbering, so we can use a
			! single work map
			allocate(wmap3(3,psize(2),psize(1)))
			!$omp parallel do collapse(2)
			do iy = 1, size(wmap3,3)
				do ix = 1, size(wmap3,2)
					wmap3(1:ncomp,ix,iy) = map(ix+pbox(2,1),iy+pbox(1,1),1:ncomp)
					wmap3(ncomp+1:3,ix,iy) = 0
				end do
			end do
			if(mmul .ne. 1) then
				!$omp parallel workshare
				wmap3 = wmap3 * mmul
				!$omp end parallel workshare
			end if
			if(tmul .eq. 0) then
				!$omp parallel do private(di, si, tazel, point, pix, phase)
				do di = 1, ndet
					do si = 1, nsamp
						! Set up our basis for this sample
						tazel = (bore(:,si)-boff)*bscale-1
						point = tazel(2)**4*posfit(:,di,1) + tazel(2)**3*posfit(:,di,2) + &
						        tazel(2)**2*posfit(:,di,3) + tazel(2)**1*posfit(:,di,4) + &
						        tazel(2)**0*posfit(:,di,5) + tazel(3)**2*posfit(:,di,6) + &
						        tazel(3)**1*posfit(:,di,7) + tazel(1)**2*posfit(:,di,8) + &
						        tazel(1)**1*posfit(:,di,9) + tazel(1)*tazel(2)*posfit(:,di,10)
						pix = nint(point)+1-pbox(:,1)
						! Bounds checking. Costs 2% performance. Worth it
						pix(1) = min(psize(1),max(1,pix(1)))
						pix(2) = min(psize(2),max(1,pix(2)))
						! Compute signal polarization projection parameters.
						phase(1) = 1
						phase(2:3) = tazel(2)**0*polfit(:,di,1) + tazel(2)**1*polfit(:,di,2) + &
						             tazel(2)**2*polfit(:,di,3) + tazel(2)**3*polfit(:,di,4) + &
						             tazel(1)**1*polfit(:,di,5)
						tod(si,di) = sum(wmap3(:,pix(2),pix(1))*phase)
					end do
				end do
			else
				!$omp parallel do private(di, si, tazel, point, pix, phase)
				do di = 1, ndet
					do si = 1, nsamp
						! Set up our basis for this sample
						tazel = (bore(:,si)-boff)*bscale-1
						point = tazel(2)**4*posfit(:,di,1) + tazel(2)**3*posfit(:,di,2) + &
						        tazel(2)**2*posfit(:,di,3) + tazel(2)**1*posfit(:,di,4) + &
						        tazel(2)**0*posfit(:,di,5) + tazel(3)**2*posfit(:,di,6) + &
						        tazel(3)**1*posfit(:,di,7) + tazel(1)**2*posfit(:,di,8) + &
						        tazel(1)**1*posfit(:,di,9) + tazel(1)*tazel(2)*posfit(:,di,10)
						pix = nint(point)+1-pbox(:,1)
						! Bounds checking. Costs 2% performance. Worth it
						pix(1) = min(psize(1),max(1,pix(1)))
						pix(2) = min(psize(2),max(1,pix(2)))
						! Compute signal polarization projection parameters.
						phase(1) = 1
						phase(2:3) = tazel(2)**0*polfit(:,di,1) + tazel(2)**1*polfit(:,di,2) + &
						             tazel(2)**2*polfit(:,di,3) + tazel(2)**3*polfit(:,di,4) + &
						             tazel(1)**1*polfit(:,di,5)
						tod(si,di) = tod(si,di) + sum(wmap3(:,pix(2),pix(1))*phase)
					end do
				end do
			end if
			deallocate(wmap3)
		else
			! Backwards transform. Here there is a risk of multiple
			! threads clobbering each other, so we either need separate
			! work spaces or critical sections.
			allocate(wmap4(3,psize(2),psize(1),nproc))
			!$omp parallel private(di, si, tazel, point, pix, phase,id)
			id = omp_get_thread_num()+1
			!$omp workshare
			wmap4 = 0
			!$omp end workshare
			!$omp do
			do di = 1, ndet
				do si = 1, nsamp
						! Set up our basis for this sample
						tazel = (bore(:,si)-boff)*bscale-1
						point = tazel(2)**4*posfit(:,di,1) + tazel(2)**3*posfit(:,di,2) + &
						        tazel(2)**2*posfit(:,di,3) + tazel(2)**1*posfit(:,di,4) + &
						        tazel(2)**0*posfit(:,di,5) + tazel(3)**2*posfit(:,di,6) + &
						        tazel(3)**1*posfit(:,di,7) + tazel(1)**2*posfit(:,di,8) + &
						        tazel(1)**1*posfit(:,di,9) + tazel(1)*tazel(2)*posfit(:,di,10)
						pix = nint(point)+1-pbox(:,1)
						! Bounds checking. Costs 2% performance. Worth it
						pix(1) = min(psize(1),max(1,pix(1)))
						pix(2) = min(psize(2),max(1,pix(2)))
						! Compute signal polarization projection parameters.
						phase(1) = 1
						phase(2:3) = tazel(2)**0*polfit(:,di,1) + tazel(2)**1*polfit(:,di,2) + &
						             tazel(2)**2*polfit(:,di,3) + tazel(2)**3*polfit(:,di,4) + &
						             tazel(1)**1*polfit(:,di,5)
					wmap4(:,pix(2),pix(1),id) = wmap4(:,pix(2),pix(1),id) + tod(si,di)*phase
				end do
			end do
			!$omp end parallel
			! Copy out result. Applying mmul and tmul here costs 1%
			!$omp parallel do collapse(1) private(iy,ix,ic)
			do iy = 1, size(wmap4,3)
				do ix = 1, size(wmap4,2)
					do ic = 1, ncomp
						map(ix+pbox(2,1),iy+pbox(1,1),ic) = map(ix+pbox(2,1),iy+pbox(1,1),ic)*mmul + sum(wmap4(ic,ix,iy,:))*tmul
					end do
				end do
			end do
			deallocate(wmap4)
		end if
	end subroutine


	subroutine pmat_nearest_old( &
			dir,                              &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
			tmul, mmul,                       &! Constants to multiply tod and map by during the projection. for dir<0, mmul has no effect
			tod, map,                         &! Main inputs/outpus
			bore, det_pos, det_comps,  comps, &! Input pointing
			rbox, nbox, ys, pbox              &! Coordinate transformation
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, comps(:), nbox(:), pbox(:,:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:) ! comp, derivs, pix
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:), tmul, mmul
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, nproc, id, ic, i, j, k, l, nj
		integer(4) :: steps(size(rbox,1)), pix(2,bz), psize(2)
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1)), phase(size(map,3),bz)
		real(_)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(_)    :: phase_mul
		real(_), allocatable :: wmap(:,:,:,:)

		nsamp   = size(tod, 1)
		ndet    = size(tod, 2)
		psize   = pbox(:,2)-pbox(:,1)

		! In C order, ys has pixel axes t,ra,dec, so nbox = [nt,nra,ndec]
		! Index mapping is therefore given by [nra*ndec,ndec,1]
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		if(dir < 0) then
			nproc = omp_get_max_threads()
			allocate(wmap(psize(2),psize(1),size(map,3),nproc))
			!allocate(wmap(size(map,1),size(map,2),size(map,3),nproc))
			!$omp parallel workshare
			wmap = 0
			!$omp end parallel workshare
			phase_mul = tmul
		else
			allocate(wmap(psize(2),psize(1),size(map,3),1))
			!$omp parallel workshare
			wmap(:,:,:,1) = map(pbox(2,1)+1:pbox(2,2),pbox(1,1)+1:pbox(1,2),:)
			!$omp end parallel workshare
			phase_mul = mmul
		end if

		!$omp parallel private(di,si,nj,j,ipoint,opoint,phase,pix,id)
		id = omp_get_thread_num()+1
		!$omp do collapse(2)
		do di = 1, ndet
			do si = 1, nsamp, bz
				nj = min(bz,nsamp-si+1)
				do j = 1, nj
					ipoint(:,j) = bore(:,si+j-1)+det_pos(:,di)
				enddo
				opoint(:,:nj) = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
				! We use pixel-center coordinates, so [i-0.5:i+0.5] belongs to
				! pixel i. Hence nint. The extra +1 is due to fortran's indexing.
				pix(:,:nj)    = nint(opoint(1:2,:nj))+1
				do j = 1, nj
					! Transform from global to workspace pixels
					pix(:,j) = pix(:,j)-pbox(:,1)
					! Bounds check (<1% cost)
					pix(1,j) = min(psize(1),max(1,pix(1,j)))
					pix(2,j) = min(psize(2),max(1,pix(2,j)))
				end do
				phase(:,:nj)  = get_phase(comps, det_comps(:,di)*phase_mul, opoint(3:,:nj))
				if(dir < 0) then
					do j = 1, nj
						wmap(pix(2,j),pix(1,j),:,id) = wmap(pix(2,j),pix(1,j),:,id) + tod(si+j-1,di)*phase(:,j)
					end do
				else
					if(tmul==0) then
						do j = 1, nj
							tod(si+j-1,di) = sum(wmap(pix(2,j),pix(1,j),:,1)*phase(:,j))
						end do
					else
						do j = 1, nj
							tod(si+j-1,di) = tod(si+j-1,di)*tmul + sum(wmap(pix(2,j),pix(1,j),:,1)*phase(:,j))
						end do
					end if
				end if
			end do
		end do
		!$omp end parallel

		! Go back to full pixel space
		if(dir < 0) then
			!$omp parallel do collapse(3)
			do j = 1, size(wmap, 3)
				do k = 1, size(wmap, 2)
					do l = 1, size(wmap, 1)
						do i = 1, size(wmap,4)
							map(l+pbox(2,1),k+pbox(1,1),j) = map(l+pbox(2,1),k+pbox(1,1),j) + wmap(l,k,j,i)
						end do
					end do
				end do
			end do
		end if
	end subroutine


	subroutine pmat_linear( &
			dir,                              &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
			tod, map,                         &! Main inputs/outpus
			bore, det_pos, det_comps,  comps, &! Input pointing
			rbox, nbox, ys                    &! Coordinate transformation
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, comps(:), nbox(:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:) ! comp, derivs, pix
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, nproc, id, ic, i, j, k, l, nj, py,px
		integer(4) :: steps(size(rbox,1)), pix(2,bz)
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1)), phase(size(map,3),bz)
		real(_)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(_)    :: fpix(2,bz)
		real(_), allocatable :: wmap(:,:,:,:)

		nsamp   = size(tod, 1)
		ndet    = size(tod, 2)

		! In C order, ys has pixel axes t,ra,dec, so nbox = [nt,nra,ndec]
		! Index mapping is therefore given by [nra*ndec,ndec,1]
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		if(dir < 0) then
			nproc = omp_get_max_threads()
			allocate(wmap(size(map,1),size(map,2),size(map,3),nproc))
			!$omp parallel workshare
			wmap = 0
			!$omp end parallel workshare
		end if

		!$omp parallel private(di,si,nj,j,ipoint,opoint,phase,pix,fpix,id,py,px)
		id = omp_get_thread_num()+1
		!$omp do collapse(2)
		do di = 1, ndet
			do si = 1, nsamp, bz
				nj = min(bz,nsamp-si+1)
				do j = 1, nj
					ipoint(:,j) = bore(:,si+j-1)+det_pos(:,di)
				enddo
				opoint(:,:nj) = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
				pix(:,:nj)    = floor(opoint(1:2,:nj))
				fpix(:,:nj)   = opoint(1:2,:nj)-pix(:,:nj)
				pix(:,:nj)    = pix(:,:nj)+1 ! fortran pixel offset
				! Bounds check (<1% cost)
				phase(:,:nj)  = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
				if(dir < 0) then
					! Different bounds check in each direction, since the offset is -1
					! for tod2map and +1 for map2tod
					do j = 1, nj
						pix(1,j) = min(size(map,2),max(2,pix(1,j)))
						pix(2,j) = min(size(map,1),max(2,pix(2,j)))
					end do
					do j = 1, nj
						phase(:,j) = phase(:,j) * tod(si+j-1,di)
						py = pix(2,j); px = pix(1,j)
						wmap(py-0,px-0,:,id) = wmap(py-0,px-0,:,id) + (1 + fpix(1,j)*fpix(2,j) - fpix(1,j)-fpix(2,j))*phase(:,j)
						wmap(py-1,px-0,:,id) = wmap(py-1,px-0,:,id) + (fpix(1,j)*(1-fpix(2,j)))*phase(:,j)
						wmap(py-0,px-1,:,id) = wmap(py-0,px-1,:,id) + (fpix(2,j)*(1-fpix(1,j)))*phase(:,j)
						wmap(py-1,px-1,:,id) = wmap(py-1,px-1,:,id) + (fpix(1,j)*fpix(2,j))*phase(:,j)
					end do
				else
					! Bounds check, see above
					do j = 1, nj
						pix(1,j) = min(size(map,2)-1,max(1,pix(1,j)))
						pix(2,j) = min(size(map,1)-1,max(1,pix(2,j)))
					end do
					do j = 1, nj
						py = pix(2,j); px = pix(1,j)
						tod(si+j-1,di) = sum((&
							map(py+0,px+0,:)*(1-fpix(1,j)-fpix(2,j)+fpix(1,j)*fpix(2,j)) + &
							map(py+1,px+0,:)*(fpix(1,j)*(1-fpix(2,j))) + &
							map(py+0,px+0,:)*(fpix(2,j)*(1-fpix(1,j))) + &
							map(py+1,px+1,:)*(fpix(1,j)*fpix(2,j)))*phase(:,j))
					end do
				end if
			end do
		end do
		!$omp end parallel

		if(dir < 0) then
			!$omp parallel do collapse(3)
			do j = 1, size(wmap, 3)
				do k = 1, size(wmap, 2)
					do l = 1, size(wmap, 1)
						do i = 1, size(wmap,4)
							map(l,k,j) = map(l,k,j) + wmap(l,k,j,i)
						end do
					end do
				end do
			end do
		end if

	end subroutine

	subroutine translate( &
			bore, pix, phase,           &! Main input/output
			det_pos, det_comps, comps,  &! Pointing metainformation
			rbox, nbox, ys              &! Coordinate transformation
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: comps(:), nbox(:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:) ! comp, derivs, pix
		real(_),    intent(inout) :: phase(:,:,:), pix(:,:,:)
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, id, ic, j, nj
		integer(4) :: steps(size(rbox,1))
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1))
		real(_)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)

		nsamp   = size(bore, 2)
		ndet    = size(det_pos, 2)

		! In C order, ys has pixel axes t,ra,dec, so nbox = [nt,nra,ndec]
		! Index mapping is therefore given by [nra*ndec,ndec,1]
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		!$omp parallel do collapse(2) private(di,si,nj,j,ipoint,opoint,id)
		do di = 1, ndet
			do si = 1, nsamp, bz
				nj = min(bz,nsamp-si+1)
				do j = 1, nj
					ipoint(:,j) = bore(:,si+j-1)+det_pos(:,di)
				enddo
				opoint(:,:nj) = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
				pix(:,si:si+nj-1,di)  = opoint(1:2,:nj)
				phase(:,si:si+nj-1,di) = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
			end do
		end do
	end subroutine

	! Trilinear is 35% slower for almost no gain, it seems.
	!function lookup_trilinear(ipoint, x0, inv_dx, steps, ys) result(opoint)
	!	implicit none
	!	real(_),    intent(in) :: ipoint(:,:), x0(:), inv_dx(:)
	!	real(_),    intent(in) :: ys(:,:,:)
	!	integer(4), intent(in) :: steps(:)
	!	real(_)                :: opoint(size(ys,1),size(ipoint,2))
	!	real(_)    :: xrel(size(x0))
	!	integer(4) :: xind(size(x0)), ig, oc, j, c1, c2, c3, i
	!	do j = 1, size(ipoint,2)
	!		! Our location in grid units
	!		xrel = (ipoint(:,j)-x0)*inv_dx
	!		!write(*,*) "lookup ipoint", ipoint(:,j)
	!		!write(*,*) "lookup x0", x0
	!		!write(*,*) "lookup inv_dx", inv_dx
	!		!write(*,*) "lookup xrel", xrel
	!		xind = floor(xrel)
	!		xrel = xrel - xind ! order t,ra,dec
	!		ig   = sum(xind*steps)+1
	!		opoint(:,j) = ys(:,1,1,1,ig)
	!		opoint(:,j) = opoint(:,j) + xrel(1)*(ys(:,1,1,2,ig)+xrel(2)*(ys(:,1,2,2,ig)+xrel(3)*ys(:,2,2,2,ig)))
	!		opoint(:,j) = opoint(:,j) + xrel(2)*(ys(:,1,2,1,ig)+xrel(3)* ys(:,2,2,1,ig))
	!		opoint(:,j) = opoint(:,j) + xrel(3)*(ys(:,2,1,1,ig)+xrel(1)* ys(:,2,1,2,ig))
	!	end do
	!end function
	pure function lookup_grad(ipoint, x0, inv_dx, steps, ys) result(opoint)
		implicit none
		real(_),    intent(in) :: ipoint(:,:), x0(:), inv_dx(:)
		real(_),    intent(in) :: ys(:,:,:)
		integer(4), intent(in) :: steps(:)
		real(_)                :: opoint(size(ys,1),size(ipoint,2))
		real(_)    :: xrel(size(x0))
		integer(4) :: xind(size(x0)), ig, i, j
		if(size(xrel)==3) then
			do j = 1, size(ipoint,2)
				! Our location in grid units
				xrel = (ipoint(:,j)-x0)*inv_dx
				xind = floor(xrel)
				xrel = xrel - xind ! order t,ra,dec
				ig   = sum(xind*steps)+1
				opoint(:,j) = ys(:,1,ig) + xrel(1)*ys(:,2,ig) + xrel(2)*ys(:,3,ig) + xrel(3)*ys(:,4,ig)
			end do
		else
			do j = 1, size(ipoint,2)
				! Our location in grid units
				xrel = (ipoint(:,j)-x0)*inv_dx
				xind = floor(xrel)
				xrel = xrel - xind ! order t,ra,dec
				ig   = sum(xind*steps)+1
				opoint(:,j) = ys(:,1,ig)
				do i = 1, size(xrel)
					opoint(:,j) = opoint(:,j) + xrel(i)*ys(:,i+1,ig)
				end do
			end do
		end if
	end function

	! This function computes the projection of the polarization degrees
	! of freedom on a TOD sample, such that tod = sum(map*phase)
	! det_comps is the sensitivity of the detector to each degree of
	! freedom in its own coordinate system, while cossin represents
	! cos2psi and sin2psi of the rotation *from* detector coorinates
	! *to* map-space.
	!
	! We can look at this two ways.
	!  1. Rotate the detector sensitivities to map space: sum(map*rot(comps))
	!      oQ = c iQ - s iU
	!      oU = s iQ + c iU
	!  2. Rotate the map to detector coordinates, and then dot them
	!     sum(irot(map)*comps). Here, irot would be the inverse rotation
	!     of the one indicated by cossin. This is equivalent to the one
	!     above because R" = R'.
	!
	! This would result in the implementation
	!  if(comps(i) == 1) phase(i,j) = cossin(1,j)*det_comps(2) - cossin(2,j)*det_comps(3)
	!  if(comps(i) == 2) phase(i,j) = cossin(2,j)*det_comps(2) + cossin(1,j)*det_comps(3)
	!
	! However, we've instead been using what amounts to
	!  oQ = c iQ + s iU
	!  oU = s iQ - c iU
	! This had a sign error in U, which makes sense. But after correcting
	! for that, we get
	!  oQ = c iQ + s iU
	!  oU =-s iQ + c iU
	! That represents the opposite rotation from what one would expect:
	! If the map system in rotated by +45 degrees compared to the detector
	! system (making c=0 and s=1), then a detector with sensitivity (1,0)
	! in its own system would have sensitivity (0,1) in the map system,
	! but with this formula, we get (0,-1).
	!
	! The source of all these sign problems is the handedness of the various
	! coordinate systems. There are *four* coorindate systems involved:
	!  1. The input coordinate system
	!  2. The input polarization system
	!  3. The output coordinate system
	!  4. The output polarization system
	! All of these can be right-handed or left-handed, and can have their
	! axes pointing in various directions. For the polarization coordinate
	! systems, we will use the HEALPix convention, where the local polarization
	! system is always left-handed (as seen from inside the celestial sphere)
	! with x pointing in the theta-direction, i.e. towards the *south*.
	!
	! For normal coordinates, we will still use the IAU convention. It's tempting
	! to use HEALPix here too, as that would make all coordinates be treaded uniformly,
	! but it tends to confuse people, and requires a lot of wrapping of the underlying
	! libraries.
	!
	! When we fix the pol systems to be left-handed, the remaining possiblities are
	!  R->R: psi-phi
	!  L->R: psi-phi+pi
	!  R->L: psi+phi-pi
	!  L->L: psi+phi
	! So basically:
	!  1. If the two axis systems have different handedness, phi -> phi-pi,
	!     because the axis has been moved by pi.
	!  2. If the target axis system has different handedness than the pol systems, phi -> -phi,
	!     becuase this system determines the direction we are measuring phi in, and the sign
	!     changes if this direction is not the same as the polarization angle direction.
	!  3. psi -> psi+phi
	!
	! For our typical case, of Healpix pol (left) and hor (right) -> equ (left), we
	! get psi -> psi+phi-pi. While in our previous implementation, were we used IAU
	! convention (right) for pol angles, we had psi -> psi-(phi-pi), which is
	! consistent with the code. So I think I've understood how this works now.
	!
	! How to handle this in general? I'm willing to hardcode the Healpix convention
	! for polarization angles, but for the rest, we need to detect the handedness of
	! the input and output coordinate systems, not just their relative handedness.
	! This seems like a tall order. Handedness will have to be specified per system.

	pure function get_phase(comps, det_comps, cossin) result(phase)
		implicit none
		integer(4), intent(in) :: comps(:)
		real(_),    intent(in) :: det_comps(:), cossin(:,:)
		real(_)                :: phase(size(comps),size(cossin,2))
		integer(4) :: i, j
		do i = 1, size(phase,1)
			do j = 1, size(cossin,2)
				select case(comps(i))
					case(0); phase(i,j) = det_comps(1)
					case(1); phase(i,j) = cossin(1,j)*det_comps(2) - cossin(2,j)*det_comps(3)
					case(2); phase(i,j) = cossin(2,j)*det_comps(2) + cossin(1,j)*det_comps(3)
					case(3); phase(i,j) = det_comps(4)
				end select
			end do
		end do
	end function

	!!! Cut stuff here !!!

	! Simple cut scheme: a simple array of ({det,lstart,len,gstart,glen,type,params...},ncut)
	! This is very flexible, allowing each cut to be processed independently and with
	! different cut types per cut if needed. The glen part is redundant and could be
	! removed, but makes it easier to interpret the junk array. The disadvantage of this
	! format is that it's quite opaque.
	subroutine pmat_cut(dir, tod, junk, cuts)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, cuts(:,:)
		real(_),    intent(inout) :: tod(:,:), junk(:)
		integer(4), parameter     :: det=1, lstart=2, llen=3, gstart=4, glen=5, cuttype=6
		integer(4) :: ci, di, l1, l2, g1, g2
		!$omp parallel do private(l1,l2,g1,g2,di)
		do ci = 1, size(cuts,2)
			l1 = cuts(lstart,ci)+1; l2 = l1+cuts(llen,ci)-1
			g1 = cuts(gstart,ci)+1; g2 = g1+cuts(glen,ci)-1
			di = cuts(det,ci)+1
			call pmat_cut_range(dir, tod(l1:l2,di), junk(g1:g2), cuts(cuttype:,ci))
		end do
	end subroutine

	! Measure the number of cut parameters corresponding to each range of cut samples
	subroutine measure_cuts(cuts)
		implicit none
		integer(4), intent(inout)  :: cuts(:,:)
		integer(4), parameter      :: det=1, lstart=2, llen=3, gstart=4, glen=5, cuttype=6
		integer(4) :: ci
		real(_) :: foo(1)
		do ci = 1, size(cuts,2)
			call pmat_cut_range(0, foo, foo, cuts(cuttype:,ci), cuts(llen,ci), cuts(glen,ci))
		end do
	end subroutine

	! This handles the cut samples in the tod. It implements
	! several different kinds of cut, which allow tradeoffs between
	! memory use, speed and accuracy, given by cuttype. The possibilities
	! are:
	!   1: full cuts
	!   2: binned cuts with bin-size given by cuttype(2)
	!   3: exponential cuts: high-res near edges, low-res in middle
	! cutglob must point at the first position in junk for
	! this cut range. If we didn't parallelize, then cutglob
	! could just be initialized at 1, but when paralellizing over
	! detectors we need to know how long the junk portion for
	! each detector for each scan is. The easiest way to get
	! that is to just run pmat_cut for each cut range once
	! and for all, to establish the inital cutglob values.
	subroutine pmat_cut_range(dir, tod, junk, cuttype, ilen, olen)
		implicit none
		integer(4), intent(in)    :: cuttype(:), dir
		integer(4), intent(in),  optional :: ilen
		integer(4), intent(out), optional :: olen
		real(_),    intent(inout) :: junk(:), tod(:)
		integer(4) :: si, w, bi, si2, si3, n, ol, i
		real(_), allocatable :: x(:), Pa(:), Pb(:), Pc(:)
		n = size(tod)
		if(present(ilen)) n = ilen
		ol = 0
		select case(cuttype(1))
		case(0)
			! Ignore the cut samples. This is usually inconsistent. Use with care.
			continue
		case(1)
			! Full resolution cuts. All cut samples are stored.
			if(dir < 0) then
				junk = tod
			elseif(dir > 0) then
				tod = junk
			end if
			ol = n
		case(2)
			! Downgraded cuts. Cut area stored in bins of constant width.
			! Warning: Don't use this. It results in a step-function-like
			! TOD with lots of sharp edges. This gives lare spurious modes
			! in the solved map.
			w = cuttype(2)
			do bi = 1, (n-1)/w
				ol = ol+1
				if(dir < 0) then
					junk(ol) = sum(tod((bi-1)*w+1:bi*w))
				elseif(dir > 0) then
					tod((bi-1)*w+1:bi*w) = junk(ol)
				end if
			end do
			ol = ol+1
			if(dir < 0) then
				junk(ol) = sum(tod((bi-1)*w+1:n))
			elseif(dir < 0) then
				tod((bi-1)*w+1:n) = junk(ol)
			end if
		case(3)
			! Exponential cuts. Full resolution near edges, low in the middle,
			! with bin size doubling with distance to edge.
			! Warning: Don't use this. It results in a step-function-like
			! TOD with lots of sharp edges. This gives lare spurious modes
			! in the solved map.

			! Left edge
			w  = 1
			si = 1
			do while(si+w < (n+1)/2)
				ol = ol+1
				if(dir < 0) then
					junk(ol) = sum(tod(si:si+w-1))
				elseif(dir > 0) then
					tod(si:si+w-1) = junk(ol)
				end if
				si            = si+w
				w             = w*2
			end do
			! Right edge
			w   = 1
			si2 = 1
			do while(si2+w < (n+1)/2)
				ol  = ol+1
				si3 = n-si2+1
				if(dir < 0) then
					junk(ol) = sum(tod(si3-w+1:si3))
				elseif(dir > 0) then
					tod(si3-w+1:si3) = junk(ol)
				end if
				si2           = si2+w
				w             = w*2
			end do
			ol  = ol+1
			si3 = n-si2+1
			! Middle
			if(dir < 0) then
				junk(ol)   = sum(tod(si:si3))
			elseif(dir > 0) then
				tod(si:si3) = junk(ol)
			end if
		case(4)
			! Legendre polynomial projection
			w = min(n,4+n/cuttype(2))
			if(w <= 1) then
				if(dir > 0) then
					tod = junk(1)
				elseif(dir < 0) then
					junk(1) = sum(tod)
				end if
				ol = 1
			else
				if(dir > 0) tod = 0
				! This approach, with vectors for xv etc. Was several
				! times faster than the scalar version due to greater
				! parallelism.
				allocate(x(n),Pa(n),Pb(n),Pc(n))
				do si = 1, n
					x(si) = -1d0 + 2d0*(si-1)/(n-1)
				end do
				ol = 0
				do i = 0, w-1
					ol = ol + 1
					select case(i)
						case(0); Pa = 1
						case(1); Pb = 1; Pa = x
						case default; Pc = Pb; Pb = Pa; Pa = ((2*i+1)*x*Pb-i*Pc)/(i+1)
					end select
					if(dir < 0) then
						junk(ol) = sum(Pa*tod)
					elseif(dir > 0) then
						tod = tod + junk(ol) * Pa
					end if
				end do
			end if
		end select
		! These samples have been handled, so remove them so that
		! the map pmats do not use them again.
		if(dir < 0 .and. cuttype(1) .ne. 0) tod = 0
		if(present(olen)) olen = ol
	end subroutine

	pure function mean(a)
		implicit none
		real(_), intent(in)  :: a(:)
		real(_)              :: mean
		mean = sum(a)/size(a)
	end function

  !!! Azimuth binning stuff !!!
	subroutine pmat_scan(dir, tod, model, inds, det_comps, comps)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir, inds(:)
		real(_),    intent(inout) :: tod(:,:), model(:,:)
		real(_),    intent(in)    :: det_comps(:,:)
		integer(4), intent(in)    :: comps(:)
		! Work
		integer(4) :: ndet, nsamp, ncomp, npix, si, di, ci, nproc, id, i
		real(_), allocatable :: wmodel(:,:,:)

		! This is meant to be called together with pmat_tod, so it doesn't
		! overwrite the tod, but instead adds to it.
		nsamp = size(tod,1)
		ndet  = size(tod,2)
		npix  = size(model,1)
		ncomp = size(model,2)

		if(dir < 0) then
			nproc = omp_get_max_threads()
			allocate(wmodel(size(model,1),size(model,2),nproc))
			!$omp parallel workshare
			wmodel = 0
			!$omp end parallel workshare
			!$omp parallel private(di,si,ci,id)
			id = omp_get_thread_num()+1
			!$omp do
			do di = 1, ndet
				do ci = 1, ncomp
					do si = 1, nsamp
						wmodel(inds(si)+1,ci,id) = wmodel(inds(si)+1,ci,id) + tod(si,di)*det_comps(comps(ci)+1,di)
					end do
				end do
			end do
			!$omp end parallel
			!$omp parallel do collapse(2) private(ci,i)
			do ci = 1, ncomp
				do i = 1, npix
					model(i,ci) = sum(wmodel(i,ci,:))
				end do
			end do
		else
			!$omp parallel do private(di,si,ci)
			do di = 1, ndet
				do si = 1, nsamp
					do ci = 1, ncomp
						tod(si,di) = tod(si,di) + model(inds(si)+1,ci)*det_comps(comps(ci)+1,di)
					end do
				end do
			end do
		end if
	end subroutine

	subroutine pmat_map_rebin(dir, map_high, map_low)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		real(_),    intent(inout) :: map_high(:,:,:), map_low(:,:,:)
		! Work
		integer(4) :: hx,hy,lx,ly, n, step
		real(_)    :: val(size(map_low,3))

		step = nint(real(size(map_high,1))/size(map_low,1))

		if(dir > 0) then
			!$omp parallel do collapse(2) private(ly,lx,hy,hx,n,val)
			do ly = 0, size(map_low,2)-1
				do lx = 0, size(map_low,1)-1
					n = 0
					val = 0
					do hy = ly*step, min((ly+1)*step,size(map_high,2))-1
						do hx = lx*step, min((lx+1)*step,size(map_high,1))-1
							val = val + map_high(hx+1,hy+1,:)
							n   = n + 1
						end do
					end do
					map_low(lx+1,ly+1,:) = val/n
				end do
			end do
		else
			!$omp parallel do collapse(2) private(ly,lx,hy,hx)
			do ly = 0, size(map_low,2)-1
				do lx = 0, size(map_low,1)-1
					do hy = ly*step, min((ly+1)*step,size(map_high,2))-1
						do hx = lx*step, min((lx+1)*step,size(map_high,1))-1
							map_high(hx+1,hy+1,:) = map_low(lx+1,ly+1,:)
						end do
					end do
				end do
			end do
		end if
	end subroutine

	subroutine pmat_cut_rebin(dir, junk_high, cut_high, junk_low, cut_low)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir, cut_high(:,:), cut_low(:,:)
		real(_),    intent(inout) :: junk_high(:), junk_low(:)
		integer(4), parameter     :: det=1, lstart=2, llen=3, gstart=4, glen=5, cuttype=6
		integer(4) :: ci, gl1, gl2, gh1, gh2, nl, nh, i
		real(_), allocatable :: ibuf(:), obuf(:)
		! Assume that the high-res and low-res cuts have the same number of entries
		! and come in the same order.
		!$omp parallel do default(private) shared(junk_high,junk_low,cut_high,cut_low,dir)
		do ci = 1, size(cut_high,2)
			gl1 = cut_low (gstart,ci)+1; gl2 = gl1+cut_low (glen,ci)-1
			gh1 = cut_high(gstart,ci)+1; gh2 = gh1+cut_high(glen,ci)-1
			nl  = cut_low (llen,ci);     nh  = cut_high(llen,ci)
			allocate(ibuf(nh),obuf(nl))
			if(dir > 0) then
				call pmat_cut_range( 1, ibuf, junk_high(gh1:gh2), cut_high(cuttype:,ci))
				do i = 1, nl
					obuf(i) = mean(ibuf((i-1)*nh/nl+1:min(nh,i*nh/nl)))
				end do
				call pmat_cut_range(-1, obuf, junk_low (gl1:gl2), cut_low (cuttype:,ci))
			else
				call pmat_cut_range(+1, obuf, junk_low (gl1:gl2), cut_low (cuttype:,ci))
				do i = 1, nl
					ibuf((i-1)*nh/nl+1:min(nh,i*nh/nl)) = obuf(i)
				end do
				call pmat_cut_range(-1, ibuf, junk_high(gh1:gh2), cut_high(cuttype:,ci))
			end if
			deallocate(ibuf,obuf)
		end do
	end subroutine

	!!! Point source stuff !!!

	! Fast point source projection for a single source. Can't do OMP over
	! sources in this case. Current scheme can't easily OMP over dets,
	! as each (source,det,range) maps to a different set of samples, which
	! causes clobbering. I think this is hard to avoid in the range approach.
	!
	! Need to transpose the loop somehow:
	!  for det, for samp, for relevant src
	! But can't afford to compute distance from each source to each samp.
	! The prepare function uses a grid lookup, which is a good appraoch.
	! Precompute a grid that looks like:
	!  srclist=(ny,nx,nmax), srchits(ny,nx)
	! for det, samp
	!  y,x = interpol(det,samp)
	!  for src in srclist(y,x,1:srchits(ny,nx))
	!   etc.
	! This will have no TOD clobbering. The transpose operation involves
	! much fewer degrees of freedom, so we can use duplicate arrays as normal.
	!
	! How accurate do the distances and angles need to be?
	!  1. Euclidean pixels: Ignores cos(theta) variation inside image.
	!     Probably not good enough - expect 10% ellipticitty
	!  2. Semi-flat sky: r**2 = dtheta**2 + cos(theta_src)**2 * dphi**2
	!     This is what the current approach uses.
	!  3. Curved sky: r = acos(p_src*p_point), angles = something complicated
	!     Probably too expensive if implemented directly.
	!     But what about caching? For each source, precompute the transformation
	!     from az,el,t to source-centered coordinates. Then both r and angle
	!     are just a quick lookup away. If all interpolation arrays use the
	!     same resolution grid, then this simply amounts to having an extra
	!     ys_src(:,:,:,nsrc). HOWEVER, this approach requires a heavy
	!     recomputation step every time the source changes position. We can't
	!     afford that when fitting for the position.
	! So precompute hit grid, normal interpolation and interpolation.
	!
	! What if we wanted to support both intrinsic and beam ellipticity?
	! These are defined in different coordinate systems. But one gaussian
	! convoluted with another gaussian is still a gaussian with
	! cov_tot = cov_A + cov_B:  r'(A+B)"r. Chisquare not decomposable :/
	! We will restrict ourselves either intrinsic or beam ellipticity, not both.
	! It will be up to the user to disentangle these later. The user chooses
	! which coordinate system to use based on how he sets up ys_src.
	subroutine pmat_ptsrc2( &
			dir, tmul, pmul,           &! Projection direction, tod multiplier, src multiplier
			tod, srcs,                 &! Main inputs/outputs. tod(nsamp,ndet), srcs(nparam,ndir,nsrc)
			bore, det_pos, det_comps,  &
			rbox, nbox, ys_pos,        &! Coordinate transformation
			beam, rbeam, rmax,         &! Beam profile and max radial offset to consider
			cell_srcs, cell_nsrc, cbox &! Relevant source lookup. cell_srcs(:,nx,ny,ndir), cell_nsrc(nx,ny,ndir)
		)
		use omp_lib
		implicit none
		integer, intent(in)    :: dir
		real(_), intent(in)    :: tmul, pmul
		real(_), intent(inout) :: tod(:,:), srcs(:,:,:)
		real(_), intent(in)    :: bore(:,:), det_pos(:,:), det_comps(:,:)
		real(_), intent(in)    :: rbox(:,:), ys_pos(:,:,:), cbox(:,:)
		real(_), intent(in)    :: beam(:), rbeam, rmax
		integer, intent(in)    :: nbox(:), cell_srcs(:,:,:,:), cell_nsrc(:,:,:)
		! Work
		integer :: nsamp, ndet, nsrc, nproc
		integer :: ic, i, id, di, si, xind(3), ig, cell(2), cell_ind, cid, sdir, ndir
		integer :: steps(3), bind
		real(_) :: x0(3), inv_dx(3), c0(2), inv_dc(2), xrel(3)
		real(_) :: point(4), phase(3), dec, ra, ddec, dra, ibeam(3)
		real(_) :: inv_bres, bx,by,br,brel,bval, c2p,s2p,c1p,s1p
		real(_), parameter   :: pi = 3.14159265359d0
		real(_), allocatable :: amps(:,:,:,:), cosdec(:,:)
		integer, allocatable :: scandir(:)
		nsamp   = size(tod, 1)
		ndet    = size(tod, 2)
		ndir    = size(srcs,2)
		nsrc    = size(srcs,3)

		! Set up scanning direction. Two modes are supported. If ndir is 1, then
		! the same set of parameters are used for both left and rightgoing scans.
		! If ndir is 2, then these are separated.
		allocate(scandir(nsamp))
		if(ndir > 1) then
			scandir(1) = 1
			do si = 2, nsamp
				scandir(si) = merge(1,2,bore(2,si)>=bore(2,si-1))
			end do
		else
			scandir = 1
		end if

		! Precompute a few interpolation-relevant numbers
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))
		c0 = cbox(:,1)
		inv_dc(1) = size(cell_nsrc,2)/(cbox(1,2)-cbox(1,1))
		inv_dc(2) = size(cell_nsrc,1)/(cbox(2,2)-cbox(2,1))
		inv_bres = size(beam)/rbeam

		nproc = omp_get_max_threads()
		allocate(cosdec(ndir,nsrc),amps(3,ndir,nsrc,nproc))
		cosdec = cos(srcs(1,:,:))
		if(dir > 0) then
			do i = 1, nproc; amps(:,:,:,i) = srcs(3:5,:,:)*pmul; end do
		else
			amps = 0
		end if
		!$omp parallel private(id,di,si,xrel,xind,ig,point,phase,cell,cell_ind,cid,dec,ra,ibeam,ddec,dra,sdir,c2p,s2p,c1p,s1p,bx,by,br,brel,bind,bval)
		id = omp_get_thread_num()+1
		!$omp do
		do di = 1, ndet
			do si = 1, nsamp
				sdir = scandir(si)
				! Transform from hor to cel
				xrel = (bore(:,si)+det_pos(:,di)-x0)*inv_dx
				xind = floor(xrel)
				xrel = xrel - xind
				ig   = sum(xind*steps)+1
				point= ys_pos(:,1,ig) + xrel(1)*ys_pos(:,2,ig) + xrel(2)*ys_pos(:,3,ig) + xrel(3)*ys_pos(:,4,ig)
				! Find which point source lookup cell we are in.
				! dec,ra -> cy,cx
				cell = floor((point(1:2)-c0)*inv_dc)+1
				! Bounds checking. Costs 2% performance. Worth it
				cell(1) = min(size(cell_nsrc,2),max(1,cell(1)))
				cell(2) = min(size(cell_nsrc,1),max(1,cell(2)))
				if(dir > 0) tod(si,di) = tod(si,di)*tmul
				! Avoid expensive operations if we don't hit any sources
				if(cell_nsrc(cell(2),cell(1),sdir) == 0) cycle
				! The spin-2 and spin-1 rotations associated with the transformation
				! We need these to get the polarization rotation and beam orientation
				! right.
				c2p = point(3);                  s2p = point(4)
				c1p = sign(sqrt((1+c2p)/2),s2p); s1p = sqrt((1-c2p)/2)
				phase(1) = det_comps(1,di)
				phase(2) = c2p*det_comps(2,di) - s2p*det_comps(3,di)
				phase(3) = s2p*det_comps(2,di) + c2p*det_comps(3,di)
				! Process each point source in this cell
				do cell_ind = 1, cell_nsrc(cell(2),cell(1),sdir)
					cid = cell_srcs(cell_ind,cell(2),cell(1),sdir)+1
					dec   = srcs(1,sdir,cid)
					ra    = srcs(2,sdir,cid)
					ibeam = srcs(6:8,sdir,cid)
					! Calc effective distance from this source in terms of the beam distortions.
					! The beam shape is defined in the same coordinate system the polarization
					! orientation is defined in. We can either rotate the beam (like we do phase)
					! or rotate the offset vector the opposite direction. I choose the latter
					! because it is simpler.
					ddec = point(1)-dec
					dra  = (point(2)-ra)*cosdec(sdir,cid) ! Caller should beware angle wrapping!
					if(abs(ddec)>rmax .or. abs(dra)>rmax) cycle
					bx   =  c1p*dra + s1p*ddec
					by   = -s1p*dra + c1p*ddec
					br   = sqrt(by*(ibeam(1)*by+2*ibeam(3)*bx) + bx**2*ibeam(2))
					! Linearly interpolate the beam value
					brel = br*inv_bres+1
					bind = floor(brel)
					if(bind >= size(beam)-1) cycle
					brel = brel-bind
					bval = beam(bind)*(1-brel) + beam(bind+1)*brel
					!if(bval > 0.3) then
					!	write(*,'(a,i4,i8,i4,13f13.6,e15.7,f13.6)') 'A ', di, si, cid-1, bore(1,si), bore(2:3,si)*180/pi, (bore(2:3,si)+det_pos(2:3,di))*180/pi, det_pos(2:3,di)*180/pi, point(1:2)*180/pi, dec*180/pi, ra*180/pi, ddec*180*60/pi, dra*180*60/pi, tod(si,di), bval
					!end if
					! And perform the actual projection
					if(dir > 0) then
						tod(si,di) = tod(si,di) + sum(amps(:,sdir,cid,1)*phase)*bval
					else
						amps(:,sdir,cid,id) = amps(:,sdir,cid,id) + tod(si,di)*bval*phase
					end if
				end do
			end do
		end do
		!$omp end parallel
		if(dir < 0) then
			srcs(3:5,:,:) = srcs(3:5,:,:)*pmul + sum(amps,4)*tmul
		end if
	end subroutine

	! params(nparam,nsrc) takes the form [dec,ra,amps[namp],ibeam[3]]
	! The beam is defined in the target coordinate system. This is appropriate
	! if the beam is a physical property of each source (an elliptical galaxy
	! for example), but it is not appropriate if the beam is a property of the
	! telescope (as it usually is), unless the beam is circular.
	subroutine pmat_ptsrc( &
			tmul, pmul,                       &! Tod multiplier, src multiplier
			tod, params,                      &! Main inputs/outpus
			bore, det_pos, det_comps,  comps, &! Input pointing
			rbox, nbox, ys,                   &! Coordinate transformation
			ranges, rangesets, offsets        &! Precomputed relevant sample info
		)
		use omp_lib
		implicit none
		! Parameters
		real(_),    intent(in)    :: tmul, pmul    ! multiply tod by tmul, then add pmul times ptsrcs
		real(_),    intent(inout) :: tod(:,:)      ! (nsamp,ndet)
		real(_),    intent(in)    :: params(:,:)   ! (nparam,nsrc)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:)
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		integer(4), intent(in)    :: comps(:), nbox(:), offsets(:,:,:), ranges(:,:), rangesets(:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsrc, di, ri, oi, si, s0, i, j, i1, i2, ic, nj, namp, nsamp, nmin
		integer(4) :: steps(size(rbox,1))
		real(_)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1)), phase(size(comps),bz)
		real(_)    :: ra, dec, amps(size(comps)), ibeam(3), ddec, dra, r2, cosdec

		! NB: Ranges has units of nsamp*ndet, i.e. a range-value of R corresonds to
		! sample R%nsamp of detector R/nsamp.

		ndet  = size(tod,2)
		nsrc  = size(params,2)
		namp  = size(amps)
		nsamp = size(tod,1)
		nmin  = min(namp,size(comps))

		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		if(tmul .ne. 1) then
			!$omp parallel workshare
			tod = tod*tmul
			!$omp end parallel workshare
		end if

		! Note: don't add collapse(2) here without handling tod-clobbering
		!$omp parallel do private(si,di,oi,s0,ri,i1,i2,i,j,nj,ipoint,opoint,phase,ddec,dra,r2,dec,ra,amps,ibeam,cosdec)
		do si = 1, nsrc
			do di = 1, ndet
				dec   = params(1,si)
				ra    = params(2,si)
				amps  = params(3:2+namp,si)*pmul
				ibeam = params(3+namp:5+namp,si)
				cosdec= cos(dec)
				do oi = offsets(1,di,si)+1, offsets(2,di,si)
					ri = rangesets(oi)+1
					s0 = ranges(1,ri)/nsamp*nsamp
					i1=ranges(1,ri)-s0+1; i2 = ranges(2,ri)-s0
					do i = i1, i2, bz
						! Ok, we're finally at the relevant sample block. We now
						! need the physical coordinate of that sample.
						nj = min(i2+1-i,bz)
						do j = 1, nj
							ipoint(:,j) = bore(:,i+j-1)+det_pos(:,di)
						enddo
						opoint(:,:nj) = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
						phase(:,:nj)  = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
						do j = 1, nj
							! Compute shape-normalized distance from each sample to the current source.
							ddec = dec-opoint(1,j)
							dra  = (ra-opoint(2,j))*cosdec
							r2   = ddec*(ibeam(1)*ddec+ibeam(3)*dra) + dra*(ibeam(2)*dra+ibeam(3)*ddec)
							! And finally evaluate the model.
							tod(i+j-1,di) = tod(i+j-1,di) + sum(amps(1:nmin)*phase(1:nmin,j))*exp(-0.5*r2)
						end do
					end do
				end do
			end do
		end do
	end subroutine

	! Loops through all samples for all detectors for all sources, computing
	! the distance between each. Builds up a list of which samples are close
	! enough to each source to matter. Also builds up an inverse variance for
	! each source based on how much each detector hits it. This can be used to
	! divide sources into sets of sufficient sensitivity.
	!  pos(2,nsrc), rhit(nsrc), rmax(nsrc), det_ivars(nsrc), src_ivars(nsrc)
	!  ranges(2,maxrange,ndet,nsrc), nrange(ndet,nsrc)
	! ranges is zero-based and half-open, like python.
	! maxrange must be large enough to hold all the discovered ranges. Otherwise,
	! the last ranges will be lost. Nrange holds the actual number of discovered
	! ranges. ranges has units of det-local samples.
	! The distance measure used here is only approximate, suitable for small
	! distances on the sky. So for very large sources something else must be used.
	subroutine pmat_ptsrc_prepare( &
			pos, rhit, rmax, det_ivars, src_ivars, ranges, nrange, &
			bore, det_pos,  &! Input pointing
			rbox, nbox, ys  &! Coordinate transformation
		)
		use omp_lib
		implicit none
		! Parameters
		real(_),    intent(in)    :: pos(:,:), rhit(:), rmax(:), det_ivars(:)
		real(_),    intent(inout) :: src_ivars(:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:)
		real(_),    intent(in)    :: det_pos(:,:), rbox(:,:)
		integer(4), intent(inout) :: nrange(:,:), ranges(:,:,:,:)
		integer(4), intent(in)    :: nbox(:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4), allocatable :: grid(:,:,:), ngrid(:,:)
		integer(4) :: ndet, nsamp, nsrc, di, si, i, j, k, ic, nj, maxrange, gi(2)
		integer(4) :: steps(size(rbox,1)), glen(2), off(2), gsi
		real(_)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1))
		real(_)    :: ddec, dra, r2, rmaxmax
		real(_)    :: rmax2(size(pos,2)), rhit2(size(pos,2)), cosdec(size(pos,2))
		real(_)    :: g0(2), g1(2), igstep(2)
		integer(4) :: prev(size(det_pos,2),size(pos,2))

		nsamp = size(bore,2)
		ndet  = size(det_pos,2)
		nsrc  = size(pos,2)
		maxrange = size(ranges,2)

		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		! Set up our source pruning grid. The cell size must be at least rmax in
		! order to ensure that all relevant samples are within +-1 neighbor.
		! We also need an extra cell of padding at each side. Without this
		! approach, things were far too slow (minute instead of seconds).
		rmaxmax = maxval(rmax)
		g0 = minval(pos(1:2,:),2)-1.5*rmaxmax
		g1 = maxval(pos(1:2,:),2)+1.5*rmaxmax
		igstep = 1.0/rmaxmax
		glen = int((g1-g0)*igstep)+1
		allocate(ngrid(glen(1),glen(2)))
		ngrid  = 0
		do si = 1, nsrc
			gi = int((pos(1:2,si)-g0)*igstep)
			do i = -1,1; do j = -1,1;
				off(1)=i;off(2)=j
				if(any(gi+off<=0.or.gi+off>glen)) cycle
				ngrid(gi(1)+off(1),gi(2)+off(2)) = ngrid(gi(1)+off(1),gi(2)+off(2))+1
			end do; end do;
		end do
		! And then allocate and do it all over again, for real this time.
		allocate(grid(maxval(ngrid),glen(1),glen(2)))
		ngrid = 0
		do si = 1, nsrc
			gi = int((pos(1:2,si)-g0)*igstep)
			do i = -1,1; do j = -1,1;
				off(1)=i;off(2)=j
				if(any(gi+off<=0.or.gi+off>glen)) cycle
				ngrid(gi(1)+off(1),gi(2)+off(2)) = ngrid(gi(1)+off(1),gi(2)+off(2))+1
				grid(ngrid(gi(1)+off(1),gi(2)+off(2)),gi(1)+off(1),gi(2)+off(2)) = si
			end do; end do;
		end do

		src_ivars = 0
		prev   = -1
		cosdec = cos(pos(1,:))
		rmax2  = rmax**2
		rhit2  = rhit**2
		nrange = 0

		! For each source, find the samples that are close enough to
		! it to matter.
		!$omp parallel do private(di,i,nj,j,k,ipoint,opoint,gi,gsi,si,ddec,dra,r2) reduction(+:src_ivars)
		do di = 1, ndet
			do i = 1, nsamp, bz
				! Ok, we're finally at the relevant sample block. We now
				! need the physical coordinate of that sample.
				nj = min(nsamp+1-i,bz)
				do j = 1, nj
					ipoint(:,j) = bore(:,i+j-1)+det_pos(:,di)
				enddo
				opoint(:,:nj) = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
				do j = 1, nj
					! Find which grid point we're in
					gi = int((opoint(1:2,j)-g0)*igstep)
					if(all(gi>0 .and. gi <= glen)) then
						! Loop through all nearby sources.
						do gsi = 1, ngrid(gi(1),gi(2))
							si   = grid(gsi,gi(1),gi(2))
							ddec = pos(1,si)-opoint(1,j)
							dra  = (pos(2,si)-opoint(2,j))*cosdec(si)
							r2   = ddec**2+dra**2
							if(r2 < rmax2(si)) then
								! If the previous visited sample was not our last one, we must
								! have made a jump, so start a new range.
								k = i+j-1
								if(prev(di,si) .ne. k-1) then
									nrange(di,si) = nrange(di,si)+1
									ranges(1,min(maxrange,nrange(di,si)),di,si) = k-1
								end if
								ranges(2,min(maxrange,nrange(di,si)),di,si) = k
								prev(di,si) = k

								! Update source hits if close enough to fiducial position
								if(r2 < rhit2(si)) then
									src_ivars(si) = src_ivars(si) + det_ivars(di)
								end if
							end if
						end do
					end if
				end do
			end do
		end do
	end subroutine

	subroutine pmat_ptsrc_extract( &
			tod, out_tod, point, phase, oranges, &! Main inputs/outpus
			bore, det_pos, det_comps,  comps,    &! Input pointing
			rbox, nbox, ys,                      &! Coordinate transformation
			ranges, rangesets, offsets, raw      &! Precomputed relevant sample info
		)
		use omp_lib
		implicit none
		! Parameters
		real(_),    intent(in)    :: tod(:,:)      ! (nsamp,ndet)
		real(_),    intent(inout) :: out_tod(:), point(:,:), phase(:,:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:)
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		integer(4), intent(inout) :: oranges(:,:)
		integer(4), intent(in)    :: comps(:), nbox(:), offsets(:,:,:), rangesets(:), ranges(:,:), raw
		! Work
		real(_),    parameter :: pi = 3.14159265359d0
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsrc, di, ri, si, s0, oi, i, j, k, i1, i2, ic, nj, nsamp
		integer(4) :: r2det(size(ranges,2)), srcoff(size(ranges,2)+1)
		integer(4) :: steps(size(rbox,1))
		real(_)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(_)    :: x0(size(rbox,1)), inv_dx(size(rbox,1))

		ndet  = size(tod,2)
		nsrc  = size(offsets,3)
		nsamp = size(tod,1)

		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		! For each range we need to know which detector that range belongs to
		do si = 1, nsrc
			do di = 1, ndet
				do oi = offsets(1,di,si)+1, offsets(2,di,si)
					ri = rangesets(oi)+1
					r2det(ri) = di
				end do
			end do
		end do
		srcoff(1) = 0
		do ri = 1, size(ranges,2)
			srcoff(ri+1) = srcoff(ri) + ranges(2,ri)-ranges(1,ri)
		end do

		! Then loop through each range, extracting information as necessary
		!$omp parallel do private(ri,di,k,s0,i1,i2,i,nj,j,ipoint,opoint)
		do ri = 1, size(ranges,2)
			di = r2det(ri)
			k  = srcoff(ri)
			s0 = ranges(1,ri)/nsamp*nsamp
			i1=ranges(1,ri)-s0+1; i2 = ranges(2,ri)-s0
			oranges(1,ri) = k
			do i = i1, i2, bz
				! Ok, we're finally at the relevant sample block. We now
				! need the physical coordinate of that sample.
				nj = min(i2+1-i,bz)
				if(raw > 0) then
					point(:,k+1:k+nj) = bore(:,i:i+nj-1)
					do j = 1, nj
						phase(:,k+j) = det_comps(1:size(phase,1),di)
					end do
				else
					do j = 1, nj
						ipoint(:,j) = bore(:,i+j-1)+det_pos(:,di)
					enddo
					opoint(:,:nj)     = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
					phase(:,k+1:k+nj) = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
					point(:,k+1:k+nj) = opoint(1:2,:nj)
				end if
				!do j = 1, nj
				!	if(i+j-1 == 57168 .and. di == 497) then
				!		write(*,'(a,i4,2i8,7f13.6,e15.7)') "B ", di, i+j-1, k+j, bore(1,i+j-1), bore(2:3,i+j-1)*180/pi, point(2:3,k+j)*180/pi, det_pos(2:3,di)*180/pi, tod(i+j-1,di)
				!	end if
				!end do
				out_tod(k+1:k+nj) = tod(i:i+nj-1,di)
				k = k+nj
			end do
			oranges(2,ri) = k
		end do
	end subroutine


	subroutine pmat_az(dir, tod, map, az, dets, az0, daz)
		implicit none
		integer, intent(in)    :: dir, dets(:)
		real(_), intent(inout) :: tod(:,:), map(:,:)
		real(_), intent(in)    :: az0, daz, az(:)
		integer, allocatable   :: ais(:)
		integer :: di, si, ndet, nsamp, naz
		ndet  = size(tod,2)
		nsamp = size(tod,1)
		naz   = size(map,1)
		allocate(ais(nsamp))
		ais = min(int((az-az0)/daz)+1,naz)

		if(dir > 0) then
			!$omp parallel do private(di,si)
			do di = 1, ndet
				do si = 1, nsamp
					tod(si,di) = map(ais(si),dets(di)+1)
				end do
			end do
		elseif(dir < 0) then
			!$omp parallel do private(di,si)
			do di = 1, ndet
				do si = 1, nsamp
					map(ais(si),dets(di)+1) = map(ais(si),dets(di)+1) + tod(si,di)
				end do
			end do
		end if
	end subroutine

	subroutine pmat_phase(dir, tod, map, az, dets, az0, daz)
		implicit none
		integer, intent(in)    :: dir, dets(:)
		real(_), intent(inout) :: tod(:,:), map(:,:,:)
		real(_), intent(in)    :: az0, daz, az(:)
		integer, allocatable   :: ais(:), pis(:)
		integer :: di, si, ndet, nsamp, naz
		ndet  = size(tod,2)
		nsamp = size(tod,1)
		naz   = size(map,1)
		allocate(ais(nsamp),pis(nsamp))
		ais = min(int((az-az0)/daz)+1,naz)
		pis(1) = 1
		do si = 2, nsamp
			if(az(si) >= az(si-1)) then
				pis(si) = 1
			else
				pis(si) = 2
			end if
		end do

		if(dir > 0) then
			!$omp parallel do private(di,si)
			do di = 1, ndet
				do si = 1, nsamp
					tod(si,di) = tod(si,di) + map(ais(si),dets(di)+1,pis(si))
				end do
			end do
		elseif(dir < 0) then
			!$omp parallel do private(di,si)
			do di = 1, ndet
				do si = 1, nsamp
					map(ais(si),dets(di)+1,pis(si)) = map(ais(si),dets(di)+1,pis(si)) + tod(si,di)
				end do
			end do
		end if
	end subroutine

	subroutine pmat_plain(dir, map, tod, pix)
		use omp_lib
		implicit none
		integer, intent(in)    :: dir
		real(_), intent(in)    :: pix(:,:)
		real(_), intent(inout) :: map(:,:,:), tod(:,:)
		integer :: npix, ipix(size(pix,1)), i, j, k, nproc, id
		real(_), allocatable   :: wmap(:,:,:,:)

		nproc= omp_get_max_threads()
		npix = size(pix,2)
		if(dir > 0) then
			!$omp parallel do private(i,ipix)
			do i = 1, npix
				ipix = nint(pix(:,i))+1
				ipix(1) = max(1,min(size(map,2),ipix(1)))
				ipix(2) = max(1,min(size(map,1),ipix(2)))
				tod(i,:) = map(ipix(2),ipix(1),:)
			end do
		else
			allocate(wmap(size(map,1),size(map,2),size(map,3),nproc))
			!$omp parallel workshare
			wmap = 0
			!$omp end parallel workshare
			!$omp parallel private(i, ipix, id)
			id = omp_get_thread_num()+1
			!$omp do
			do i = 1, npix
				ipix = nint(pix(:,i))+1
				ipix(1) = max(1,min(size(map,2),ipix(1)))
				ipix(2) = max(1,min(size(map,1),ipix(2)))
				wmap(ipix(2),ipix(1),:,id) = tod(i,:)
			end do
			!$omp end do
			!$omp end parallel
			!$omp parallel do private(i,j)
			do i = 1, size(map, 3)
				do j = 1, size(map,2)
					do k = 1, size(map,1)
						map(k,j,i) = map(k,j,i) + sum(wmap(k,j,i,:))
					end do
				end do
			end do
		end if
	end subroutine

end module
