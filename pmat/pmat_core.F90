module pmat_core

	private map_block_prepare, map_block_finish
	private map_block_prepare_noatomic, map_block_finish_noatomic
	private map_block_prepare_shifted_flat, map_block_finish_shifted_flat
	private map_block_prepare_direct_flat, map_block_finish_direct_flat

contains

! Scinet benchmark results (OMP 16 on standard node):
!ip fast     d  1 tb 1.9105 t  0.889: 0.091 0.000  0.789 0.009 v 7.2496799e+09 1.182
!ip std_bi_0 d  1 tb 0.1321 t  3.353: 0.034 1.846  1.384 0.014 v 7.2498084e+09 0.000
!ip std_bi_1 d  1 tb 0.1325 t  4.104: 0.039 1.756  2.220 0.014 v 3.6996657e+09 0.000
!ip std_bi_3 d  1 tb 0.1322 t  5.563: 0.039 1.646  3.786 0.014 v 5.0880599e+09 0.000
!ip fast     d -1 tb 1.9821 t  1.321: 0.059 0.000  1.221 0.041 v 1.5651278e+13 1.174
!ip std_bi_0 d -1 tb 0.1270 t  4.717: 0.026 1.572  3.012 0.056 v 1.5650680e+13 0.000
!ip std_bi_1 d -1 tb 0.1265 t  7.240: 0.026 1.542  5.568 0.053 v 1.5856977e+13 0.000
!ip std_bi_3 d -1 tb 0.1328 t 18.186: 0.026 1.492 16.562 0.055 v 1.6075638e+13 0.000
!
!fast corresponds to sint below. std is faster than before because of the simplified
!phase optimization, but not as fast as sphase_bi because it doesn't use shifting.
!Shifting makes assumptions about the scanning pattern, and when those assumptions are
!valid, we can just as well go all the way to fast.

! Scinet benchmark results from testing phase (these are quite different on my dektop):
!std_bi    d  1 tb 0.1304 t 4.066: 0.000 0.040 2.201 1.736 0.015 v 7.2498084e+09 0.000
!shift_bi  d  1 tb 0.1230 t 3.493: 0.000 0.027 2.558 0.843 0.010 v 7.2498084e+09 0.000
!sbuf_bi   d  1 tb 0.1287 t 3.330: 0.000 0.027 2.484 0.750 0.010 v 7.2498084e+09 0.000
!sphase_bi d  1 tb 0.1290 t 2.591: 0.000 0.027 1.922 0.574 0.010 v 7.2498084e+09 0.000
!spoly     d  1 tb 0.1229 t 2.417: 0.000 0.031 1.438 0.883 0.010 v 7.2496794e+09 0.000
!sint      d  1 tb 0.1222 t 0.890: 0.000 0.027 0.000 0.800 0.009 v 7.2496794e+09 1.165
!std_bi    d -1 tb 0.1229 t 5.501: 0.000 0.026 2.183 3.192 0.049 v 1.5650679e+13 0.000
!shift_bi  d -1 tb 0.1232 t 3.971: 0.000 0.017 2.426 1.445 0.050 v 1.5650678e+13 0.000
!sbuf_bi   d -1 tb 0.1255 t 5.716: 0.000 0.237 2.618 1.017 1.343 v 1.5650677e+13 0.000
!sphase_bi d -1 tb 0.1232 t 4.456: 0.000 0.236 1.848 0.566 1.334 v 1.5647073e+13 0.000
!spoly     d -1 tb 0.1294 t 4.180: 0.000 0.243 1.236 0.877 1.346 v 1.5651279e+13 0.000
!sint      d -1 tb 0.1729 t 1.289: 0.000 0.017 0.000 1.197 0.041 v 1.5651279e+13 1.480
!
! Overall sint (shifted poly int-pixels with simplified phase) takes
! 1.3 (build) + 0.9 (forward) + 1.3 (backward) = 3.5 s, while old takes
! 4.1 (forward) + (5.5) (backward) = 9.6 s, so sint is 2.7 times faster.
! Just applying the shift gives 3.5 (forw) + 4.0 (back) = 7.5 s, which is
! a much smaller gain because the time is dominated by the pointing interpolation.
! Simplifying the phase appears always to be a win, and should be implemented as
! standard, I think. We don't need to support scaling of T or a different number
! than 3 components.
!
! The faster techniques I've implemented here are less
! general than the standard mapmaker, so I want to keep
! that too. So I don't think removing most of the options
! is a good choice either. But I could support just a few
! extreme cases. First of all: only support precompute for
! shifted integer-pixels, since these only need 4+4+4 bytes
! per det-sample (~2.8 GB) instead of (8+8+4+4) bytes (5.6 GB).
! (For double prec data, these would be 4+8+8 and 8+8+8+8,
! so a smaller relative difference).
! Should pixels be stored as 1d indices? Could be faster, as
! multiplications are precomputed, and could support longer scans
! as long as they aren't too wide. The full sky has less than
! 0.9 billion pixels, so a single 4-byte index can support the
! whole sky in the same space as two 2d pixels. They are much
! more opaque, though, and would prevent me from reusing the
! same projections.

! 1. Precomputed integer-pixel shifted polynomial. The fastest
!    case, and sufficient for normal mapmaking in normal coordinate
!    systems.
! 2. Direct unshifted grid. Slowest but general, and can support higher
!    order mapmaking because it isn't shifted.


	!!!! Direct unshifted grid !!!!

	subroutine pmat_map_direct_grid( &
		dir,                           &! Direction of direction: 1: forward (map2tod), -1: backward (tod2map)
		tod, tmul,                     &! The tod(nsamp,ndet)  and what to multiply it by
		map, mmul,                     &! The map(nx,ny,ncomp) and what to multiply it by
		pmet,                          &! Grid pointing interpol variant: 1: bilinear, 2:gradient
		mmet,                          &! Map projection method: 1: nearest, 2:bilinear, 3:bicubic
		bore, hwp, det_pos, det_comps, &! Input pointing
		rbox, nbox, yvals,             &! Interpolation grid
		wbox, nphi,                    &! wbox({y,x},{from,to}) pixbox and sky wrap in pixels
		times,                         &! Benchmark times for each step.
		split                          &! Which sub-tod split to use. Activated using mmet=4. Send in length-1 dummy otherwise
	)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, nbox(:), wbox(:,:), nphi, pmet, mmet, split(:)
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), yvals(:,:), det_pos(:,:), rbox(:,:)
		real(8),    intent(in)    :: det_comps(:,:)
		real(_),    intent(in)    :: tmul, mmul
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		real(8),    intent(inout) :: times(:)
		! Work
		real(8),    allocatable   :: pix(:,:)
		real(_),    allocatable   :: wmap(:,:,:), phase(:,:)
		integer(4), allocatable   :: xmap(:)
		integer(4) :: nsamp, ndet, di, steps(3)
		real(8)    :: x0(3), inv_dx(3), t1, t2, tloc1, tloc2, tpoint, tproj
		nsamp   = size(bore, 2)
		ndet    = size(det_comps, 2)
		t1 = omp_get_wtime()
		call interpol_prepare(nbox, rbox, steps, x0, inv_dx)
		t2 = omp_get_wtime()
		times(1) = times(1) + t2-t1
		call map_block_prepare(dir, wbox, nphi, mmul, map, wmap, xmap)
		t1 = omp_get_wtime()
		times(2) = times(1) + t1-t2
		tpoint = 0; tproj = 0 ! avoid ifort overeager optimization
		!$omp parallel do private(di, pix, phase, tloc1, tloc2) reduction(+:tpoint,tproj)
		do di = 1, ndet
			tloc1 = omp_get_wtime()
			allocate(pix(2,nsamp), phase(3,nsamp))
			call build_pointing_grid(pmet, bore, hwp, pix, phase, &
				det_pos(:,di), det_comps(:,di), steps, x0, inv_dx, yvals)
			call cap_pixels(pix, wbox)
			tloc2 = omp_get_wtime()
			tpoint = tpoint + tloc2-tloc1
			select case(mmet)
			! 0.0648 / 0.1714, tod2map slower due to atomic, but separate buffers
			! is even slower for nthread > 8
			case(0); call project_map_nearest (dir, tod(:,di), tmul, wmap, pix, phase, .true.)
			case(1); call project_map_bilinear(dir, tod(:,di), tmul, wmap, pix, phase, .true.)
			case(3); call project_map_bicubic (dir, tod(:,di), tmul, wmap, pix, phase, .true.)
			case(4); call project_map_split   (dir, tod(:,di), tmul, wmap, pix, phase, split, .true.)
			case(5)
				call project_map_nearest (dir, tod(:,di), tmul, wmap(3*split(di)+1:3*split(di)+3,:,:), pix, phase, .true.)
			end select
			deallocate(pix, phase)
			tloc1 = omp_get_wtime()
			tproj = tproj + tloc1-tloc2
		end do
		t2 = omp_get_wtime()
		times(3) = times(3) + (t2-t1)*tpoint/(tpoint+tproj)
		times(4) = times(4) + (t2-t1)*tproj /(tpoint+tproj)
		call map_block_finish(dir, wbox, mmul, map, wmap, xmap)
		t1 = omp_get_wtime()
		times(5) = times(5) + t1-t2
	end subroutine

	subroutine map_block_prepare(dir, wbox, nphi, mmul, map, wmap, xmap)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir, wbox(:,:), nphi
		real(_),    intent(in)    :: map(:,:,:), mmul
		real(_),    intent(inout), allocatable :: wmap(:,:,:)
		integer(4), intent(inout), allocatable :: xmap(:)
		integer(4) :: nwx, nwy, ix, iy, ox, oy, ic, pcut, ncomp
		! Set up our work map based on the relevant subset of pixels.
		nwy = wbox(1,2)-wbox(1,1)
		nwx = wbox(2,2)-wbox(2,1)
		ncomp = size(map,3)
		allocate(wmap(ncomp,nwx,nwy))
		! Set up the pixel wrap remapper
		allocate(xmap(nwx))
		pcut = -(nphi-size(map,1))/2
		do ix = 1, nwx
			ox = modulo(ix-1+wbox(2,1)-pcut,nphi)+pcut+1
			xmap(ix) = max(1,min(size(map,1),ox))
		end do
		!$omp parallel workshare
		wmap = 0
		!$omp end parallel workshare
		if (dir > 0) then
			! map2tod. Copy values over so we can add them to the tod later
			! 5% of total cost
			!$omp parallel do private(iy,ix,ic,oy)
			do iy = 1, nwy
				oy = max(1,min(size(map,2),iy+wbox(1,1)))
				do ic = 1, ncomp
					do ix = 1, nwx
						wmap(ic,ix,iy) = map(xmap(ix),oy,ic)*mmul
					end do
				end do
			end do
		end if
	end subroutine

	subroutine map_block_finish(dir, wbox, mmul, map, wmap, xmap)
		implicit none
		integer(4), intent(in)    :: dir, wbox(:,:)
		real(_),    intent(inout) :: map(:,:,:)
		real(_),    intent(in)    :: mmul
		real(_),    intent(inout), allocatable :: wmap(:,:,:)
		integer(4), intent(inout), allocatable :: xmap(:)
		integer(4) :: nwx, nwy, ix, iy, ox, ic, oy
		nwy = wbox(1,2)-wbox(1,1)
		nwx = wbox(2,2)-wbox(2,1)
		if (dir < 0) then
			if(mmul .ne. 1) then
				!$omp parallel workshare
				map = map*mmul
				!$omp end parallel workshare
			end if
			! tod2map, must copy out result from wmap. map is
			! usually bigger than wmap, so optimize loop for it
			!$omp parallel do private(iy,ix,ic,ox,oy)
			do iy = 1, nwy
				oy = max(1,min(size(map,2),iy+wbox(1,1)))
				do ic = 1, size(map,3)
					do ix = 1, nwx
						map(xmap(ix),oy,ic) = map(xmap(ix),oy,ic) + wmap(ic,ix,iy)
					end do
				end do
			end do
		end if
		deallocate(wmap, xmap)
	end subroutine

	! Same as the above, but without atomics
	subroutine pmat_map_direct_grid_noatomic( &
		dir,                           &! Direction of direction: 1: forward (map2tod), -1: backward (tod2map)
		tod, tmul,                     &! The tod(nsamp,ndet)  and what to multiply it by
		map, mmul,                     &! The map(nx,ny,ncomp) and what to multiply it by
		pmet,                          &! Grid pointing interpol variant: 1: bilinear, 2:gradient
		mmet,                          &! Map projection method: 1: nearest, 2:bilinear, 3:bicubic
		bore, hwp, det_pos, det_comps, &! Input pointing
		rbox, nbox, yvals,             &! Interpolation grid
		wbox, nphi,                    &! wbox({y,x},{from,to}) pixbox and sky wrap in pixels
		times,                         &! Benchmark times for each step.
		split                          &! Which sub-tod split to use. Activated using mmet=4. Send in length-1 dummy otherwise
	)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, nbox(:), wbox(:,:), nphi, pmet, mmet, split(:)
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), yvals(:,:), det_pos(:,:), rbox(:,:)
		real(8),    intent(in)    :: det_comps(:,:)
		real(_),    intent(in)    :: tmul, mmul
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		real(8),    intent(inout) :: times(:)
		! Work
		real(8),    allocatable   :: pix(:,:)
		real(_),    allocatable   :: wmap(:,:,:,:), phase(:,:)
		integer(4), allocatable   :: xmap(:)
		integer(4) :: nsamp, ndet, di, steps(3), id, myind
		real(8)    :: x0(3), inv_dx(3), t1, t2, tloc1, tloc2, tpoint, tproj
		nsamp   = size(bore, 2)
		ndet    = size(det_comps, 2)
		t1 = omp_get_wtime()
		call interpol_prepare(nbox, rbox, steps, x0, inv_dx)
		t2 = omp_get_wtime()
		times(1) = times(1) + t2-t1
		call map_block_prepare_noatomic(dir, wbox, nphi, mmul, map, wmap, xmap)
		t1 = omp_get_wtime()
		times(2) = times(1) + t1-t2
		tpoint = 0; tproj = 0 ! avoid ifort overeager optimization
		!$omp parallel do private(di, pix, phase, tloc1, tloc2, id, myind) reduction(+:tpoint,tproj)
		do di = 1, ndet
			tloc1 = omp_get_wtime()
			id    = omp_get_thread_num()
			myind = min(id+1, size(wmap,4))
			allocate(pix(2,nsamp), phase(3,nsamp))
			call build_pointing_grid(pmet, bore, hwp, pix, phase, &
				det_pos(:,di), det_comps(:,di), steps, x0, inv_dx, yvals)
			call cap_pixels(pix, wbox)
			tloc2 = omp_get_wtime()
			tpoint = tpoint + tloc2-tloc1
			select case(mmet)
			! 0.0648 / 0.1714, tod2map slower due to atomic, but separate buffers
			! is even slower for nthread > 8
			case(0); call project_map_nearest (dir, tod(:,di), tmul, wmap(:,:,:,myind), pix, phase, .false.)
			case(1); call project_map_bilinear(dir, tod(:,di), tmul, wmap(:,:,:,myind), pix, phase, .false.)
			case(3); call project_map_bicubic (dir, tod(:,di), tmul, wmap(:,:,:,myind), pix, phase, .false.)
			case(4); call project_map_split   (dir, tod(:,di), tmul, wmap(:,:,:,myind), pix, phase, split, .false.)
			case(5)
				call project_map_nearest (dir, tod(:,di), tmul, wmap(3*split(di)+1:3*split(di)+3,:,:,myind), pix, phase, .false.)
			end select
			deallocate(pix, phase)
			tloc1 = omp_get_wtime()
			tproj = tproj + tloc1-tloc2
		end do
		t2 = omp_get_wtime()
		times(3) = times(3) + (t2-t1)*tpoint/(tpoint+tproj)
		times(4) = times(4) + (t2-t1)*tproj /(tpoint+tproj)
		call map_block_finish_noatomic(dir, wbox, mmul, map, wmap, xmap)
		t1 = omp_get_wtime()
		times(5) = times(5) + t1-t2
	end subroutine

	subroutine map_block_prepare_noatomic(dir, wbox, nphi, mmul, map, wmap, xmap)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir, wbox(:,:), nphi
		real(_),    intent(in)    :: map(:,:,:), mmul
		real(_),    intent(inout), allocatable :: wmap(:,:,:,:)
		integer(4), intent(inout), allocatable :: xmap(:)
		integer(4) :: nwx, nwy, ix, iy, ox, oy, ic, pcut, ncomp, nbuf
		! Set up our work map based on the relevant subset of pixels.
		nwy = wbox(1,2)-wbox(1,1)
		nwx = wbox(2,2)-wbox(2,1)
		ncomp = size(map,3)
		if(dir < 0) then
			nbuf = omp_get_max_threads()
		else
			nbuf = 1
		end if
		allocate(wmap(ncomp,nwx,nwy,nbuf))
		! Set up the pixel wrap remapper
		allocate(xmap(nwx))
		pcut = -(nphi-size(map,1))/2
		do ix = 1, nwx
			ox = modulo(ix-1+wbox(2,1)-pcut,nphi)+pcut+1
			xmap(ix) = max(1,min(size(map,1),ox))
		end do
		!$omp parallel workshare
		wmap = 0
		!$omp end parallel workshare
		if (dir > 0) then
			! map2tod. Copy values over so we can add them to the tod later
			! 5% of total cost
			!$omp parallel do private(iy,ix,ic,oy)
			do iy = 1, nwy
				oy = max(1,min(size(map,2),iy+wbox(1,1)))
				do ic = 1, ncomp
					do ix = 1, nwx
						wmap(ic,ix,iy,1) = map(xmap(ix),oy,ic)*mmul
					end do
				end do
			end do
		end if
	end subroutine

	subroutine map_block_finish_noatomic(dir, wbox, mmul, map, wmap, xmap)
		implicit none
		integer(4), intent(in)    :: dir, wbox(:,:)
		real(_),    intent(inout) :: map(:,:,:)
		real(_),    intent(in)    :: mmul
		real(_),    intent(inout), allocatable :: wmap(:,:,:,:)
		integer(4), intent(inout), allocatable :: xmap(:)
		integer(4) :: nwx, nwy, ix, iy, ox, ic, oy
		real(_)    :: v
		nwy = wbox(1,2)-wbox(1,1)
		nwx = wbox(2,2)-wbox(2,1)
		if (dir < 0) then
			! Reduce buffers
			!$omp parallel do private(iy,ix,ic,v)
			do iy = 1, size(wmap,3)
				do ix = 1, size(wmap,2)
					do ic = 1, size(wmap,1)
						v = sum(wmap(ic,ix,iy,:))*mmul
						wmap(ic,ix,iy,1) = v
					end do
				end do
			end do
			! tod2map, must copy out result from wmap. map is
			! usually bigger than wmap, so optimize loop for it
			!$omp parallel do private(iy,ix,ic,ox,oy)
			do iy = 1, nwy
				oy = max(1,min(size(map,2),iy+wbox(1,1)))
				do ic = 1, size(map,3)
					do ix = 1, nwx
						map(xmap(ix),oy,ic) = map(xmap(ix),oy,ic) + wmap(ic,ix,iy,1)
					end do
				end do
			end do
		end if
		deallocate(wmap, xmap)
	end subroutine


	subroutine build_pointing_grid( &
		pmet, bore, hwp, pix, phase, &
		det_pos, det_comps, steps, x0, inv_dx, yvals)
		implicit none
		integer(4), intent(in)    :: steps(:), pmet
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), det_pos(:), det_comps(:), x0(:), inv_dx(:), yvals(:,:)
		real(8),    intent(inout) :: pix(:,:)
		real(_),    intent(inout) :: phase(:,:)
		integer(4) :: nsamp, xind(3), ig, si
		logical    :: use_hwp
		real(8)    :: xrel(3), point(4), work(4,4), tmp
		nsamp = size(bore,2)
		use_hwp = hwp(1,1) .ne. 0 .or. hwp(2,1) .ne. 0
		do si = 1, nsamp
			xrel = (bore(:,si)+det_pos(:)-x0)*inv_dx
			xind = floor(xrel)
			xrel = xrel - xind
			ig   = sum(xind*steps)+1
			! Manual expansion of bilinear interpolation. Pretty bad memory
			! access pattern, sadly. But despite the huge number of operations
			! compared to gradient interpolation, it's about the same speed.
			select case(pmet)
			case(1)
				! ops: about (2+4+2+4+4)*7 = 112
				work(:,1) = yvals(:,ig)*(1-xrel(1)) + yvals(:,ig+steps(1))*xrel(1)
				work(:,2) = yvals(:,ig+steps(2))*(1-xrel(1)) + yvals(:,ig+steps(2)+steps(1))*xrel(1)
				work(:,3) = yvals(:,ig+steps(3))*(1-xrel(1)) + yvals(:,ig+steps(3)+steps(1))*xrel(1)
				work(:,4) = yvals(:,ig+steps(2)+steps(3))*(1-xrel(1)) + yvals(:,ig+steps(2)+steps(3)+steps(1))*xrel(1)
				work(:,1) = work(:,1)*(1-xrel(2)) + work(:,2)*xrel(2)
				work(:,2) = work(:,3)*(1-xrel(2)) + work(:,4)*xrel(2)
				point = work(:,1)*(1-xrel(3)) + work(:,2)*xrel(3)
			case(2)
				point = yvals(:,ig) + &
					(yvals(:,ig+steps(1))-yvals(:,ig))*xrel(1) + &
					(yvals(:,ig+steps(2))-yvals(:,ig))*xrel(2) + &
					(yvals(:,ig+steps(3))-yvals(:,ig))*xrel(3)
			end select
			! Make 1-indexed
			pix(1:2,si) = point(1:2)+1
			phase(1:3,si) = det_comps(1:3)
			if(use_hwp) then
				tmp = phase(2,si)
				phase(2,si) = -hwp(1,si)*tmp + hwp(2,si)*phase(3,si)
				phase(3,si) = +hwp(2,si)*tmp + hwp(1,si)*phase(3,si)
			end if
			! Then the sky rotation
			tmp = phase(2,si)
			phase(2,si) = point(3)*tmp - point(4)*phase(3,si)
			phase(3,si) = point(4)*tmp + point(3)*phase(3,si)
		end do
	end subroutine

	! ops: about nsamp * 7
	subroutine project_map_nearest( &
		dir, tod, tmul, map, pix, phase, atomic)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		real(8),    intent(in)    :: pix(:,:)
		real(_),    intent(in)    :: tmul, phase(:,:)
		real(_),    intent(inout) :: tod(:), map(:,:,:)
		logical                   :: atomic
		! Work
		real(_)    :: v
		integer(4) :: nsamp, si, ci, p(2), nproc
		nsamp = size(tod)
		nproc = omp_get_num_threads()
		if(dir > 0) then
			! No clobber avoidance needed
			do si = 1, nsamp
				p = nint(pix(:,si))
				! Skip all out-of-bounds pixels. Accumulating them at the edge is useless
				if(p(1) .eq. 0) then
					tod(si) = tod(si)*tmul
					cycle
				end if
				if(tmul .eq. 0) then
					tod(si) = sum(map(1:3,p(2),p(1))*phase(1:3,si))
				else
					tod(si) = tod(si)*tmul + sum(map(1:3,p(2),p(1))*phase(1:3,si))
				end if
			end do
		else
			if(nproc > 1 .and. .not. atomic) then
				do si = 1, nsamp
					p = nint(pix(:,si))
					if(p(1) .eq. 0) cycle ! skip OOB pixels
					do ci = 1, 3
						v = (tod(si)*tmul)*phase(ci,si)
						!$omp atomic
						map(ci,p(2),p(1)) = map(ci,p(2),p(1)) + v
					end do
				end do
			else
				! Avoid slowing down single-proc case with atomics
				do si = 1, nsamp
					p = nint(pix(:,si))
					if(p(1) .eq. 0) cycle ! skip OOB pixels
					do ci = 1, 3
						v = (tod(si)*tmul)*phase(ci,si)
						map(ci,p(2),p(1)) = map(ci,p(2),p(1)) + v
					end do
				end do
			end if
		end if
	end subroutine

	! In bilinear interpolation the value of a pixel will
	! be an estimate of the value at the center of the pixel
	! rather than the average inside it (though perhaps putting
	! the degrees of freedom off by half a pixel would be more
	! stable). Coordinates inside one pixel go from -0.5 to 0.5.
	! If less than zero, we use
	!  v(-1)*(-dx) + v(0)*(1+dx)
	! otherwise it is
	!  v(+1)*( dx) + v(0)*(1-dx)
	! We can unify these by using floor instead of nint. Let p = floor(pix),
	! v0=map(p), v1=map(p+1), x = pix-p. Then the value is
	!  v0*(1-x) + v1*x
	! Sadly, our pixel truncation in the pixel calculation is not
	! enough to avoid OOB in this case. Must handle this ourselves.
	subroutine project_map_bilinear( &
		dir, tod, tmul, map, pix, phase, atomic)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		real(8),    intent(in)    :: pix(:,:)
		real(_),    intent(in)    :: tmul, phase(:,:)
		real(_),    intent(inout) :: tod(:), map(:,:,:)
		logical                   :: atomic
		real(8)    :: rpix(2)
		integer(4) :: p(2), ci
		! Work
		real(_)    :: x(2), v1(3,2), v2(3,2), v3(3,2), v4(3), v
		integer(4) :: nsamp, ncomp, si, nproc
		nsamp = size(tod)
		ncomp = size(map,1)
		nproc = omp_get_num_threads()
		do si = 1, nsamp
			! Stricter boundary conditions
			rpix(1) = max(1d0,min(size(map,3)-1d0,pix(1,si)))
			rpix(2) = max(1d0,min(size(map,2)-1d0,pix(2,si)))
			p = floor(rpix)
			x = rpix-p
			if(dir > 0) then
				! Interpolate along y direction
				v1 = map(:,p(2):p(2)+1,p(1))
				v2 = map(:,p(2):p(2)+1,p(1)+1)
				v3 = v1*(1-x(1)) + v2*x(1)
				! Interpolate along x direction
				v4 = v3(:,1)*(1-x(2)) + v3(:,2)*x(2)
				v  = sum(v4(1:3)*phase(1:3,si))
				! Update tod
				if(tmul .eq. 0) then
					tod(si) = v
				else
					tod(si) = tod(si)*tmul + v
				end if
			else
				! Transpose of the above
				v  = tod(si)*tmul
				v4(1:3) = v*phase(1:3,si)
				v3(:,1) = v4*(1-x(2))
				v3(:,2) = v4*x(2)
				v1 = v3*(1-x(1))
				v2 = v3*x(1)
				if(nproc > 1 .and. .not. atomic) then
					! I don't like using this many atomics. With four
					! times the number I usually have, this is probably
					! slower than separate work arrays.
					do ci = 1, 3
						!$omp atomic
						map(ci,p(2)  ,p(1)  ) = map(ci,p(2)  ,p(1)  ) + v1(ci,1)
						!$omp atomic
						map(ci,p(2)+1,p(1)  ) = map(ci,p(2)+1,p(1)  ) + v1(ci,2)
						!$omp atomic
						map(ci,p(2)  ,p(1)+1) = map(ci,p(2)  ,p(1)+1) + v2(ci,1)
						!$omp atomic
						map(ci,p(2)+1,p(1)+1) = map(ci,p(2)+1,p(1)+1) + v2(ci,2)
					end do
				else
					! Avoid slowing down single-proc case with atomics
					do ci = 1, 3
						map(ci,p(2)  ,p(1)  ) = map(ci,p(2)  ,p(1)  ) + v1(ci,1)
						map(ci,p(2)+1,p(1)  ) = map(ci,p(2)+1,p(1)  ) + v1(ci,2)
						map(ci,p(2)  ,p(1)+1) = map(ci,p(2)  ,p(1)+1) + v2(ci,1)
						map(ci,p(2)+1,p(1)+1) = map(ci,p(2)+1,p(1)+1) + v2(ci,2)
					end do
				end if
			end if
		end do
	end subroutine

	subroutine project_map_bicubic( &
		dir, tod, tmul, map, pix, phase, atomic)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		real(8),    intent(in)    :: pix(:,:)
		real(_),    intent(in)    :: tmul, phase(:,:)
		real(_),    intent(inout) :: tod(:), map(:,:,:)
		logical                   :: atomic
		real(8)    :: rpix(2)
		integer(4) :: p(2), ci, i, j, i2
		! Work
		real(_)    :: x, vy(3,4), vx(3), vtot, w(4,2)
		real(_)    :: vtmp
		integer(4) :: nsamp, ncomp, si, nproc
		nsamp = size(tod)
		ncomp = size(map,1)
		nproc = omp_get_num_threads()
		! FIXME: Gives negative absolute residual in cg. Something is wrong.
		do si = 1, nsamp
			! Stricter boundary conditions
			rpix(1) = max(2d0,min(size(map,3)-2d0,pix(1,si)))
			rpix(2) = max(2d0,min(size(map,2)-2d0,pix(2,si)))
			p = floor(rpix-1)
			! Compute weights in each direciton. This is based on
			! compute_weights in enlib.interpol.fortran
			do j = 1, 2
				do i = 1, 4
					x = abs(rpix(j)-(i-1)-p(j))
					if(x < 1) then
						w(i,j) =  1.5*x**3 - 2.5*x**2 + 1
					elseif(x < 2) then
						w(i,j) = -0.5*x**3 + 2.5*x**2 - 4*x + 2
					else
						w(i,j) = 0
					end if
				end do
			end do
			if(dir > 0) then
				! Interpolate in y direction
				vy = 0
				do i = 1, 4
					vy(:,:) = vy(:,:) + map(:,p(2):p(2)+3,p(1)+i-1)*w(i,1)
				end do
				! Interpolate in x direciton
				vx = 0
				do i = 1, 4
					vx = vx + vy(:,i)*w(i,2)
				end do
				vtot = sum(vx(1:3)*phase(1:3,si))
				! Update tod
				if(tmul .eq. 0) then
					tod(si) = vtot
				else
					tod(si) = tod(si)*tmul + vtot
				end if
			else
				! Transpose of the above
				vtot = tod(si)*tmul
				vx(1:3) = vtot*phase(1:3,si)
				do i = 1, 4
					vy(:,i) = vx*w(i,2)
				end do
				if(nproc > 1 .and. .not. atomic) then
					do i = 1, 4
						do i2 = 1, 4
							do ci = 1, 3
								vtmp = vy(ci,i2) * w(i,1)
								!$omp atomic
								map(ci,p(2)+i2-1,p(1)+i-1) = map(ci,p(2)+i2-1,p(1)+i-1) + vtmp
							end do
						end do
					end do
				else
					! Avoid slowing down single-proc case with atomics
					do i = 1, 4
						do i2 = 1, 4
							do ci = 1, 3
								vtmp = vy(ci,i2) * w(i,1)
								map(ci,p(2)+i2-1,p(1)+i-1) = map(ci,p(2)+i2-1,p(1)+i-1) + vtmp
							end do
						end do
					end do
				end if
			end if
		end do
	end subroutine

	subroutine project_map_split( &
		dir, tod, tmul, map, pix, phase, split, atomic)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, split(:)
		real(8),    intent(in)    :: pix(:,:)
		real(_),    intent(in)    :: tmul, phase(:,:)
		real(_),    intent(inout) :: tod(:), map(:,:,:)
		logical                   :: atomic
		! Work
		real(_)    :: v
		integer(4) :: nsamp, si, ci, p(2), nproc, coff, co
		nsamp = size(tod)
		nproc = omp_get_num_threads()
		if(dir > 0) then
			! No clobber avoidance needed
			do si = 1, nsamp
				p = nint(pix(:,si))
				! Skip all out-of-bounds pixels. Accumulating them at the edge is useless
				if(p(1) .eq. 0) then
					tod(si) = tod(si)*tmul
					cycle
				end if
				coff = 3*split(si)
				if(tmul .eq. 0) then
					tod(si) = sum(map(1+coff:3+coff,p(2),p(1))*phase(1:3,si))
				else
					tod(si) = tod(si)*tmul + sum(map(1+coff:3+coff,p(2),p(1))*phase(1:3,si))
				end if
			end do
		else
			if(nproc > 1 .and. .not. atomic) then
				do si = 1, nsamp
					p = nint(pix(:,si))
					if(p(1) .eq. 0) cycle ! skip OOB pixels
					do ci = 1, 3
						v  = (tod(si)*tmul)*phase(ci,si)
						co = ci+3*split(si)
						!$omp atomic
						map(co,p(2),p(1)) = map(co,p(2),p(1)) + v
					end do
				end do
			else
				! Avoid slowing down single-proc case with atomics
				do si = 1, nsamp
					p = nint(pix(:,si))
					if(p(1) .eq. 0) cycle ! skip OOB pixels
					do ci = 1, 3
						v  = (tod(si)*tmul)*phase(ci,si)
						co = ci+3*split(si)
						map(co,p(2),p(1)) = map(co,p(2),p(1)) + v
					end do
				end do
			end if
		end if
	end subroutine

	!!!! Precomputed integer-pixel shifted polynomial !!!!

	! We can improve memory efficiency by using a different internal pixelization.
	! We already do that with the pbox, which is a rectangular subset of the full
	! map. But we can do better by using a skewed system:
	!
	!      e       m
	!     d f     l n        ijklmn...
	!    c   g   k       :
	!   b     h j        :
	!  a       i             abcdefgh
	!
	! Define an ex(y,sdir), the expected x-location as a function of y and scanning
	! direction and make a new coordinate system which is
	!  ox = y - obox(1,1)
	!  oy = x - xshift(ox,sdir) - obox(2,1)
	!  oz = dir
	! xshift(ox,sdir) can for example be based on a simulated padded sweep of a detector
	! in the center of the focalplane. This will make that detector trace perfectly
	! straight lines in the new coordinate system, while others will be a bit curved,
	! but still mostly straight.
	!
	! What will be bounds of this system be? In the ox direction it will be given
	! by the size of the padded sweep we used. As long as that is properly padded
	! it will be large enough. In the oy direction it will have the same width as
	! the broadest width in ra of our tod on the sky. This should be
	!  woy = det_max(ra(t1))-det_min(ra(t0))
	!
	! So the inputs to the fortran part should be
	!  xshift(ox,sdir), obox({ox,oy},{from,to})
	! and the internal buffer will be (nox,noy,2) where (nox,noy) =
	! obox(:,2)-obox(:-1), all in fortran order. These will replace pbox.
	! sdir does not need to be passed in - it can be efficiently computed
	! on the fly each time, since is isn't per-detector.
	!
	! Copying between the map and our buffer:
	!  do sdir = 0, 1
	!   do ox = 1, nox
	!    iy = max(1,min(ny,ox + obox(1,1)))
	!    do oy = 1, noy
	!     ix = max(1,min(nx,oy + xshfit(ox,sdir+1) + obox(2,1)))
	!     map(ix,iy,:) = work(:,ox,oy) ! case copy-out
	!     work(:,oy,oy) = map(ix,iy,:) ! case copy-in
	!
	! wbox convention:
	!  wbox(1,1): global y of first work pixel
	!  wbox(2,1): global x of first work pixel
	!  wbox(1,2): global y of last  work pixel+1. Equal to wbox(1,1) + nwx
	!  wbox(2,2): wbox(2,1) + nwy
	! Because the work system is shifted and tranposed compared to the
	! base system, wbox is {y,x} but {wx,wy}-ordered.

	subroutine pmat_map_get_pix_poly_shift( &
			pix,  phase,               &! Main inputs/outpus
			bore, hwp, det_comps,      &! Input pointing
			coeffs,                    &! Coordinate interpolation
			sdir, wbox, wshift         &! Pixel remapping
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: wbox(:,:), wshift(:,:), sdir(:)
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), coeffs(:,:,:)
		real(8),    intent(in)    :: det_comps(:,:)
		integer(4), intent(inout) :: pix(:,:)
		real(_),    intent(inout) :: phase(:,:,:)
		integer(4) :: ndet, di
		ndet    = size(det_comps, 2)
		!$omp parallel do
		do di = 1, ndet
			call build_pointing_int_poly_shift(bore, hwp, pix(:,di), phase(:,:,di), det_comps(:,di), coeffs(:,:,di), sdir, wbox, wshift)
		end do
	end subroutine

	subroutine pmat_map_use_pix_shift( &
			dir,                       &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
			tod, tmul,                 &! tod and what to mul it by
			map, mmul,                 &! map and what to mul it by
			pix, phase,                &! Precomputed pointing
			wbox, wshift, nphi,        &! Pixel remapping
			times                      &! Time report output array (5)
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, wbox(:,:), wshift(:,:), nphi
		real(_),    intent(in)    :: tmul, mmul, phase(:,:,:)
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		integer(4), intent(in)    :: pix(:,:)
		real(8),    intent(inout) :: times(:)
		! Work
		real(_),    allocatable   :: wmap(:,:)
		integer(4) :: ndet, di
		real(8)    :: t1, t2
		ndet    = size(tod, 2)
		t1 = omp_get_wtime()
		call map_block_prepare_shifted_flat(dir, wbox, wshift, nphi, mmul, map, wmap)
		t2 = omp_get_wtime()
		times(2) = times(2) + t2-t1
		!$omp parallel do private(di)
		do di = 1, ndet
			call project_map_nearest_int_flat (dir, tod(:,di), tmul, wmap, pix(:,di), phase(:,:,di))
		end do
		t1 = omp_get_wtime()
		times(4) = times(4) + t1-t2
		call map_block_finish_shifted_flat(dir, wbox, wshift, nphi, mmul, map, wmap)
		t2 = omp_get_wtime()
		times(5) = times(5) + t2-t1
	end subroutine

	subroutine project_map_nearest_int_flat( &
		dir, tod, tmul, map, pix, phase)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		integer(4), intent(in)    :: pix(:)
		real(_),    intent(in)    :: tmul, phase(:,:)
		real(_),    intent(inout) :: tod(:), map(:,:)
		integer    :: p, ci, nsamp, si, nproc
		real(_)    :: v
		nsamp = size(tod)
		nproc = omp_get_num_threads()
		if(dir > 0) then
			do si = 1, nsamp
				p = pix(si)
				if(tmul .eq. 0) then
					tod(si) = + sum(map(1:3,p)*phase(1:3,si))
				else
					tod(si) = tod(si)*tmul + map(1,p) + sum(map(1:3,p)*phase(1:3,si))
				end if
			end do
		else
			if(nproc > 1) then
				do si = 1, nsamp
					p = pix(si)
					v = tod(si)*tmul
					do ci = 1, 3
						v = (tod(si)*tmul)*phase(ci,si)
						!$omp atomic
						map(ci,p) = map(ci,p) + v
					end do
				end do
			else
				! Avoid slowing down single-proc case with atomics
				do si = 1, nsamp
					p = pix(si)
					do ci = 1, 3
						v = (tod(si)*tmul)*phase(ci,si)
						map(ci,p) = map(ci,p) + v
					end do
				end do
			end if
		end if
	end subroutine

	subroutine build_pointing_int_poly_shift(bore, hwp, pix, phase, det_comps, coeff, sdir, wbox, wshift)
		implicit none
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), det_comps(:), coeff(:,:)
		integer(4), intent(inout) :: pix(:)
		real(_),    intent(inout) :: phase(:,:)
		integer(4), intent(in)    :: wbox(:,:), wshift(:,:), sdir(:)
		real(_)    :: tmp
		real(8)    :: work(4), wx, wy, p(2)
		integer(4) :: nsamp, si, nwx, nwy, iwx
		logical    :: use_hwp
		real(8)    :: az, t
		nsamp = size(bore,2)
		nwy = wbox(2,2)-wbox(2,1)
		nwx = wbox(1,2)-wbox(1,1)
		use_hwp = hwp(1,1) .ne. 0 .or. hwp(2,1) .ne. 0
		do si = 1, nsamp
			t  = bore(1,si)
			az = bore(2,si)
			work = coeff(1,:) + coeff(9,:)*t + az*(coeff(2,:) + coeff(10,:)*t + &
				az*(coeff(3,:) + coeff(11,:)*t + az*(coeff(4,:) + az*(coeff(5,:) + &
				az*(coeff(6,:) + az*(coeff(7,:) + az*coeff(8,:)))))))
			! +1 to make 1-indexed
			p   = work(1:2) + 1
			wx  = p(1) - wbox(1,1) ! work x = pix y - box corner y
			iwx = nint(wx)
			! work y = pix x - box corner x - shift + dir factor
			wy  = p(2) - wbox(2,1) - wshift(iwx,sdir(si)+1) + sdir(si)*nwy
			!if(wy-sdir(si)*nwy >= nwy .or. wx >= nwx) then
			!	write(*,*) "t",t,"az",az,"p",p
			!	write(*,*) "wy-sdir", wy-sdir(si)*nwy, "wx", wx
			!	write(*,*) "wbox y", wbox(2,1:2)
			!	write(*,*) "wbox x", wbox(1,1:2)
			!	write(*,*) "wshift 1", wshift(1:size(wshift,1),1)
			!	write(*,*) "wshift 2", wshift(1:size(wshift,1),2)
			!	stop
			!end if
			pix(si) = (nint(wy)-1)*nwx+nint(wx)
			! Make 1-indexed
			phase(1:3,si) = det_comps(1:3)
			if(use_hwp) then
				tmp = phase(2,si)
				phase(2,si) = -hwp(1,si)*tmp + hwp(2,si)*phase(3,si)
				phase(3,si) = +hwp(2,si)*tmp + hwp(1,si)*phase(3,si)
			end if
			! Then the sky rotation
			tmp = phase(2,si)
			phase(2,si) = work(3)*tmp - work(4)*phase(3,si)
			phase(2,si) = work(4)*tmp + work(3)*phase(3,si)
		end do
	end subroutine

	subroutine map_block_prepare_shifted_flat(dir, wbox, wshift, nphi, mmul, map, wmap)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir, wbox(:,:), wshift(:,:), nphi
		real(_),    intent(in)    :: map(:,:,:), mmul
		real(_),    intent(inout), allocatable :: wmap(:,:)
		integer(4) :: ix, iy, iwx, iwy, iwy_sdir, ic, nx, ny, nwx, nwy, pcut, sdir
		! Set up our work map based on the relevant subset of pixels.
		nx   = size(map,1)
		ny   = size(map,2)
		nwx  = wbox(1,2)-wbox(1,1) ! {wx,wy} ordering
		nwy  = wbox(2,2)-wbox(2,1)
		pcut = -(nphi-nx)/2
		! It would be most natural to have wmap(ncomp,nwx,nwy,sdir),
		! but then it wouldn't be compatible with our binning functions,
		! which expect a 3d map. Instead, we can unroll it such that
		! the sdir == 1 case follows after the sdir == 0 case in memory.
		! So the real size would be 2*nwy. From the point of view of the
		! python code, this will be an implementation detail.
		allocate(wmap(3,nwx*2*nwy))
		!$omp parallel workshare
		wmap = 0
		!$omp end parallel workshare
		if (dir > 0) then
			! map2tod. Copy values over so we can add them to the tod later
			! 5% of total cost
			do sdir = 0, 1
				!$omp parallel do private(iy,ix,ic,iwx,iwy,iwy_sdir)
				do iwx = 1, nwx
					iy = max(1,min(ny, iwx+wbox(1,1)))
					do iwy = 1, nwy
						iwy_sdir = iwy + sdir*nwy
						ix = iwy+wbox(2,1)+wshift(iwx,sdir+1)
						ix = modulo(ix-1-pcut,nphi)+pcut+1
						ix = max(1,min(nx, ix))
						do ic = 1, size(map,3)
							wmap(ic,iwx+(iwy_sdir-1)*nwx) = map(ix,iy,ic)*mmul
						end do
					end do
				end do
			end do
		end if
	end subroutine

	subroutine map_block_finish_shifted_flat(dir, wbox, wshift, nphi, mmul, map, wmap)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir, wbox(:,:), wshift(:,:), nphi
		real(_),    intent(in)    :: mmul
		real(_),    intent(inout) :: map(:,:,:)
		real(_),    intent(inout), allocatable :: wmap(:,:)
		integer(4) :: ix, iy, iwx, iwy, iwy_sdir, ic, nx, ny, nwx, nwy, pcut, sdir
		! Set up our work map based on the relevant subset of pixels.
		nx   = size(map,1)
		ny   = size(map,2)
		nwx  = wbox(1,2)-wbox(1,1) ! {wx,wy} order
		nwy  = wbox(2,2)-wbox(2,1)
		pcut = -(nphi-nx)/2
		if (dir < 0) then
			! map2tod. Copy values over so we can add them to the tod later
			! 5% of total cost
			do sdir = 0, 1
				!$omp parallel do private(iy,ix,ic,iwx,iwy,iwy_sdir)
				do iwx = 1, nwx
					iy = max(1,min(ny, iwx+wbox(1,1)))
					do iwy = 1, nwy
						iwy_sdir = iwy + sdir*nwy
						ix = iwy+wbox(2,1)+wshift(iwx,sdir+1)
						ix = modulo(ix-1-pcut,nphi)+pcut+1
						ix = max(1,min(nx, ix))
						do ic = 1, size(map,3)
							map(ix,iy,ic) = map(ix,iy,ic)*mmul + wmap(ic,iwx+(iwy_sdir-1)*nwx)
						end do
					end do
				end do
			end do
		end if
		deallocate(wmap)
	end subroutine

	subroutine interpol_prepare(nbox, rbox, steps, x0, inv_dx)
		implicit none
		integer(4), intent(in)    :: nbox(:)
		real(8),    intent(in)    :: rbox(:,:)
		integer(4), intent(inout) :: steps(:)
		real(8),    intent(inout) :: x0(:), inv_dx(:)
		integer(4) :: ic
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = (nbox-1)/(rbox(:,2)-rbox(:,1))
	end subroutine

	subroutine cap_pixels(pix, pbox)
		implicit none
		integer(4), intent(in)    :: pbox(:,:)
		real(8),    intent(inout) :: pix(:,:)
		real(8)    :: psize(2), moo(2)
		integer(4) :: si
		psize = pbox(:,2)-pbox(:,1)
		!$!omp simd
		do si = 1, size(pix,2)
			pix(:,si) = pix(:,si) - pbox(:,1)
			! Out of bounds pixels are indicated with a 0 value. They
			! will be ignored in the projection code.
			if(any(pix(1:2,si) < 0.5d0) .or. any(pix(1:2,si) >= psize + 0.5d0)) pix(1:2,si) = 0
			!! We will round this later. The numbers ensure that we will
			!! still be in bounds after rounding.
			!pix(1,si) = min(psize(1)+0.49999d0,max(0.5d0,pix(1,si)))
			!pix(2,si) = min(psize(2)+0.49999d0,max(0.5d0,pix(2,si)))
		end do
	end subroutine

	!!! Workspace projection. Only the pix computation is new. The rest is shared. !!!

	subroutine pmat_map_use_pix_direct( &
			dir,                       &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
			tod, tmul,                 &! tod and what to mul it by
			map, mmul,                 &! map and what to mul it by
			pix, phase,                &! Precomputed pointing
			times                      &! Time report output array (5)
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		real(_),    intent(in)    :: tmul, mmul, phase(:,:,:)
		real(_),    intent(inout) :: tod(:,:), map(:,:,:)
		integer(4), intent(in)    :: pix(:,:)
		real(8),    intent(inout) :: times(:)
		! Work
		real(_),    allocatable   :: wmap(:,:)
		integer(4) :: ndet, di
		real(8)    :: t1, t2
		ndet    = size(tod, 2)
		t1 = omp_get_wtime()
		call map_block_prepare_direct_flat(dir, mmul, map, wmap)
		t2 = omp_get_wtime()
		times(2) = times(2) + t2-t1
		!$omp parallel do private(di)
		do di = 1, ndet
			call project_map_nearest_int_flat (dir, tod(:,di), tmul, wmap, pix(:,di), phase(:,:,di))
		end do
		t1 = omp_get_wtime()
		times(4) = times(4) + t1-t2
		call map_block_finish_direct_flat(dir, mmul, map, wmap)
		t2 = omp_get_wtime()
		times(5) = times(5) + t2-t1
	end subroutine

	subroutine pmat_map_get_pix_poly_shift_xy( &
			pix,  phase,               &! Main inputs/outpus
			bore, hwp, det_comps,      &! Input pointing
			coeffs,                    &! Coordinate interpolation
			sdir, y0, nwx, nwys, xshift, yshift, nphi &! Pixel remapping
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: y0, nwx, nwys(:), xshift(:,:), yshift(:,:), sdir(:), nphi
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), coeffs(:,:,:)
		real(8),    intent(in)    :: det_comps(:,:)
		integer(4), intent(inout) :: pix(:,:)
		real(_),    intent(inout) :: phase(:,:,:)
		integer(4) :: ndet, di
		ndet    = size(det_comps, 2)
		!$omp parallel do
		do di = 1, ndet
			call build_pointing_int_poly_shift_xy(bore, hwp, pix(:,di), phase(:,:,di), &
				det_comps(:,di), coeffs(:,:,di), sdir, y0, nwx, nwys, xshift, yshift, nphi)
		end do
	end subroutine

	subroutine build_pointing_int_poly_shift_xy(bore, hwp, pix, phase, det_comps, coeff, sdir, y0, nwx, nwys, xshift, yshift, nphi)
		implicit none
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), det_comps(:), coeff(:,:)
		integer(4), intent(inout) :: pix(:)
		real(_),    intent(inout) :: phase(:,:)
		integer(4), intent(in)    :: nwx, nwys(:), y0, xshift(:,:), yshift(:,:), sdir(:), nphi
		real(_)    :: tmp
		real(8)    :: work(4), p(2)
		integer(4) :: nsamp, si, iwx, iy, d, nsub, wx, wy, wytot
		logical    :: use_hwp
		real(8)    :: az, t
		nsamp = size(bore,2)
		use_hwp = hwp(1,1) .ne. 0 .or. hwp(2,1) .ne. 0
		do si = 1, nsamp
			!write(*,*) "-----------"
			t  = bore(1,si)
			az = bore(2,si)
			!write(*,*) "si", si, "t", t, "az", az*180/3.14159, "el", bore(3,si)*180/3.14159
			work = coeff(1,:) + coeff(9,:)*t + az*(coeff(2,:) + coeff(10,:)*t + &
				az*(coeff(3,:) + coeff(11,:)*t + az*(coeff(4,:) + az*(coeff(5,:) + &
				az*(coeff(6,:) + az*(coeff(7,:) + az*coeff(8,:)))))))
			p  = work(1:2)
			!write(*,*) "p", p
			d  = sdir(si)+1
			! p is the pixel index into the full map. We now want to transform
			! it onto the local workspace. The shape of the workspace is given
			! by nwys(2) and nwx.
			! Up to pixel rounding, capping and dir offsets, we have
			! wy = x - xshift[y-y0], wx = yshift[y-y0]
			iy = min(max(nint(p(1))-y0+1,1),size(xshift,1))
			!write(*,*) "y0 iy ny", y0, iy, size(xshift,1)
			wy = modulo(nint(p(2)) - xshift(iy,d),nphi) + 1
			!write(*,*) "sdir", d, "xshift", xshift(iy,d)
			!write(*,*) "wy", wy, "nwys", nwys
			! The x stretching is a bit harder
			if(iy == size(yshift,1)) then
				nsub = 1
			else
				nsub = yshift(iy+1,d)-yshift(iy,d)
			end if
			wx = yshift(iy,d) + floor((p(1)-nint(p(1))+0.5d0)*nsub) + 1
			!write(*,*) "nsub", nsub, "rel", p(1)-nint(p(1))+0.5, "off", floor((p(1)-nint(p(1))+0.5d0)*nsub)
			!write(*,*) "wx", wx, nwx
			! Cap to bounds of workspace
			wx = max(1,min(nwx, wx))
			wy = max(1,min(nwys(d),wy))
			wytot = wy + nwys(1)*sdir(si)
			!write(*,*) "wxc", wx, "wyc", wy, "wytot", wytot
			! And flatten to 1d
			pix(si) = (wytot-1)*nwx+wx
			! Make 1-indexed
			phase(1:3,si) = det_comps(1:3)
			if(use_hwp) then
				tmp = phase(2,si)
				phase(2,si) = -hwp(1,si)*tmp + hwp(2,si)*phase(3,si)
				phase(3,si) = +hwp(2,si)*tmp + hwp(1,si)*phase(3,si)
			end if
			! Then the sky rotation
			tmp = phase(2,si)
			phase(2,si) = work(3)*tmp - work(4)*phase(3,si)
			phase(3,si) = work(4)*tmp + work(3)*phase(3,si)
		end do
	end subroutine

	subroutine map_block_prepare_direct_flat(dir, mmul, map, wmap)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir
		real(_),    intent(in)    :: map(:,:,:), mmul
		real(_),    intent(inout), allocatable :: wmap(:,:)
		integer(4) :: ix, iy, ic, nx, ny
		! Set up our work map based on the relevant subset of pixels.
		nx   = size(map,1)
		ny   = size(map,2)
		allocate(wmap(3,nx*ny))
		!$omp parallel workshare
		wmap = 0
		!$omp end parallel workshare
		if (dir > 0) then
			!$omp parallel do private(iy,ix,ic)
			do ix = 1, nx
				do iy = 1, ny
					do ic = 1, size(map,3)
						wmap(ic,(iy-1)*nx+ix) = map(ix,iy,ic)*mmul
					end do
				end do
			end do
		end if
	end subroutine

	subroutine map_block_finish_direct_flat(dir, mmul, map, wmap)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir
		real(_),    intent(in)    :: mmul
		real(_),    intent(inout) :: map(:,:,:)
		real(_),    intent(inout), allocatable :: wmap(:,:)
		integer(4) :: ix, iy, ic, nx, ny
		! Set up our work map based on the relevant subset of pixels.
		nx   = size(map,1)
		ny   = size(map,2)
		if (dir < 0) then
			!$omp parallel do private(iy,ix,ic)
			do ix = 1, nx
				do iy = 1, ny
					do ic = 1, size(map,3)
						map(ix,iy,ic) = map(ix,iy,ic)*mmul + wmap(ic,ix+(iy-1)*nx)
					end do
				end do
			end do
		end if
		deallocate(wmap)
	end subroutine

	subroutine bincount_flat(hits, flat_pix, pshape, axis)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: flat_pix(:,:), pshape(:), axis
		integer(4), intent(inout) :: hits(:,:)
		integer(4) :: nsamp, ndet, si, di, pdiv, pmod, ndim, p, ax
		nsamp= size(flat_pix, 1)
		ndet = size(flat_pix, 2)
		ndim = size(pshape)
		ax   = modulo(axis, ndim)+1
		pdiv = product(pshape(ax+1:ndim))
		pmod = pshape(ax)
		!$omp parallel workshare
		hits = 0
		!$omp end parallel workshare
		!$omp parallel do private(si, p)
		do di = 1, ndet
			do si = 1, nsamp
				p = modulo((flat_pix(si,di)-1)/pdiv,pmod)+1
				hits(p,di) = hits(p,di) + 1
			end do
		end do
	end subroutine

	!!! Workspace to map !!!

	subroutine pmat_workspace(dir, work, map, y0, nwx, nwys, xshift, yshift, nphi)
		implicit none
		integer(4), intent(in)    :: dir, y0, nwx, nwys(2), xshift(:,:), yshift(:,:), nphi
		real(_),    intent(inout) :: work(:,:,:), map(:,:,:)
		integer(4) :: y, x, iy, wy, wytot, wx, d, ny, dx, nsub, i
		ny = size(xshift,1)
		! Loop through each output pixel in the map. iy and ix are the y and x pixel relative
		! to the bottom-left corner of the exposed region. Looping this way avoids the need
		! for any locking.
		do d = 1, 2
			!$omp parallel do private(iy,y,wx,x,nsub,i)
			do iy = 1, ny
				y = iy + y0
				do wy = 1, nwys(d)
					wytot = wy + nwys(1)*(d-1)
					x = modulo(xshift(iy,d) + wy - 1, nphi)+1
					nsub = 1
					if(iy < ny) nsub = yshift(iy+1,d)-yshift(iy,d)
					do i = 1, nsub
						wx = yshift(iy,d) + i
						if(dir > 0) then
							work(wx,wytot,:) = map(x,y,:)
						else
							map(x,y,:) = map(x,y,:) + work(wx,wytot,:)
						end if
					end do
				end do
			end do
		end do
	end subroutine

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
			elseif(dir == 1) then
				tod = junk
			elseif(dir >= 2) then
				tod = tod + junk
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
				elseif(dir == 1) then
					tod((bi-1)*w+1:bi*w) = junk(ol)
				elseif(dir >= 2) then
					tod((bi-1)*w+1:bi*w) = tod((bi-1)*w+1:bi*w) + junk(ol)
				end if
			end do
			ol = ol+1
			if(dir < 0) then
				junk(ol) = sum(tod((bi-1)*w+1:n))
			elseif(dir == 1) then
				tod((bi-1)*w+1:n) = junk(ol)
			elseif(dir >= 2) then
				tod((bi-1)*w+1:n) = tod((bi-1)*w+1:n) + junk(ol)
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
				elseif(dir == 1) then
					tod(si:si+w-1) = junk(ol)
				elseif(dir >= 2) then
					tod(si:si+w-1) = tod(si:si+w-1) + junk(ol)
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
				elseif(dir == 1) then
					tod(si3-w+1:si3) = junk(ol)
				elseif(dir >= 2) then
					tod(si3-w+1:si3) = tod(si3-w+1:si3) + junk(ol)
				end if
				si2           = si2+w
				w             = w*2
			end do
			ol  = ol+1
			si3 = n-si2+1
			! Middle
			if(dir < 0) then
				junk(ol)   = sum(tod(si:si3))
			elseif(dir == 1) then
				tod(si:si3) = junk(ol)
			elseif(dir >= 2) then
				tod(si:si3) = tod(si:si3) + junk(ol)
			end if
		case(4)
			! Legendre polynomial projection, taken from Jon. The odd determination of
			! numbers of degrees of freedom is also from him.
			select case(n)
				case(0:1); w = 1
				case(2:3); w = 2
				case(4:6); w = 3
				case(7:20);w = 4
				case default;   w = 5 + n/cuttype(2)
			end select
			!w = min(n,4+n/cuttype(2))
			if(w <= 1) then
				if(dir >= 2) then
					tod = tod + junk(1)
				elseif(dir == 1) then
					tod = junk(1)
				elseif(dir < 0) then
					junk(1) = sum(tod)
				end if
				ol = 1
			else
				if(dir == 1) tod = 0
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
						case default; Pc = Pb; Pb = Pa; Pa = ((2*i-1)*x*Pb-(i-1)*Pc)/i
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
		if(dir .eq. -1 .and. cuttype(1) .ne. 0) tod = 0
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

	! Project a point source model to/from TOD. The model uses a radial
	! beam profile that can be stretched to make it elliptical. Might be
	! better to just support passing a 2d beam profile instead...
	!
	! The model is srcs[{dec,ra,T,Q,U,ibx,iby,ibxy},ndir,ndet_or_1,nsrc]
	! The ib parameters control beam stretching. For no stretching,
	! set ibx=iby=1 and ibxy=0.
	!
	! To make things efficient, the beam is only evaluated in an area around
	! each source. This area is controlled by cell_srcs, cell_nsrc and cbox,
	! which tell us which sources are present in which small tile of the map,
	! and and rmax, which spcifies how far out to evaluate the beam itself.
	subroutine pmat_ptsrc( &
			dir, tmul, pmul,           &! Projection direction, tod multiplier, src multiplier
			tod, srcs,                 &! Main inputs/outputs. tod(nsamp,ndet), srcs(nparam,ndet_or_1,ndir,nsrc)
			bore, det_pos, det_comps,  &
			rbox, nbox, yvals,         &! Coordinate transformation
			beam, rbeam, rmax,         &! Beam profile and max radial offset to consider
			cell_srcs, cell_nsrc, cbox &! Relevant source lookup. cell_srcs(:,nx,ny,ndet_or_1,ndir), cell_nsrc(nx,ny,ndet_or_1,ndir)
		)
		use omp_lib
		implicit none
		integer, intent(in)    :: dir
		real(_), intent(in)    :: tmul, pmul
		real(_), intent(inout) :: tod(:,:)
		real(8), intent(inout) :: srcs(:,:,:,:)
		real(8), intent(in)    :: bore(:,:), det_pos(:,:), rbox(:,:), yvals(:,:)
		real(8), intent(in)    :: cbox(:,:), beam(:), rbeam, rmax
		real(_), intent(in)    :: det_comps(:,:)
		integer, intent(in)    :: nbox(:), cell_srcs(:,:,:,:,:), cell_nsrc(:,:,:,:)
		! Work
		integer :: nsamp, ndet, nsrc, nproc, nsrcdet
		! Not the same sdir as in the shift stuff
		integer :: ic, i, id, di, si, xind(3), ig, ig2, cell(2), cell_ind, cid, sdir, ndir
		integer :: steps(3), bind, sdi
		real(8) :: x0(3), inv_dx(3), c0(2), inv_dc(2), xrel(3), work(size(yvals,1),4)
		real(8) :: point(4), phase(3), dec, ra, ddec, dra, bscale(3)
		real(_) :: inv_bres, bx,by,br,brel,bval, c2p,s2p,c1p,s1p
		real(_), parameter   :: pi = 3.14159265359d0
		real(8), allocatable :: amps(:,:,:,:,:), cosdec(:,:,:), ys(:,:,:)
		integer, allocatable :: scandir(:)
		nsamp   = size(tod, 1)
		ndet    = size(tod, 2)
		nsrcdet = size(srcs,2)
		ndir    = size(srcs,3)
		nsrc    = size(srcs,4)

		! Set up scanning direction. Two modes are supported. If ndir is 1, then
		! the same set of parameters are used for both left and rightgoing scans.
		! If ndir is 2, then these are separated.
		allocate(scandir(nsamp))
		if(ndir > 1) then
			call calc_scandir(bore(2,:), scandir)
		else
			scandir = 1
		end if
		! Set up interpolation
		call interpol_prepare(nbox, rbox, steps, x0, inv_dx)
		! And the beam interpolation. The last cell ends at cbox(:,2), but
		! starts one cell-width before that.
		c0 = cbox(:,1)
		inv_dc(1) = size(cell_nsrc,2)/(cbox(1,2)-cbox(1,1))
		inv_dc(2) = size(cell_nsrc,1)/(cbox(2,2)-cbox(2,1))
		inv_bres = (size(beam)-1)/rbeam

		nproc = omp_get_max_threads()
		allocate(cosdec(nsrcdet,ndir,nsrc),amps(3,nsrcdet,ndir,nsrc,nproc))
		cosdec = cos(srcs(1,:,:,:))
		if(dir > 0) then
			do i = 1, nproc; amps(:,:,:,:,i) = srcs(3:5,:,:,:)*pmul; end do
		else
			amps = 0
		end if
		!$omp parallel private(id,di,si,sdi,xrel,xind,ig,work,point,phase,cell,cell_ind,cid,dec,ra,bscale,ddec,dra,sdir,c2p,s2p,c1p,s1p,bx,by,br,brel,bind,bval)
		id = omp_get_thread_num()+1
		!$omp do
		do di = 1, ndet
			do si = 1, nsamp
				sdir = scandir(si)
				sdi  = min(di,nsrcdet)
				! Transform from hor to cel
				include 'helper_bilin.F90'
				! Find which point source lookup cell we are in.
				! dec,ra -> cy,cx
				cell = floor((point(1:2)-c0)*inv_dc)+1
				! Bounds checking. Costs 2% performance. Worth it
				cell(1) = min(size(cell_nsrc,2),max(1,cell(1)))
				cell(2) = min(size(cell_nsrc,1),max(1,cell(2)))
				if(dir > 0) tod(si,di) = tod(si,di)*tmul
				! Avoid expensive operations if we don't hit any sources
				if(cell_nsrc(cell(2),cell(1),sdi,sdir) == 0) cycle
				! The spin-2 and spin-1 rotations associated with the transformation
				! We need these to get the polarization rotation and beam orientation
				! right.
				c2p = point(3);                  s2p = point(4)
				c1p = sign(sqrt((1+c2p)/2),s2p); s1p = sqrt((1-c2p)/2)
				phase(1) = det_comps(1,di)
				phase(2) = c2p*det_comps(2,di) - s2p*det_comps(3,di)
				phase(3) = s2p*det_comps(2,di) + c2p*det_comps(3,di)
				! Process each point source in this cell
				do cell_ind = 1, cell_nsrc(cell(2),cell(1),sdi,sdir)
					cid = cell_srcs(cell_ind,cell(2),cell(1),sdi,sdir)+1
					dec   = srcs(1,sdi,sdir,cid)
					ra    = srcs(2,sdi,sdir,cid)
					bscale= srcs(6:8,sdi,sdir,cid)
					! Calc effective distance from this source in terms of the beam distortions.
					! The beam shape is defined in the same coordinate system the polarization
					! orientation is defined in. We can either rotate the beam (like we do phase)
					! or rotate the offset vector the opposite direction. I choose the latter
					! because it is simpler.
					ddec = point(1)-dec
					dra  = (point(2)-ra)*cosdec(sdi,sdir,cid) ! Caller should beware angle wrapping!
					if(abs(ddec)>rmax .or. abs(dra)>rmax) cycle
					bx   =  c1p*dra + s1p*ddec
					by   = -s1p*dra + c1p*ddec
					br   = sqrt(by*(bscale(1)*by+2*bscale(3)*bx) + bx**2*bscale(2))
					! Linearly interpolate the beam value
					brel = br*inv_bres+1
					bind = floor(brel)
					if(bind >= size(beam)) cycle
					brel = brel-bind
					bval = beam(bind)*(1-brel) + beam(bind+1)*brel
					! And perform the actual projection
					if(dir > 0) then
						tod(si,di) = tod(si,di) + sum(amps(:,sdi,sdir,cid,1)*phase)*bval
					else
						amps(:,sdi,sdir,cid,id) = amps(:,sdi,sdir,cid,id) + tod(si,di)*bval*phase
					end if
				end do
			end do
		end do
		!$omp end parallel
		if(dir < 0) then
			srcs(3:5,:,:,:) = srcs(3:5,:,:,:)*pmul + sum(amps,5)*tmul
		end if
	end subroutine


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
			rbox, nbox, yvals,         &! Coordinate transformation
			beam, rbeam, rmax,         &! Beam profile and max radial offset to consider
			cell_srcs, cell_nsrc, cbox &! Relevant source lookup. cell_srcs(:,nx,ny,ndir), cell_nsrc(nx,ny,ndir)
		)
		use omp_lib
		implicit none
		integer, intent(in)    :: dir
		real(_), intent(in)    :: tmul, pmul
		real(_), intent(inout) :: tod(:,:)
		real(8), intent(inout) :: srcs(:,:,:)
		real(8), intent(in)    :: bore(:,:), det_pos(:,:), rbox(:,:), yvals(:,:)
		real(8), intent(in)    :: cbox(:,:), beam(:), rbeam, rmax
		real(_), intent(in)    :: det_comps(:,:)
		integer, intent(in)    :: nbox(:), cell_srcs(:,:,:,:), cell_nsrc(:,:,:)
		! Work
		integer :: nsamp, ndet, nsrc, nproc
		! Not the same sdir as in the shift stuff
		integer :: ic, i, id, di, si, xind(3), ig, ig2, cell(2), cell_ind, cid, sdir, ndir
		integer :: steps(3), bind
		real(8) :: x0(3), inv_dx(3), c0(2), inv_dc(2), xrel(3), work(size(yvals,1),4)
		real(8) :: point(4), phase(3), dec, ra, ddec, dra, ibeam(3)
		real(_) :: inv_bres, bx,by,br,brel,bval, c2p,s2p,c1p,s1p
		real(_), parameter   :: pi = 3.14159265359d0
		real(8), allocatable :: amps(:,:,:,:), cosdec(:,:), ys(:,:,:)
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
		! inv_dx uses (nbox-1) because we use an endpoint-inclusive
		! grid here: Last data point is exactly at rbox(:,2) --
		! it isn't a cell that just ends there.
		x0 = rbox(:,1); inv_dx = (nbox-1)/(rbox(:,2)-rbox(:,1))
		c0 = cbox(:,1)
		! inv_dc does not use -1 because it uses cells rather
		! than points. The last cell ends at cbox(:,2), but
		! starts one cell-width before that.
		inv_dc(1) = size(cell_nsrc,2)/(cbox(1,2)-cbox(1,1))
		inv_dc(2) = size(cell_nsrc,1)/(cbox(2,2)-cbox(2,1))
		inv_bres = (size(beam)-1)/rbeam

		! Precompute derivatives of yvals, called ys. This will be very
		! fast, as we're only talking about at most 1e6 samples or so.
		allocate(ys(size(yvals,1),3,size(yvals,2)))
		do ig = 1, size(yvals,2)
			do ic = 1, 3
				! This is only valid up to nbox-1 of the interpolation grid, but
				! that's the only part we will use when we use these derivatives
				ig2 = min(size(yvals,2),ig+steps(ic))
				ys(:,ic,ig) = yvals(:,ig2)-yvals(:,ig)
			end do
		end do

		nproc = omp_get_max_threads()
		allocate(cosdec(ndir,nsrc),amps(3,ndir,nsrc,nproc))
		cosdec = cos(srcs(1,:,:))
		if(dir > 0) then
			do i = 1, nproc; amps(:,:,:,i) = srcs(3:5,:,:)*pmul; end do
		else
			amps = 0
		end if
		!$omp parallel private(id,di,si,xrel,xind,ig,work,point,phase,cell,cell_ind,cid,dec,ra,ibeam,ddec,dra,sdir,c2p,s2p,c1p,s1p,bx,by,br,brel,bind,bval)
		id = omp_get_thread_num()+1
		!$omp do
		do di = 1, ndet
			do si = 1, nsamp
				sdir = scandir(si)
				! Transform from hor to cel
				include 'helper_bilin.F90'
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

	subroutine pmat_az_off(dir, tod, map, az, azoff, az0, daz, yoff, comps)
		use omp_lib
		implicit none
		integer, intent(in)    :: dir, yoff(:)
		real(_), intent(inout) :: tod(:,:), map(:,:,:)
		real(_), intent(in)    :: az0, daz, az(:), azoff(:), comps(:,:)
		real(_), allocatable   :: wmap(:,:,:,:)
		real(_)                :: idaz, v, q
		integer :: di, si, ndet, nsamp, naz, ai, nproc, ci, rank, x, y
		ndet  = size(tod,2)
		nsamp = size(tod,1)
		naz   = size(map,1)
		idaz  = 1/daz

		if(dir > 0) then
			!$omp parallel do private(di,si,ai,v)
			do di = 1, ndet
				do si = 1, nsamp
					ai = int((az(si)+azoff(di)-az0)*idaz)+1
					! Skip all out-of-bounds pixels
					if(ai < 1 .or. ai > naz) cycle
					tod(si,di) = sum(map(ai,yoff(di)+1,1:3)*comps(1:3,di))
				end do
			end do
		else
			nproc = omp_get_max_threads()
			allocate(wmap(size(map,3),size(map,1),size(map,2),nproc))
			wmap = 0
			!$omp parallel private(di,si,ai,v,rank)
			rank = omp_get_thread_num()+1
			!$omp do
			do di = 1, ndet
				do si = 1, nsamp
					ai = int((az(si)+azoff(di)-az0)*idaz)+1
					! Skip all out-of-bounds pixels
					if(ai < 1 .or. ai > naz) cycle
					wmap(1:3,ai,yoff(di)+1,rank) = wmap(1:3,ai,yoff(di)+1,rank) + tod(si,di)*comps(1:3,di)
				end do
			end do
			!$omp end parallel
			do y = 1, size(map,2)
				do x = 1, size(map,1)
					do ci = 1, 3
						map(x,y,ci) = map(x,y,ci) + sum(wmap(ci,x,y,:))
					end do
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

	subroutine pmat_plain(dir, tod, map, pix)
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
				wmap(ipix(2),ipix(1),:,id) = wmap(ipix(2),ipix(1),:,id) + tod(i,:)
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

	subroutine add_rows(map, ys, xranges, vals, nphi)
		use omp_lib
		implicit none
		integer, intent(in)    :: ys(:), xranges(:,:), nphi
		real(_), intent(in)    :: vals(:)
		real(_), intent(inout) :: map(:,:)
		integer :: y, x, xraw, i
		do i = 1, size(ys)
			y = max(1,min(size(map,2),ys(i)+1))
			do xraw = xranges(1,i)+1, xranges(2,i)
				x = xraw
				if(nphi > 0) x = modulo(x-1,nphi)+1
				x = max(1,min(size(map,1),x))
				map(x,y) = map(x,y) + vals(i)
			end do
		end do
	end subroutine

	subroutine calc_scandir(az, scandir)
		! Samples where az decreases or increass will be
		! assigned a scandir of 1 or 2.
		implicit none
		real(8), intent(in)    :: az(:)
		integer, intent(inout) :: scandir(:)
		integer :: si
		scandir(1) = 1
		do si = 2, size(az)
			scandir(si) = merge(1,2,az(si)>=az(si-1))
		end do
	end subroutine

	subroutine pmat_noise_rect(dir, tod, tmul, map, mmul, bore, det_offs, det_comps, box, scandir)
		! Pmat for a flat, wrapping coordinate system
		use omp_lib
		implicit none
		integer, intent(in)      :: dir, scandir(:)
		real(_), intent(inout)   :: tod(:,:), map(:,:,:)
		real(_), intent(in)      :: tmul, mmul
		real(8), intent(in)      :: bore(:,:), det_offs(:,:), det_comps(:,:), box(:,:)
		real(_), allocatable     :: wmap(:,:,:)
		real(8) :: pos(2), pixdens(2), p0(2)
		real(_) :: v
		integer :: di, si, ndet, nsamp, ai, nproc, ci, pix(2), mshape(2), y, x, o
		nsamp = size(tod,1)
		ndet  = size(tod,2)
		p0    = box(:,1)
		! mshape is ny,nx
		mshape(1) = size(map,2)
		mshape(2) = size(map,1)
		pixdens = mshape/(box(:,2)-box(:,1))
		if(dir > 0) then
			!$omp parallel do private(di,si,pos,pix,o)
			do di = 1, ndet
				do si = 1, nsamp
					o = 0; if(size(map,3) >= 6) o = 3*scandir(si)
					pos = bore(1:2,si) + det_offs(1:2,di)
					pix = modulo(nint((pos-p0)*pixdens), mshape)+1
					tod(si,di) = tod(si,di)*tmul + sum(map(pix(2),pix(1),1+o:3+o)*det_comps(1:3,di))*mmul
				end do
			end do
		else
			allocate(wmap(size(map,1),size(map,2),size(map,3)))
			nproc = omp_get_max_threads()
			if(nproc == 1) then
				wmap = 0
				do di = 1, ndet
					do si = 1, nsamp
						o = 0; if(size(map,3) >= 6) o = 3*scandir(si)
						pos = bore(1:2,si) + det_offs(1:2,di)
						pix = modulo(nint((pos-p0)*pixdens), mshape)+1
						wmap(pix(2),pix(1),1+o:3+o) = wmap(pix(2),pix(1),1+o:3+o) + tod(si,di)*det_comps(1:3,di)
					end do
				end do
				map = map*mmul + wmap*tmul
			else
				!$omp parallel private(di,si,ci,pos,pix,v,y,x,o)
				!$omp workshare
				wmap = 0
				!$omp end workshare
				!$omp barrier
				!$omp do
				do di = 1, ndet
					do si = 1, nsamp
						o = 0; if(size(map,3) >= 6) o = 3*scandir(si)
						pos = bore(1:2,si) + det_offs(1:2,di)
						pix = modulo(nint((pos-p0)*pixdens), mshape)+1
						do ci = 1+o, 3+o
							v = tod(si,di)*det_comps(ci-o,di)
							!$omp atomic
							wmap(pix(2),pix(1),ci) = wmap(pix(2),pix(1),ci) + v
							!$omp end atomic
						end do
					end do
				end do
				!$omp barrier
				!$omp do collapse(3)
				do ci = 1, size(map,3)
					do y = 1, size(map,2)
						do x = 1, size(map,1)
							map(x,y,ci) = map(x,y,ci)*mmul + wmap(x,y,ci)*tmul
						end do
					end do
				end do
				!$omp end parallel
			end if
			deallocate(wmap)
		end if
	end subroutine

	subroutine precompute_pointing_grid( &
		pix, phase,                    &! The output pixel[yx,nsamp,ndet] and phase[TQU,nsamp,ndet] arrays
		pmet,                          &! Grid pointing interpol variant: 1: bilinear, 2:gradient
		bore, hwp, det_pos, det_comps, &! Input pointing
		rbox, nbox, yvals,             &! Interpolation grid
		wbox, nphi                     &! wbox({y,x},{from,to}) pixbox and sky wrap in pixels
	)
		use omp_lib
		implicit none
		! Parameters
		real(8),    intent(inout) :: pix(:,:,:)
		real(_),    intent(inout) :: phase(:,:,:)
		integer(4), intent(in)    :: nbox(:), wbox(:,:), nphi, pmet
		real(8),    intent(in)    :: bore(:,:), hwp(:,:), yvals(:,:), det_pos(:,:), rbox(:,:)
		real(8),    intent(in)    :: det_comps(:,:)
		! Work
		integer(4) :: nsamp, ndet, di, steps(3)
		real(8)    :: x0(3), inv_dx(3)
		nsamp   = size(bore, 2)
		ndet    = size(det_comps, 2)
		call interpol_prepare(nbox, rbox, steps, x0, inv_dx)
		!$omp parallel do
		do di = 1, ndet
			call build_pointing_grid(pmet, bore, hwp, pix(:,:,di), phase(:,:,di), &
				det_pos(:,di), det_comps(:,di), steps, x0, inv_dx, yvals)
			call cap_pixels(pix(:,:,di), wbox)
			pix(1,:,di) = pix(1,:,di) + wbox(1,1)
			pix(2,:,di) = pix(2,:,di) + wbox(2,1)
		end do
	end subroutine

	! Comap test stuff
	subroutine pmat_comap(dir, tod, map, pix)
		use omp_lib
		implicit none
		integer, intent(in)    :: dir, pix(:,:,:)
		real(_), intent(inout) :: map(:,:,:,:), tod(:,:,:)
		integer :: ndet, nfreq, nsamp, di, fi, si, y, x
		! pix = [{y,x},nsamp,ndet]
		! tod = [nfreq,nsamp,ndet]
		! map = [nfreq,nx,ny,ndet]
		ndet  = size(tod,3)
		nsamp = size(tod,2)
		nfreq = size(tod,1)
		if(dir > 0) then
		else
			!$omp parallel do collapse(2) private(di,si,y,x,fi)
			do di = 1, ndet
				do si = 1, nsamp
					y = pix(si,di,1)
					x = pix(si,di,2)
					do fi = 1, nfreq
						map(fi,x,y,di) = map(fi,x,y,di) + tod(fi,si,di)
					end do
				end do
			end do
		end if
	end subroutine

end module
