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

	subroutine pmat_nearest( &
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
		integer(4) :: si, w, bi, si2, si3, n, ol
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
			w = cuttype(2)
			do bi = 1, (n-1)/w
				ol = ol+1
				if(dir < 0) then
					junk(ol) = mean(tod((bi-1)*w+1:bi*w))
				elseif(dir > 0) then
					tod((bi-1)*w+1:bi*w) = junk(ol)
				end if
			end do
			ol = ol+1
			if(dir < 0) then
				junk(ol) = mean(tod((bi-1)*w+1:n))
			elseif(dir < 0) then
				tod((bi-1)*w+1:n) = junk(ol)
			end if
		case(3)
			! Exponential cuts. Full resolution near edges, low in the middle,
			! with bin size doubling with distance to edge.
			! Left edge
			w  = 1
			si = 1
			do while(si+w < (n+1)/2)
				ol = ol+1
				if(dir < 0) then
					junk(ol) = mean(tod(si:si+w-1))
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
					junk(ol) = mean(tod(si3-w+1:si3))
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
				junk(ol)   = mean(tod(si:si3))
			elseif(dir > 0) then
				tod(si:si3) = junk(ol)
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
			!$omp do collapse(2)
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
			!$omp parallel do collapse(2) private(di,si,ci)
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
	! each source based on how much eac detector hits it. This can be used to
	! divide sources into sets of sufficient sensitivity.
	!  pos(2,nsrc), rhit(nsrc), rmax(nsrc), det_ivars(nsrc), src_ivars(nsrc)
	!  ranges(2,maxrange,ndet,nsrc), nrange(ndet,nsrc)
	! ranges is zero-based and half-open, like python.
	! maxrange must be large enough to hold all the discovered ranges. Otherwise,
	! the last ranges will be lost. Nrange holds the actual number of discovered
	! ranges. ranges has units of det-local samples.
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
			ranges, rangesets, offsets           &! Precomputed relevant sample info
		)
		use omp_lib
		implicit none
		! Parameters
		real(_),    intent(in)    :: tod(:,:)      ! (nsamp,ndet)
		real(_),    intent(inout) :: out_tod(:), point(:,:), phase(:,:)
		real(_),    intent(in)    :: bore(:,:), ys(:,:,:)
		real(_),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		integer(4), intent(inout) :: oranges(:,:)
		integer(4), intent(in)    :: comps(:), nbox(:), offsets(:,:,:), rangesets(:), ranges(:,:)
		! Work
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
				do j = 1, nj
					ipoint(:,j) = bore(:,i+j-1)+det_pos(:,di)
				enddo
				opoint(:,:nj)     = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
				phase(:,k+1:k+nj) = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
				point(:,k+1:k+nj) = opoint(1:2,:nj)
				out_tod(k+1:k+nj) = tod(i:i+nj-1,di)
				k = k+nj
			end do
			oranges(2,ri) = k
		end do
	end subroutine

end module


