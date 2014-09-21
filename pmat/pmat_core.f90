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
			tod, map,                         &! Main inputs/outpus
			bore, det_pos, det_comps,  comps, &! Input pointing
			rbox, nbox, ys                    &! Coordinate transformation
		)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir, comps(:), nbox(:)
		real(4),    intent(in)    :: bore(:,:), ys(:,:,:) ! comp, derivs, pix
		real(4),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		real(4),    intent(inout) :: tod(:,:), map(:,:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, nproc, id, ic, i, j, k, l, nj
		integer(4) :: steps(size(rbox,1)), pix(2,bz)
		real(4)    :: x0(size(rbox,1)), inv_dx(size(rbox,1)), phase(size(map,3),bz)
		real(4)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(4), allocatable :: wmap(:,:,:,:)

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
				pix(:,:nj)    = nint(opoint(1:2,:nj))+1 ! fortran index offset
				! Bounds check (<1% cost)
				do j = 1, nj
					pix(1,j) = min(size(map,2),max(1,pix(1,j)))
					pix(2,j) = min(size(map,1),max(1,pix(2,j)))
				end do
				phase(:,:nj)  = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
				if(dir < 0) then
					do j = 1, nj
						wmap(pix(2,j),pix(1,j),:,id) = wmap(pix(2,j),pix(1,j),:,id) + tod(si+j-1,di)*phase(:,j)
					end do
				else
					do j = 1, nj
						tod(si+j-1,di) = sum(map(pix(2,j),pix(1,j),:)*phase(:,j))
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
		real(4),    intent(in)    :: bore(:,:), ys(:,:,:) ! comp, derivs, pix
		real(4),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		real(4),    intent(inout) :: tod(:,:), map(:,:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, nproc, id, ic, i, j, k, l, nj, py,px
		integer(4) :: steps(size(rbox,1)), pix(2,bz)
		real(4)    :: x0(size(rbox,1)), inv_dx(size(rbox,1)), phase(size(map,3),bz)
		real(4)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		real(4)    :: mt(2,2), fpix(2,bz)
		real(4), allocatable :: wmap(:,:,:,:)

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
		real(4),    intent(in)    :: bore(:,:), ys(:,:,:) ! comp, derivs, pix
		real(4),    intent(inout) :: phase(:,:,:), pix(:,:,:)
		real(4),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, nproc, id, ic, i, j, k, l, nj
		integer(4) :: steps(size(rbox,1))
		real(4)    :: x0(size(rbox,1)), inv_dx(size(rbox,1))
		real(4)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)

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

	subroutine hitcount(hitmap, bore, det_pos, rbox, nbox, ys)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: nbox(:)
		real(4),    intent(in)    :: bore(:,:), ys(:,:,:)
		real(4),    intent(in)    :: det_pos(:,:), rbox(:,:)
		integer(4), intent(inout) :: hitmap(:,:)
		! Work
		integer(4), parameter :: bz = 321
		integer(4) :: ndet, nsamp, di, si, nproc, id, ic, i, j, k, l, nj
		integer(4) :: steps(size(rbox,1)), pix(2,bz)
		real(4)    :: x0(size(rbox,1)), inv_dx(size(rbox,1))
		real(4)    :: ipoint(size(bore,1),bz), opoint(size(ys,1),bz)
		integer(4), allocatable :: wmap(:,:,:)

		nsamp   = size(bore,    2)
		ndet    = size(det_pos, 2)

		! In C order, ys has pixel axes t,ra,dec, so nbox = [nt,nra,ndec]
		! Index mapping is therefore given by [nra*ndec,ndec,1]
		steps(size(steps)) = 1
		do ic = size(steps)-1, 1, -1
			steps(ic) = steps(ic+1)*nbox(ic+1)
		end do
		x0 = rbox(:,1); inv_dx = nbox/(rbox(:,2)-rbox(:,1))

		nproc = omp_get_max_threads()
		allocate(wmap(size(hitmap,1),size(hitmap,2),nproc))
		!$omp parallel workshare
		wmap = 0
		!$omp end parallel workshare

		!$omp parallel private(di,si,nj,j,ipoint,opoint,pix,id)
		id = omp_get_thread_num()+1
		!$omp do collapse(2)
		do di = 1, ndet
			do si = 1, nsamp, bz
				nj = min(bz,nsamp-si+1)
				do j = 1, nj
					ipoint(:,j) = bore(:,si+j-1)+det_pos(:,di)
				enddo
				opoint(:,:nj) = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
				pix(:,:nj)    = nint(opoint(1:2,:nj))+1 ! fortran index offset
				! Bounds check (<1% cost)
				do j = 1, nj
					pix(1,j) = min(size(hitmap,2),max(1,pix(1,j)))
					pix(2,j) = min(size(hitmap,1),max(1,pix(2,j)))
				end do
				do j = 1, nj
					wmap(pix(2,j),pix(1,j),id) = wmap(pix(2,j),pix(1,j),id) + 1
				end do
			end do
		end do
		!$omp end parallel

		!$omp parallel do collapse(3)
		do k = 1, size(wmap, 2)
			do l = 1, size(wmap, 1)
				do i = 1, size(wmap,3)
					hitmap(l,k) = hitmap(l,k) + wmap(l,k,i)
				end do
			end do
		end do
	end subroutine

	! Trilinear is 35% slower for almost no gain, it seems.
	!function lookup_trilinear(ipoint, x0, inv_dx, steps, ys) result(opoint)
	!	implicit none
	!	real(4),    intent(in) :: ipoint(:,:), x0(:), inv_dx(:)
	!	real(4),    intent(in) :: ys(:,:,:)
	!	integer(4), intent(in) :: steps(:)
	!	real(4)                :: opoint(size(ys,1),size(ipoint,2))
	!	real(4)    :: xrel(size(x0))
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
	function lookup_grad(ipoint, x0, inv_dx, steps, ys) result(opoint)
		implicit none
		real(4),    intent(in) :: ipoint(:,:), x0(:), inv_dx(:)
		real(4),    intent(in) :: ys(:,:,:)
		integer(4), intent(in) :: steps(:)
		real(4)                :: opoint(size(ys,1),size(ipoint,2))
		real(4)    :: xrel(size(x0))
		integer(4) :: xind(size(x0)), ig, oc, i, j
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
		real(4),    intent(in) :: det_comps(:), cossin(:,:)
		real(4)                :: phase(size(comps),size(cossin,2))
		integer(4) :: i, j
		do j = 1, size(cossin,2)
			do i = 1, size(phase,1)
				if(comps(i) == 0) phase(i,j) = det_comps(1)
				if(comps(i) == 1) phase(i,j) = cossin(1,j)*det_comps(2) - cossin(2,j)*det_comps(3)
				if(comps(i) == 2) phase(i,j) = cossin(2,j)*det_comps(2) + cossin(1,j)*det_comps(3)
				if(comps(i) == 3) phase(i,j) = det_comps(4)
			end do
		end do
	end function

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
		real(4),    intent(inout) :: tod(:,:), junk(:)
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
		real(4) :: foo(1)
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
		real(4),    intent(inout) :: junk(:), tod(:)
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
		real(4), intent(in)  :: a(:)
		real(4)              :: mean
		mean = sum(a)/size(a)
	end function

	subroutine pmat_map_rebin(dir, map_high, map_low)
		use omp_lib
		implicit none
		! Parameters
		integer(4), intent(in)    :: dir
		real(4),    intent(inout) :: map_high(:,:,:), map_low(:,:,:)
		! Work
		integer(4) :: hx,hy,lx,ly, n, step
		real(4)    :: val(size(map_low,3))

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
		real(4),    intent(inout) :: junk_high(:), junk_low(:)
		integer(4), parameter     :: det=1, lstart=2, llen=3, gstart=4, glen=5, cuttype=6
		integer(4) :: ci, di, gl1, gl2, gh1, gh2, nl, nh, i
		real(4), allocatable :: ibuf(:), obuf(:)
		! Assume that the high-res and low-res cuts have the same number of entries
		! and come in the same order.
		!$omp parallel do default(private) shared(junk_high,junk_low,cut_high,cut_low)
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

end module


