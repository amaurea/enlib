module pmat_core

contains

! Performance for tod[445,262000]:
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
! Bounds checking costs about 1%. That's worth it.
subroutine map_nearest( &
		dir,                              &! Direction of the projection: 1: forward (map2tod), -1: backard (tod2map)
		tod, map,                         &! Main inputs/outpus
		bore, det_pos, det_comps,  comps, &! Input pointing
		rbox, nbox, ys                    &! Coordinate transformation
	)
	use omp_lib
	implicit none
	! Parameters
	integer(4), intent(in)    :: dir, comps(:), nbox(:)
	real(4),    intent(in)    :: bore(:,:)
	real(4),    intent(in)    :: det_pos(:,:), det_comps(:,:), rbox(:,:)
	real(4),    intent(in)    :: ys(:, :, :) ! comp, derivs, pix
	real(4),    intent(inout) :: tod(:,:), map(:,:,:)
	! Work
	integer(4), parameter :: bz = 321
	integer(4) :: ndet, nsamp, di, si, nproc, myid, ic, i, j, k, l, nj
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

	!$omp parallel do collapse(2) private(di,si,nj,j,ipoint,opoint,pix)
	do di = 1, ndet
		do si = 1, nsamp, bz
			nj = min(bz,nsamp-si+1)
			do j = 1, nj
				ipoint(:,j) = bore(:,si+j-1)+det_pos(:,di)
			enddo
			opoint(:,:nj)  = lookup_grad(ipoint(:,:nj), x0, inv_dx, steps, ys)
			pix(:,:nj)    = floor(opoint(1:2,:nj))+1 ! fortran index offset
			! Bounds check (<1% cost)
			do j = 1, nj
				pix(1,j) = min(size(map,2),max(1,pix(1,j)))
				pix(2,j) = min(size(map,1),max(1,pix(2,j)))
			end do
			phase(:,:nj)  = get_phase(comps, det_comps(:,di), opoint(3:,:nj))
			if(dir < 0) then
				do j = 1, nj
					map(pix(2,j),pix(1,j),:) = map(pix(2,j),pix(1,j),:) + tod(si+j-1,di)*phase(:,j)
				end do
			else
				do j = 1, nj
					tod(si+j-1,di) = sum(map(pix(2,j),pix(1,j),:)*phase(:,j))
				end do
			end if
		end do
	end do

	if(dir > 0) then
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

pure function get_phase(comps, det_comps, cossin) result(phase)
	implicit none
	integer(4), intent(in) :: comps(:)
	real(4),    intent(in) :: det_comps(:), cossin(:,:)
	real(4)                :: phase(size(comps),size(cossin,2))
	integer(4) :: i, j
	do j = 1, size(cossin,2)
		do i = 1, size(phase,1)
			if(comps(i) == 0) phase(i,j) = det_comps(1)
			if(comps(i) == 1) phase(i,j) = det_comps(2)*cossin(1,j) + det_comps(3)*cossin(2,j)
			if(comps(i) == 2) phase(i,j) =+det_comps(2)*cossin(2,j) - det_comps(3)*cossin(1,j)
			if(comps(i) == 3) phase(i,j) = det_comps(4)
		end do
	end do
end function

end module
