module fortran
	implicit none

contains

	subroutine test(tod, ranges, rangesets, offsets, vars, nvars, detrend)
		implicit none
		real(_), intent(in) :: tod(:)
		real(_), intent(inout) :: vars(:,:)
		integer(4), intent(inout) :: nvars(:,:)
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2)), r2src(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi, i, i1, i2, detrend
		real(_)    :: m,s,x,sn,mid
		nsrc = size(offsets,2)
		vars = 0
		nvars = 0
	end subroutine

	! Apply a simple white noise + mean subtraction noise model to a subrange tod with
	! ranges specified by offsets[src,det], ranges[nrange,2].
	subroutine nmat_mwhite(tod, ranges, rangesets, offsets, ivar, detrend, rangemask)
		implicit none
		real(_), intent(inout) :: tod(:)
		integer(4), intent(in)    :: rangemask(:)
		real(_)    :: ivar(:)
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi, i, i1, i2, detrend, err
		real(_)    :: m,s,x,sn,mid
		nsrc = size(offsets,2)
		ndet = size(offsets,1)-1

		! Prepare for parallel loop
		do si = 1, nsrc
			do di = 1, ndet
				do oi = offsets(di,si)+1, offsets(di+1,si)
					ri = rangesets(oi)+1
					r2det(ri) = di
				end do
			end do
		end do

		!$omp parallel do private(ri,i1,i2,m,s,x,sn,mid)
		do ri = 1, size(ranges,2)
			if(rangemask(ri) .eq. 0) cycle
			i1 = ranges(1,ri)+1
			i2 = ranges(2,ri)
			if(i2-i1 < 0) cycle
			if(detrend > 1 .and. i1 .ne. i2) then
				m = 0; s = 0; sn = 0
				mid = 0.5*(i1+i2)
				do i = i1, i2
					m = m + tod(i)
					x = i - mid
					s = s + tod(i)*x
					sn= sn+ x*x
				end do
				m = m/(i2-i1+1)
				s = s/sn
				do i = i1, i2
					x = i - mid
					tod(i) = tod(i) - m - x*s
				end do
			elseif(detrend > 0) then
				tod(i1:i2) = tod(i1:i2) - sum(tod(i1:i2))/(i2-i1+1)
			end if
			tod(i1:i2) = tod(i1:i2) * ivar(r2det(ri))
		end do
	end subroutine

	subroutine measure_mwhite(tod, ranges, rangesets, offsets, vars, nvars, detrend)
		implicit none
		real(_), intent(in) :: tod(:)
		real(_), intent(inout) :: vars(:,:)
		integer(4), intent(inout) :: nvars(:,:)
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2)), r2src(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi, i, i1, i2, detrend
		real(_)    :: m,s,x,sn,mid
		nsrc = size(offsets,2)
		ndet = size(offsets,1)-1

		vars  = 0
		nvars = 0

		! Prepare for parallel loop
		!$omp parallel do collapse(2) private(si,di,oi,ri,i1,i2,m,s,sn,mid,x)
		do si = 1, nsrc
			do di = 1, ndet
				do oi = offsets(di,si)+1, offsets(di+1,si)
					ri = rangesets(oi)+1
					i1 = ranges(1,ri)+1
					i2 = ranges(2,ri)
					if(i2-i1 < 0) cycle
					if(detrend > 1 .and. i1 .ne. i2) then
						m = 0; s = 0; sn = 0
						mid = 0.5*(i1+i2)
						do i = i1, i2
							m = m + tod(i)
							x = i - mid
							s = s + tod(i)*x
							sn= sn+ x*x
						end do
						m = m/(i2-i1+1)
						s = s/sn
						do i = i1, i2
							x = i - mid
							vars(di,si)	= vars(di,si) + (tod(i)-m-x*s)**2
						end do
					elseif(detrend > 0) then
						vars(di,si) = vars(di,si) + sum((tod(i1:i2)-sum(tod(i1:i2))/(i2-i1+1))**2)
					else
						vars(di,si) = vars(di,si) + sum(tod(i1:i2)**2)
					end if
					nvars(di,si)= nvars(di,si) + i2-i1+1
				end do
			end do
		end do
	end subroutine

	! Apply a simple white + basis projection noise model to a subrange tod with
	! ranges specified by offsets[src,det], ranges[nrange,2]. The basis functions
	! are supplied as bvecs[nvec,nsamp] with the same ordering as tod. We assume
	! that there aren't that many of these. For a given range, one can project
	! out these by using the woodbury formula:
	!  (N+VEV')" = N" - N"V(E" + V'N"V)"V'N"
	! Assuming we want to get completely rid of those modes, E=inf and we get
	!  N" - N"V(V'N"V)"V'N"
	! If N is proportional to I, then this is simply N"(1-V(V'V)"V'), i.e.
	! project out V-part and then apply N". Q = V(V'V)**-0.5 can be precomputed
	! and used instead of V, to avoid needing lapack. Then we get N"(1-QQ').
	! Q is then [nvec,nsamp]
	subroutine nmat_basis(tod, ranges, rangesets, offsets, ivar, Q, rangemask)
		implicit none
		real(_), intent(inout) :: tod(:)
		integer(4), intent(in) :: rangemask(:)
		real(_)    :: ivar(:), Q(:,:), y(size(Q,1))
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi, i, j, i1, i2, n
		real(_), allocatable :: x(:)
		nsrc = size(offsets,2)
		ndet = size(offsets,1)-1

		! Prepare for parallel loop
		do si = 1, nsrc
			do di = 1, ndet
				do oi = offsets(di,si)+1, offsets(di+1,si)
					ri = rangesets(oi)+1
					r2det(ri) = di
				end do
			end do
		end do

		!$omp parallel do private(ri,i1,i2,n,x,i,y)
		do ri = 1, size(ranges,2)
			if(rangemask(ri) .eq. 0) cycle
			i1 = ranges(1,ri)+1
			i2 = ranges(2,ri)
			if(i2-i1 < 0) cycle
			! Project out given vectors
			n = i2-i1+1
			x = tod(i1:i2)
			! The Q we receive from python is [nmode,nsamp], which
			! is the transpose of what we want here. So we must
			! compute Q'Q rather than QQ'.
			!  y = Qd; d -= (Q'y = Q'Qd)
			do i = 1, size(y)
				y(i) = sum(Q(i,i1:i2)*tod(i1:i2))
			end do
			do i = i1, i2
				tod(i) = tod(i) - sum(Q(:,i)*y)
			end do
			tod(i1:i2) = tod(i1:i2) * ivar(r2det(ri))
		end do
	end subroutine

	subroutine measure_basis(tod, ranges, rangesets, offsets, vars, nvars, Q)
		implicit none
		real(_), intent(in) :: tod(:), Q(:,:)
		real(_), intent(inout) :: vars(:,:)
		integer(4), intent(inout) :: nvars(:,:)
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2)), r2src(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi, i, i1, i2, n
		real(_), allocatable :: x(:)
		nsrc = size(offsets,2)
		ndet = size(offsets,1)-1

		vars  = 0
		nvars = 0

		! Prepare for parallel loop
		!$omp parallel do collapse(2) private(si,di,oi,ri,i1,i2,n,x)
		do si = 1, nsrc
			do di = 1, ndet
				do oi = offsets(di,si)+1, offsets(di+1,si)
					ri = rangesets(oi)+1
					i1 = ranges(1,ri)+1
					i2 = ranges(2,ri)
					if(i2-i1 < 0) cycle
					n = i2-i1+1
					allocate(x(n))
					x = tod(i1:i2) - matmul(transpose(Q(:,i1:i2)),matmul(Q(:,i1:i2),tod(i1:i2)))
					vars(di,si) = vars(di,si) + sum(x**2)
					nvars(di,si)= nvars(di,si) + n
					deallocate(x)
				end do
			end do
		end do
	end subroutine

	subroutine pmat_thumbs(dir, tod, maps, point, phase, boxes)
		use omp_lib
		implicit none
		integer(4), intent(in)    :: dir
		real(_),    intent(inout) :: tod(:), maps(:,:,:,:)
		real(_),    intent(in)    :: point(:,:), phase(:,:), boxes(:,:,:)
		real(_),    allocatable   :: wmaps(:,:,:,:,:)
		integer(4) :: i,j,k,l,a,jprev,p(2), nsamp, n(2), nmap, ncomp, nproc, id
		nsamp = size(tod)
		! map is (nx,ny,nc,nm). boxes is ({dec,ra},2,nm), point is ({dec,ra},n)
		! we want ra = x, dec = y, so some indices will be transposed
		n(1) = size(maps,2)
		n(2) = size(maps,1)
		ncomp= size(maps,3)
		nmap = size(maps,4)

		if(dir < 0) then
			nproc = omp_get_max_threads()
			allocate(wmaps(size(maps,1),size(maps,2),size(maps,3),size(maps,4),nproc))
			!$omp parallel workshare
			wmaps = 0
			!$omp end parallel workshare
		end if

		!$omp parallel private(jprev,i,k,j,p,id)
		id = omp_get_thread_num()+1
		jprev= 1
		!$omp do
		do i = 1, nsamp
			! Semi-naive approach: Brute force search, but start at previous match.
			do k = 0, nmap
				j = k; if(k<1) j=jprev
				p = floor((point(:,i)-boxes(:,1,j))*n/(boxes(:,2,j)-boxes(:,1,j)))+1
				jprev=j
				if(all(p>0) .and. all(p<=n)) exit
			end do
			if(j > nmap) cycle
			if(dir < 0) then
				wmaps(p(2),p(1),:,j,id) = wmaps(p(2),p(1),:,j,id) + tod(i)*phase(1:ncomp,i)
			else
				tod(i) = sum(maps(p(2),p(1),:,j)*phase(1:ncomp,i))
			end if
		end do
		!$omp end parallel

		if(dir < 0) then
			!$omp parallel do collapse(4)
			do a = 1, size(wmaps, 4)
				do j = 1, size(wmaps, 3)
					do k = 1, size(wmaps, 2)
						do l = 1, size(wmaps, 1)
							do i = 1, size(wmaps,5)
								maps(l,k,j,a) = maps(l,k,j,a) + wmaps(l,k,j,a,i)
							end do
						end do
					end do
				end do
			end do
		end if
	end subroutine

	subroutine pmat_model(dir, tod, params, ranges, rangesets, offsets, point, phase, rangemask)
		use omp_lib
		implicit none
		! Parameters
		real(_),    intent(inout) :: tod(:), params(:,:)
		real(_),    intent(in)    :: point(:,:), phase(:,:)
		integer(4), intent(in)    :: offsets(:,:), ranges(:,:), rangesets(:), dir
		integer(4), intent(in)    :: rangemask(:)
		! Work
		integer(4) :: si, di, oi, ri, i, nsrc, ndet, namp
		real(_)    :: ra, dec, amps(size(params,1)-5), ibeam(3), ddec, dra, r2, cosdec
		real(_)    :: oamps(size(params,1)-5,size(offsets,2))

		ndet  = size(offsets,1)-1
		nsrc  = size(offsets,2)
		namp  = size(amps)

		if(dir > 0) then
			!$omp parallel workshare
			tod = 0
			!$omp end parallel workshare
		else
			oamps = 0
		end if

		!Note: it's safe to do di in parallel, but no si, as multiple sources may contribute
		!to the same sample.
		!$omp parallel do private(di,si,dec,ra,amps,ibeam,cosdec,oi,ri,i,ddec,dra,r2) reduction(+:oamps)
		do di = 1, ndet
			do si = 1, nsrc
				dec   = params(1,si)
				ra    = params(2,si)
				amps  = params(3:2+namp,si)
				if(dir > 0 .and. all(amps==0)) cycle
				ibeam = params(3+namp:5+namp,si)
				cosdec= cos(dec)
				do oi = offsets(di,si)+1, offsets(di+1,si)
					ri = rangesets(oi)+1
					if(rangemask(ri) .eq. 0) cycle
					do i = ranges(1,ri)+1, ranges(2,ri)
						! Compute shape-normalized distance from each sample to the current source.
						ddec = dec-point(1,i)
						dra  = (ra-point(2,i))*cosdec
						r2   = ddec*(ibeam(1)*ddec+ibeam(3)*dra) + dra*(ibeam(2)*dra+ibeam(3)*ddec)
						if(dir > 0) then
							! And finally evaluate the model.
							tod(i) = tod(i) + sum(amps*phase(:namp,i))*exp(-0.5*r2)
						else
							! Project onto the amps part of the parameters
							oamps(:,si) = oamps(:,si) + tod(i)*phase(:namp,i)*exp(-0.5*r2)
						end if
					end do
				end do
			end do
		end do
		if(dir <= 0) params(3:2+namp,:) = oamps
	end subroutine

	subroutine srcmask2rangemask(srcmask, rangesets, offsets, rangemask)
		implicit none
		integer(4), intent(in) :: srcmask(:)
		integer(4), intent(in) :: rangesets(:), offsets(:,:)
		integer(4), intent(inout) :: rangemask(:)
		integer(4) :: si, di, oi, ri
		rangemask = 0
		do si = 1, size(srcmask)
			if(srcmask(si) .eq. 0) cycle
			do di = 1, size(offsets,1)-1
				do oi = offsets(di,si)+1, offsets(di+1,si)
					ri = rangesets(oi)+1
					rangemask(ri) = 1
				end do
			end do
		end do
	end subroutine

	subroutine rangesub(tod1, tod2, ranges, rangemask)
		implicit none
		real(_), intent(inout) :: tod1(:)
		real(_), intent(in)    :: tod2(:)
		integer(4), intent(in) :: ranges(:,:)
		integer(4), intent(in) :: rangemask(:)
		integer(4) :: ri, i1, i2
		do ri = 1, size(ranges,2)
			if(rangemask(ri) .eq. 0) cycle
			i1 = ranges(1,ri)+1; i2 = ranges(2,ri)
			tod1(i1:i2) = tod1(i1:i2) - tod2(i1:i2)
		end do
	end subroutine

	subroutine rangechisq(tod1, tod2, ranges, chisqs, rangemask)
		implicit none
		real(_),    intent(in)    :: tod1(:), tod2(:)
		real(8),    intent(inout) :: chisqs(:)
		integer(4), intent(in)    :: ranges(:,:)
		integer(4), intent(in)    :: rangemask(:)
		integer(4) :: ri, i1, i2, i
		do ri = 1, size(ranges,2)
			if(rangemask(ri) .eq. 0) cycle
			i1 = ranges(1,ri)+1; i2 = ranges(2,ri)
			chisqs(ri) = 0
			do i = i1, i2
				chisqs(ri) = chisqs(ri) + dble(tod1(i))*tod2(i)
			end do
		end do
	end subroutine

end module
