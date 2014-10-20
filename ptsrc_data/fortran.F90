module fortran
	implicit none

contains

	! Apply a simple white noise + mean subtraction noise model to a subrange tod with
	! ranges specified by offsets[src,det], ranges[nrange,2].
	subroutine nmat_mwhite(tod, ranges, rangesets, offsets, ivar, detrend)
		implicit none
		real(_), intent(inout) :: tod(:)
		real(_)    :: ivar(:)
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi, i, i1, i2, detrend
		real(_)    :: m,s,x,sn,mid, foo(size(ivar),2)
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

		foo = 0
		!$omp parallel do private(ri,i1,i2,m,s,x,sn,mid) reduction(+:foo)
		do ri = 1, size(ranges,2)
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
			foo(r2det(ri),1) = foo(r2det(ri),1) + sum(tod(i1:i2)**2)
			foo(r2det(ri),2) = foo(r2det(ri),2) + i2-i1+1
			tod(i1:i2) = tod(i1:i2) * ivar(r2det(ri))
		end do

		if(detrend>0) then
			write(*,*)
			do di = 1, ndet
				if(foo(di,1)/foo(di,2)*ivar(di) > 4) then
					do ri = 1, size(ranges,2)
						if(r2det(ri) .ne. di) cycle
						i1 = ranges(1,ri)+1
						i2 = ranges(2,ri)
						do i = i1, i2
							write(*,'(i4,i9,f6.2,2f15.3)') di,i,foo(di,1)/foo(di,2)*ivar(di),tod(i)**2/ivar(di), tod(i)/ivar(di)
						end do
					end do
				end if
			end do
		end if

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

end module
