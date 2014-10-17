module fortran
	implicit none

contains

	! Apply a simple white noise + mean subtraction noise model to a subrange tod with
	! ranges specified by offsets[src,det], ranges[nrange,2].
	subroutine nmat_mwhite(tod, ranges, rangesets, offsets, ivar, submean)
		implicit none
		real(_), intent(inout) :: tod(:)
		real(_)    :: ivar(:), submean
		integer(4) :: offsets(:,:), ranges(:,:), rangesets(:), r2det(size(ranges,2))
		integer(4) :: si, di, ri, nsrc, ndet, oi,  i1, i2
		real(_)    :: mean
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

		!$omp parallel do private(ri,i1,i2,mean)
		do ri = 1, size(ranges,2)
			i1 = ranges(1,ri)+1
			i2 = ranges(2,ri)
			if(i2-i1 < 0) cycle
			mean = 0
			if(submean .ne. 0) mean = submean*sum(tod(i1:i2))/(i2-i1)
			tod(i1:i2) = (tod(i1:i2)-mean)*ivar(r2det(ri))
		end do
	end subroutine

	subroutine pmat_thumbs(dir, tod, maps, point, phase, boxes)
		implicit none
		integer(4), intent(in)    :: dir
		real(_),    intent(inout) :: tod(:), maps(:,:,:,:)
		real(_),    intent(in)    :: point(:,:), phase(:,:), boxes(:,:,:)
		integer(4) :: i,j,p(2), nsamp, n(2), nmap, ncomp
		nsamp = size(tod)
		! map is (nx,ny,nc,nm). boxes is ({dec,ra},2,nm), point is ({dec,ra},n)
		! we want ra = x, dec = y, so some indices will be transposed
		n(1) = size(maps,2)
		n(2) = size(maps,1)
		ncomp= size(maps,3)
		nmap = size(maps,4)
		do i = 1, nsamp
			! Let's do this the naive way for now
			do j = 1, nmap
				p = floor((point(:,i)-boxes(:,1,j))*n/(boxes(:,2,j)-boxes(:,1,j)))+1
				if(any(p<=0) .or. any(p>n)) cycle
				if(dir < 0) then
					maps(p(2),p(1),:,j) = maps(p(2),p(1),:,j) + tod(i)*phase(1:ncomp,i)
				else
					tod(i) = sum(maps(p(2),p(1),:,j)*phase(1:ncomp,i))
				end if
			end do
		end do
	end subroutine

end module
