! Cuts are represented as ranges[nrange,{from,to}], detmap[ndet+1]
! such that the cuts for det i are in ranges[detmap[i]:detmap[i+1]]

module fortran

contains

	! Given two sets of (sorted) detector cuts, stack them into one
	! set of (sorted) detector cuts, without merging any overlapping
	! cuts etc. So the total number of cut regions in the output will
	! equal the sum of the input cut regions. Sorted here refers to the
	! starting point of each cut range. It also removes empty ranges.
	subroutine cut_stack(ranges1, detmap1, ranges2, detmap2, oranges, odetmap)
		implicit none
		integer, intent(in)    :: ranges1(:,:), ranges2(:,:), detmap1(:), detmap2(:)
		integer, intent(inout) :: oranges(:,:), odetmap(:)
		integer :: di, ndet, oi, r(2), i1, i2
		ndet = size(detmap1)-1
		oi   = 0
		odetmap(1) = 0
		do di = 1, ndet
			i1 = detmap1(di)+1
			i2 = detmap2(di)+1
			do while(i1 <= detmap1(di+1) .or. i2 <= detmap2(di+1))
				if(i1 > detmap1(di+1)) then
					r  = ranges2(:,i2)
					i2 = i2+1
				elseif(i2 > detmap2(di+1)) then
					r  = ranges1(:,i1)
					i1 = i1+1
				elseif(ranges1(1,i1) < ranges2(1,i2)) then
					r  = ranges1(:,i1)
					i1 = i1+1
				else
					r  = ranges2(:,i2)
					i2 = i2+1
				end if
				if(r(2) <= r(1)) cycle
				oi = oi + 1
				oranges(:,oi) = r
			end do
			odetmap(di+1) = oi
		end do
	end subroutine

	! Given a set of (sorted) detector cuts. Merge overlapping cuts
	! into their union. The result will be no longer than the input.
	subroutine cut_union(iranges, idetmap, oranges, odetmap)
		implicit none
		integer, intent(in)    :: iranges(:,:), idetmap(:)
		integer, intent(inout) :: oranges(:,:), odetmap(:)
		integer :: di, ndet, oi, r(2), i
		ndet = size(idetmap)-1
		oi   = 0
		odetmap(1) = 0
		do di = 1, ndet
			i = idetmap(di)+1
			do while(i <= idetmap(di+1))
				! Start a new output range
				r = iranges(:,i)
				! Grow it as long as we find overlapping ranges
				do i = i+1, idetmap(di+1)
					if(iranges(1,i) > r(2)) exit
					r(2) = max(r(2),iranges(2,i))
				end do
				! No more overlap, so output it
				oi = oi + 1
				oranges(:,oi) = r
			end do
			odetmap(di+1) = oi
		end do
	end subroutine

	! Restrict detector cuts to a detector subset
	subroutine cut_detslice(iranges, idetmap, detinds, oranges, odetmap)
		implicit none
		integer, intent(in)    :: iranges(:,:), idetmap(:), detinds(:)
		integer, intent(inout) :: oranges(:,:), odetmap(:)
		integer :: oi, odi, idi, i
		odetmap(1) = 0
		oi = 0
		do odi = 1, size(detinds)
			idi = detinds(odi)+1
			do i = idetmap(idi)+1, idetmap(idi+1)
				oi = oi + 1
				oranges(:,oi) = iranges(:,i)
			end do
			odetmap(odi+1) = oi
		end do
	end subroutine

	! Apply the same sample range slice[{from,to,step}] for each detector.
	! Downsampling using step>1 is done inclusively, so that no cut data
	! leaks through.
	subroutine cut_sampslice(iranges, idetmap, slice, oranges, odetmap)
		implicit none
		integer, intent(in)    :: iranges(:,:), idetmap(:), slice(:)
		integer, intent(inout) :: oranges(:,:), odetmap(:)
		integer :: oi, di, i, from, to, step, r(2), tmp, tmp2(2), nsamp
		logical :: flip
		from = slice(1); to = slice(2); step = slice(3)
		flip = step < 0
		if(flip) then
			tmp = from; from = to+1; to = tmp+1
			step = -step
		end if
		nsamp = (to-from+step-1)/step
		odetmap(1) = 0
		oi = 0
		do di = 1, size(idetmap)-1
			! No ranges for this detector by default
			odetmap(di+1) = odetmap(di)
			! Add all ranges that fall within the remaining
			! samples
			do i = idetmap(di)+1, idetmap(di+1)
				r = iranges(:,i)
				r = r - from
				r(1) = max(r(1),0)
				r(2) = min(r(2),to-from)
				if(r(2) <= r(1)) cycle
				! Round start down and end up
				r(1) = r(1) / step
				r(2) = (r(2)+step-1)/step
				oi   = oi + 1
				oranges(:,oi) = r
				odetmap(di+1) = oi
			end do
			! Flip order of ranges if we used negative slice
			if(flip) then
				do i = 1, (odetmap(di+1)-odetmap(di)+1)/2
					tmp2 = oranges(:,odetmap(di)+i)
					oranges(:,odetmap(di)+i) = oranges(:,odetmap(di+1)-i+1)
					oranges(:,odetmap(di+1)-i+1) = tmp2
				end do
			end if
		end do
		! And count from the right side
		if(flip) then
			do i = 1, odetmap(size(odetmap))
				tmp = oranges(1,i)
				oranges(1,i) = nsamp-1-oranges(2,i)
				oranges(2,i) = nsamp-1-tmp
			end do
		end if
	end subroutine

	! Given cuts for N detectors, produce cuts for N*ncopy detectors
	! by repeating the input cuts ncopy times.
	subroutine cut_mul(iranges, idetmap, ncopy, oranges, odetmap)
		implicit none
		integer, intent(in)    :: iranges(:,:), idetmap(:), ncopy
		integer, intent(inout) :: oranges(:,:), odetmap(:)
		integer :: i, di, di2, k, k2
		di2 = 0; k2 = 0
		odetmap(1) = 0
		do i = 1, ncopy
			do di = 1, size(idetmap)-1
				di2 = di2+1
				do k = idetmap(di)+1, idetmap(di+1)
					k2 = k2 + 1
					oranges(:,k2) = iranges(:,k)
				end do
				odetmap(di2+1) = k2
			end do
		end do
	end subroutine

	! Count the number of cut samples per detector
	subroutine cut_nsamp(ranges, detmap, ncut)
		implicit none
		integer, intent(in)    :: ranges(:,:), detmap(:)
		integer, intent(inout) :: ncut(:)
		integer :: i, j
		ncut = 0
		do i = 1, size(detmap)-1
			do j = detmap(i)+1, detmap(i+1)
				ncut(i) = ncut(i) + ranges(2,j)-ranges(1,j)
			end do
		end do
	end subroutine

	! Invert a cut, so that cut samples become uncut, and vice
	! versa.
	subroutine cut_invert(iranges, idetmap, nsamp, oranges, odetmap)
		implicit none
		integer, intent(in)    :: iranges(:,:), idetmap(:), nsamp
		integer, intent(inout) :: oranges(:,:), odetmap(:)
		integer :: di, j, j2, pend
		odetmap(1) = 0
		j2 = 0
		do di = 1, size(idetmap)-1
			! Cut everything between the end of the previous cut and
			! the start of this cut
			pend = 0
			do j = idetmap(di)+1, idetmap(di+1)
				if(iranges(1,j) > pend) then
					j2 = j2+1
					oranges(1,j2) = pend
					oranges(2,j2) = iranges(1,j)
				end if
				pend = iranges(2,j)
			end do
			! Cut the remaining samples towards the end
			if(pend < nsamp) then
				j2 = j2+1
				oranges(1,j2) = pend
				oranges(2,j2) = nsamp
			end if
			odetmap(di+1) = j2
		end do
	end subroutine

	! Copy out cut samples from a tod, and store them
	! in 1d array samps, which must be long enough.
	subroutine cut_extract(ranges, detmap, tod, samps)
		implicit none
		integer, intent(in) :: ranges(:,:), detmap(:)
		real(_), intent(in) :: tod(:,:)
		real(_), intent(inout) :: samps(:)
		integer :: i, j, k1, k2
		k2 = 0
		do i = 1, size(detmap)-1
			do j = detmap(i)+1, detmap(i+1)
				do k1 = ranges(1,j)+1, ranges(2,j)
					k2 = k2+1
					samps(k2) = tod(k1,i)
				end do
			end do
		end do
	end subroutine

	! Inverse of cut_extract. Copies back samples from samps into tod.
	subroutine cut_insert(ranges, detmap, tod, samps)
		implicit none
		integer, intent(in)    :: ranges(:,:), detmap(:)
		real(_), intent(inout) :: tod(:,:)
		real(_), intent(in)    :: samps(:)
		integer :: i, j, k1, k2
		k2 = 0
		do i = 1, size(detmap)-1
			do j = detmap(i)+1, detmap(i+1)
				do k1 = ranges(1,j)+1, ranges(2,j)
					k2 = k2+1
					tod(k1,i) = samps(k2)
				end do
			end do
		end do
	end subroutine

	subroutine gapfill_const(ranges, detmap, tod, const)
		implicit none
		integer, intent(in)    :: ranges(:,:), detmap(:)
		real(_), intent(inout) :: tod(:,:)
		real(_), intent(in)    :: const
		integer :: di, i, r(2)
		do di = 1, size(detmap)-1
			do i = detmap(di)+1, detmap(di+1)
				r = ranges(:,i)+1
				tod(r(1):r(2),di) = const
			end do
		end do
	end subroutine

	subroutine gapfill_linear(ranges, detmap, tod, context)
		implicit none
		integer, intent(in)    :: ranges(:,:), detmap(:), context
		real(_), intent(inout) :: tod(:,:)
		real(_) :: v1, v2, c1, c2
		integer :: di, i, j, r0, r1, r2, r3, nsamp
		nsamp = size(tod,1)
		do di = 1, size(detmap)-1
			do i = detmap(di)+1, detmap(di+1)
				! cut range
				r1 = max(0,    ranges(1,i))+1 ! first cut index
				r2 = min(nsamp,ranges(2,i))+1 ! first uncut index
				if(r1 <= 1 .and. r2 >= nsamp+1) then
					tod(:,di) = 0
					cycle
				end if
				! We need some samples of context on either side to do the
				! linear fill. But make sure we don't use any other cut samples
				! in the context, so avoid the previous and next cut edges.
				! prev cut end
				if(i == detmap(di)+1) then; r0 = 1; else; r0 = ranges(2,i-1)+1; end if
				! next cut start
				if(i == detmap(di+1)) then; r3 = nsamp+1; else; r3 = min(nsamp,ranges(1,i+1))+1; end if
				r0 = max(r0, r1-context)
				r3 = min(r3, r2+context)
				! Now calculate the representive value on either side
				v1 = sum(tod(r0:r1-1,di))/(r1-r0)
				v2 = sum(tod(r2:r3-1,di))/(r3-r2)
				if(r0 == r1) then
					tod(r1:r2-1,di) = v2
				elseif(r2 == r3) then
					tod(r1:r2-1,di) = v1
				else
					! v1 has a center of mass at (r0+r1-1)/2 instead of r1 where we
					! want it.
					c1 = (r0+r1-1)/2d0
					c2 = (r2+r3-1)/2d0
					do j = r1, r2-1
						tod(j,di) = v1 + (v2-v1)*(j-c1)/(c2-c1)
					end do
				end if
			end do
		end do
	end subroutine

	function count_mask(mask) result(n)
		implicit none
		integer(1), intent(in) :: mask(:,:)
		integer :: n, di, i
		n = 0
		!$omp parallel do private(di,i) reduction(+:n)
		do di = 1, size(mask,2)
			if(mask(1,di) .ne. 0) n = n+1
			do i = 2, size(mask,1)
				if(mask(i,di) .ne. 0) then
					if(mask(i-1,di) .eq. 0) n = n+1
				end if
			end do
		end do
	end function

	subroutine mask_to_cut(mask, oranges, odetmap)
		implicit none
		integer(1), intent(in)    :: mask(:,:)
		integer,    intent(inout) :: oranges(:,:), odetmap(:)
		logical :: incut
		integer :: di, i, j
		j = 0
		do di = 1, size(mask,2)
			odetmap(di) = j
			incut = .false.
			do i = 1, size(mask,1)
				if(.not. incut .and. mask(i,di) .ne. 0) then
					j = j+1
					oranges(1,j) = i-1
					incut = .true.
				elseif(incut .and. mask(i,di) .eq. 0) then
					oranges(2,j) = i-1
					incut = .false.
				end if
			end do
			if(incut) then
				oranges(2,j) = size(mask,1)
			end if
		end do
		odetmap(size(mask,2)+1) = j
	end subroutine

	subroutine cut_to_mask(ranges, detmap, mask)
		implicit none
		integer(1), intent(inout) :: mask(:,:)
		integer,    intent(in)    :: ranges(:,:), detmap(:)
		integer :: di, j
		!$omp parallel workshare
		mask = 0
		!$omp end parallel workshare
		!$omp parallel do private(j)
		do di = 1, size(detmap)-1
			do j = detmap(di)+1, detmap(di+1)
				mask(ranges(1,j)+1:ranges(2,j),di) = 1
			end do
		end do
	end subroutine

end module

