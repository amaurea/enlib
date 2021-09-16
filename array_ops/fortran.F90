module array_ops

contains

! T: data type
! _: data precision
! C: lapack dtype char
! L, L0: eigpow limits
! ONE, ZERO: gemm constants
! SY: matrix form

subroutine matmul_multi_sym(A, b)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	implicit none
	T(_), intent(in)    :: A(:,:,:)
	T(_), intent(inout) :: b(:,:,:)
	T(_)    :: x(size(b,1),size(b,2))
	integer(4) :: i
	!$omp parallel do private(i,x)
	do i = 1, size(A,3)
		x = b(:,:,i)
		b(:,:,i) = matmul(A(:,:,i),x)
	end do
end subroutine

subroutine matmul_multi(A, b, x)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	implicit none
	T(_), intent(in)    :: A(:,:,:)
	T(_), intent(in)    :: b(:,:,:)
	T(_), intent(inout) :: x(:,:,:)
	integer(4) :: i
	!$omp parallel do private(i)
	do i = 1, size(A,3)
		x(:,:,i) = matmul(transpose(A(:,:,i)),b(:,:,i))
	end do
end subroutine

! Functions for solving large sets of small systems, along the first dimensions
subroutine solve_multi(A, b)
	implicit none
	T(_), intent(in)    :: A(:,:,:)
	T(_), intent(inout) :: b(:,:)
	integer(4) :: i, n, m, piv(size(b,1)), err
	n = size(A,3); m = size(A,1)
	if(m == 1) then
		!$omp parallel workshare
		b(1,:) = b(1,:)/A(1,1,:)
		!$omp end parallel workshare
		return
	end if
	err = 0
	do i = 1, n
		call C##gesv(m, 1, A(:,:,i), m, piv, b(:,i), m, err)
	end do
end subroutine

subroutine solve_masked(A, b)
	implicit none
	T(_), intent(in)    :: A(:,:,:)
	T(_), intent(inout) :: b(:,:)
	! Work
	T(_)    :: Awork(size(A,1),size(A,2)), bwork(size(A,1)), work(size(A,1)*size(A,1))
	integer(4) :: x, nx, i, j, ni, nj, nc, err, piv(size(A,1))
	nx = size(A,3); nc = size(A,1)
	!$omp parallel do private(x,nj,j,Awork,bwork,ni,i,err,piv,work)
	do x = 1, nx
		nj = 0
		do j = 1, nc
			if(A(j,j,x) .ne. 0) then
				nj = nj+1
				Awork(nj,nj) = A(j,j,x)
				bwork(nj)    = b(j,  x)
				ni = nj
				do i = j+1, nc
					if(A(i,i,x) .ne. 0) then
						ni = ni+1
						Awork(ni,nj) = A(i,j,x)
					end if
				end do
			end if
		end do
		err = 0
		call C##sysv('l', nj, 1, Awork, nc, piv, bwork, nc, work, size(work), err)
		nj = 0
		do j = 1, nc
			if(A(j,j,x) .ne. 0) then
				nj = nj+1
				b(j,x) = bwork(nj)
			end if
		end do
	end do
end subroutine

subroutine condition_number_multi(A, nums)
	implicit none
	T(_), intent(inout) :: A(:,:,:)
	real(_), intent(inout) :: nums(:)
	real(_) :: eigs(size(A,1)), meig, badval, rwork(size(A,1)*3-2)
	T(_) :: vecs(size(A,1),size(A,2)), tmp2(size(A,1),size(A,2)), tmp(1)
	T(_), allocatable :: work(:)
	integer(4) :: i, j, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	! Workspace query
	call C##SY##ev('v', 'u', m, A(:,:,1), m, eigs, tmp, -1, R, info)
	lwork = int(tmp(1))
	info   = 0
	badval = 1d0/info
	!$omp parallel private(work,i,vecs,tmp2,info,eigs,j,meig)
	allocate(work(lwork))
	!$omp do
	do i = 1, n
		if(all(A(:,:,i) == 0)) then
			nums(i) = badval
		else
			vecs = A(:,:,i)
			call C##SY##ev('v', 'u', m, vecs, m, eigs, work, lwork, R, info)
			nums(i) = maxval(eigs)/max(0d0,minval(eigs))
		end if
	end do
	deallocate(work)
	!$omp end parallel
end subroutine

subroutine eigpow(A, pow, lim, lim0)
	implicit none
	T(_), intent(inout) :: A(:,:,:)
	real(_), intent(in) :: pow, lim, lim0
	real(_) :: eigs(size(A,1)), rwork(size(A,1)*3-2), meig
	T(_) :: vecs(size(A,1),size(A,2)), tmp2(size(A,1),size(A,2)), tmp(1)
	T(_), allocatable :: work(:)
	integer(4) :: i, j, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	! Workspace query
	call C##SY##ev('v', 'u', m, A(:,:,1), m, eigs, tmp, -1, R, info)
	lwork = int(tmp(1))
	!$omp parallel private(work,i,vecs,tmp2,info,eigs,j,rwork,meig)
	allocate(work(lwork))
	!$omp do
	do i = 1, n
		vecs = A(:,:,i)
		call C##SY##ev('v', 'u', m, vecs, m, eigs, work, lwork, R, info)
		if(maxval(eigs) <= lim0) then
			A(:,:,i) = 0
		else
			meig = lim*maxval(eigs)
			do j = 1, m
				if(eigs(j) < meig) then
					tmp2(:,j) = 0
				else
					tmp2(:,j) = vecs(:,j) * eigs(j)**pow
				end if
			end do
			call C##gemm('n','c', m, m, m, ONE, tmp2, m, vecs, m, ZERO, A(:,:,i), m)
		end if
	end do
	deallocate(work)
	!$omp end parallel
end subroutine

! This doesn't make sense

!subroutine svdpow(A, pow, lim, lim0)
!	implicit none
!	T(_), intent(inout) :: A(:,:,:)
!	real(_), intent(in) :: pow, lim, lim0
!	real(_) :: sigma(size(A,1)), rwork(size(A,1)*5), slim
!	T(_) :: uvecs(size(A,1),size(A,1)), vvecs(size(A,1),size(A,1)), tmp(1)
!	T(_), allocatable :: work(:)
!	integer(4) :: i, j, n, m, lwork, info
!	n = size(A,3)
!	m = size(A,1)
!	! Workspace query
!	call C##GE##svd('o', 'a', m, m, uvecs, m, sigma, uvecs, m, vvecs, m, tmp, -1, R, info)
!	lwork = int(tmp(1))
!	!$omp parallel private(work,i,uvecs,vvecs,info,sigma,j,rwork)
!	allocate(work(lwork))
!	!$omp do
!	do i = 1, n
!		uvecs = A(:,:,i)
!		call C##GE##svd('o', 'a', m, m, uvecs, m, sigma, uvecs, m, vvecs, m, work, lwork, R, info)
!		if(maxval(sigma) <= lim0) then
!			A(:,:,i) = 0
!		else
!			slim = lim*maxval(sigma)
!			do j = 1, m
!				if(sigma(j) < slim) then
!					uvecs(:,j) = 0
!				else
!					uvecs(:,j) = uvecs(:,j) * sigma(j)**pow
!				end if
!			end do
!			call C##gemm('n','n', m, m, m, ONE, uvecs, m, vvecs, m, ZERO, A(:,:,i), m)
!		end if
!	end do
!	deallocate(work)
!	!$omp end parallel
!end subroutine

subroutine eigflip(A)
	implicit none
	T(_), intent(inout) :: A(:,:,:)
	real(_)    :: eigs(size(A,1)), rwork(size(A,1)*3-2)
	T(_) :: vecs(size(A,1),size(A,2)), tmp2(size(A,1),size(A,2)), tmp(1)
	T(_), allocatable :: work(:)
	integer(4) :: i, j, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	! Workspace query
	call C##SY##ev('v', 'u', m, A(:,:,1), m, eigs, tmp, -1, R, info)
	lwork = int(tmp(1))
	!$omp parallel private(work,i,vecs,tmp2,info,eigs,j,rwork)
	allocate(work(lwork))
	!$omp do
	do i = 1, n
		vecs = A(:,:,i)
		call C##SY##ev('v', 'u', m, vecs, m, eigs, work, lwork, R, info)
		where(eigs < 0)
			eigs = -eigs
		end where
		do j = 1, m
			tmp2(:,j) = vecs(:,j)*eigs(j)
		end do
		call C##gemm('n','c', m, m, m, ONE, tmp2, m, vecs, m, ZERO, A(:,:,i), m)
	end do
	deallocate(work)
	!$omp end parallel
end subroutine

! This subroutine is like eigpow, except that instead of
! setting too small eigenvalues to zero, it zeros all but
! the [0,0] entry of the matrix when the matrix is poorly
! conditioned. This is a pretty weird operation that's
! mostly useful for making the binned preconditioner safe
! along the edges of the map.
subroutine eigpow_scalar_fallback(A, pow, lim, lim0)
	implicit none
	T(_), intent(inout) :: A(:,:,:)
	real(_), intent(in) :: pow, lim, lim0
	real(_) :: eigs(size(A,1)), rwork(size(A,1)*3-2), meig
	T(_) :: vecs(size(A,1),size(A,2)), tmp2(size(A,1),size(A,2)), tmp(1)
	real(_) :: vmax, vmin
	T(_), allocatable :: work(:)
	integer(4) :: i, j, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	! Workspace query
	call C##SY##ev('v', 'u', m, A(:,:,1), m, eigs, tmp, -1, R, info)
	lwork = int(tmp(1))
	!$omp parallel private(work,i,vecs,tmp2,info,eigs,j,rwork,meig,vmax,vmin)
	allocate(work(lwork))
	!$omp do
	do i = 1, n
		vecs = A(:,:,i)
		call C##SY##ev('v', 'u', m, vecs, m, eigs, work, lwork, R, info)
		vmax = maxval(eigs)
		if(vmax <= lim0) then
			A(:,:,i) = 0
		else
			vmin = minval(eigs)
			if(vmin < 0 .or. vmax*lim/vmin > 1 .and. pow < 0) then
				! Matrix is bad, make it scalar
				A(:,:,i) = 0
				A(1,1,i) = vmax**pow
			else
				! Matrix is fine, use normal inversion
				do j = 1, m
					tmp2(:,j) = vecs(:,j) * eigs(j)**pow
				end do
				call C##gemm('n','c', m, m, m, ONE, tmp2, m, vecs, m, ZERO, A(:,:,i), m)
			end if
		end if
	end do
	deallocate(work)
	!$omp end parallel
end subroutine

subroutine measure_cov(d, cov, delay)
	implicit none
	T(_), intent(in) :: d(:,:)
	integer, intent(in) :: delay
	real(_), intent(inout) :: cov(:,:)
	T(_), allocatable :: tcov(:,:)
	T(_) :: norm
	allocate(tcov(size(cov,1),size(cov,2)))
	norm = ONE/(size(d,1)-delay)
	call C##gemm('c', 'n', size(d,2), size(d,2)-delay, size(d,1), norm, d(1+delay,1), size(d,1), d, size(d,1), ZERO, tcov, size(tcov,1))
	cov=tcov
end subroutine

subroutine ang2rect(ang, rect)
	implicit none
	T(_), intent(in)    :: ang(:,:)
	T(_), intent(inout) :: rect(:,:)
	T(_) :: st,ct,sp,cp
	integer :: i
	!$omp parallel do private(i,st,ct,sp,cp)
	do i = 1, size(ang,2)
		sp = sin(ang(1,i)); cp = cos(ang(1,i))
		st = sin(ang(2,i)); ct = cos(ang(2,i))
		rect(1,i) = cp*ct
		rect(2,i) = sp*ct
		rect(3,i) = st
	end do
end subroutine

! Find areas in imap where values cross from below to above each
! value in vals, which must be sorted in ascending order. omap
! will be 0 in pixels where no crossing happens, and i where
! a crossing for vals(i) happens.
subroutine find_contours(imap, vals, omap)
	implicit none
	real(_), intent(in) :: imap(:,:), vals(:)
	integer, intent(inout) :: omap(:,:)
	integer, allocatable   :: work(:,:)
	real(_) :: v
	integer :: y, x, ip, i, ny, nx, nv
	logical :: left, right
	ny = size(imap,1)
	nx = size(imap,2)
	nv = size(vals)
	allocate(work(ny,nx))
	do x = 1, nx
		do y = 1, ny
			ip = 1
			! Find which "bin" each value belongs in: 0 for for less
			! than vals(1), and so on
			v = imap(y,x)
			! nan is a pretty common case
			if(.not. (v .eq. v)) then
				work(y,x) = 1
				cycle
			end if
			left  = .true.
			right = .true.
			if(ip >   1) left  = v >= vals(ip-1)
			if(ip <= nv) right = v <  vals(ip)
			if(left .and. right) then
				i = ip
			else
				! Full search. No binary for now.
				do i = 1, nv
					if(v < vals(i)) exit
				end do
			end if
			work(y,x) = i
			ip = i
		end do
	end do
	! Edge detection
	omap = 0
	do x = 1, nx-1
		do y = 1, ny-1
			if(work(y,x) .ne. work(y+1,x)) then
				omap(y,x) = min(work(y,x),work(y+1,x))
			elseif(work(y,x) .ne. work(y,x+1)) then
				omap(y,x) = min(work(y,x),work(y,x+1))
			end if
		end do
	end do
end subroutine

subroutine maxbin(map, inds, vals)
	implicit none
	real(_), intent(inout) :: map(:)
	real(_), intent(in)    :: vals(:)
	integer, intent(in)    :: inds(:)
	integer :: i, j
	do i = 1, size(inds)
		j = inds(i)+1
		if(vals(i) > map(j)) map(j) = vals(i)
	end do
end subroutine

subroutine scale_rows(tod, det_scales)
	implicit none
	real(_), intent(inout) :: tod(:,:)
	real(_), intent(in)    :: det_scales(:)
	integer :: di, si
	!$omp parallel do collapse(2)
	do di = 1, size(tod,2)
		do si = 1, size(tod,1)
			tod(si,di) = tod(si,di) * det_scales(di)
		end do
	end do
end subroutine

end module
