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
	T(_), intent(in)  :: A(:,:,:)
	real(_), intent(inout) :: nums(:)
	real(_)              :: eigs(size(A,1)), badval, tmp(1), rwork(size(A,1)*3-2)
	T(_)              :: Acopy(size(A,1),size(A,2))
	T(_), allocatable :: work(:)
	integer(4) :: i, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	call C##SY##ev('n', 'u', m, A(:,:,1), m, eigs, tmp, -1, R, info)
	lwork = int(tmp(1))
	! Generate +inf as badval, while hiding this from gfortran
	info   = 0
	badval = 1d0/info
	!$omp parallel private(i,Acopy,info,work,eigs)
	allocate(work(lwork))
	!$omp parallel do
	do i = 1, n
		if(all(A(:,:,i) == 0)) then
			nums(i) = badval
		else
			Acopy = A(:,:,i)
			call C##SY##ev('n', 'u', m, Acopy, m, eigs, work, lwork, R, info)
			if(info .ne. 0) then
				nums(i) = badval
			else
				nums(i) = eigs(m)/eigs(1)
			end if
		end if
	end do
	deallocate(work)
	!$omp end parallel
end subroutine

subroutine eigpow(A, pow)
	implicit none
	T(_), intent(inout) :: A(:,:,:)
	real(_)    :: eigs(size(A,1)), pow, rwork(size(A,1)*3-2)
	T(_) :: vecs(size(A,1),size(A,2)), tmp2(size(A,1),size(A,2)), tmp(1)
	real(_), parameter   :: lim = L, lim0 = L0
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
		if(maxval(eigs) <= lim0) then
			A(:,:,i) = 0
		else
			where(eigs < lim*maxval(eigs))
				eigs = 0
			elsewhere
				eigs = eigs**pow
			end where
			do j = 1, m
				tmp2(:,j) = vecs(:,j)*eigs(j)
			end do
			call C##gemm('n','c', m, m, m, ONE, tmp2, m, vecs, m, ZERO, A(:,:,i), m)
		end if
	end do
	deallocate(work)
	!$omp end parallel
end subroutine

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

subroutine measure_cov(d, cov)
	implicit none
	T(_), intent(in) :: d(:,:)
	real(_), intent(inout) :: cov(:,:)
	T(_), allocatable :: tcov(:,:)
	T(_) :: norm
	allocate(tcov(size(cov,1),size(cov,2)))
	norm = ONE/size(d,1)
	call C##gemm('c', 'n', size(d,2), size(d,2), size(d,1), norm, d, size(d,1), d, size(d,1), ZERO, tcov, size(tcov,1))
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


end module
