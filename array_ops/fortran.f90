module array_ops

contains

subroutine matmul_32(A, b)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	implicit none
	real(4), intent(in)    :: A(:,:,:)
	real(4), intent(inout) :: b(:,:,:)
	real(4)    :: x(size(b,1),size(b,2))
	integer(4) :: i
	!$omp parallel do private(i,x)
	do i = 1, size(A,3)
		x = b(:,:,i)
		b(:,:,i) = matmul(A(:,:,i),x)
	end do
end subroutine

subroutine matmul_64(A, b)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	implicit none
	real(8), intent(in)    :: A(:,:,:)
	real(8), intent(inout) :: b(:,:,:)
	real(8)    :: x(size(b,1),size(b,2))
	integer(4) :: i
	!$omp parallel do private(i,x)
	do i = 1, size(A,3)
		x = b(:,:,i)
		b(:,:,i) = matmul(A(:,:,i),x)
	end do
end subroutine

subroutine matmul_c64(A, b)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	implicit none
	complex(4), intent(in)    :: A(:,:,:)
	complex(4), intent(inout) :: b(:,:,:)
	complex(4)    :: x(size(b,1),size(b,2))
	integer(4) :: i
	!$omp parallel do private(i,x)
	do i = 1, size(A,3)
		x = b(:,:,i)
		b(:,:,i) = matmul(A(:,:,i),x)
	end do
end subroutine

subroutine matmul_c128(A, b)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	implicit none
	complex(8), intent(in)    :: A(:,:,:)
	complex(8), intent(inout) :: b(:,:,:)
	complex(8)    :: x(size(b,1),size(b,2))
	integer(4) :: i
	!$omp parallel do private(i,x)
	do i = 1, size(A,3)
		x = b(:,:,i)
		b(:,:,i) = matmul(A(:,:,i),x)
	end do
end subroutine

! Functions for solving large sets of small systems, along the first dimensions
subroutine solve_multi_32(A, b)
	implicit none
	real(4), intent(in)    :: A(:,:,:)
	real(4), intent(inout) :: b(:,:)
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
		call sgesv(m, 1, A(:,:,i), m, piv, b(:,i), m, err)
	end do
end subroutine

subroutine solve_multi_64(A, b)
	implicit none
	real(8), intent(in)    :: A(:,:,:)
	real(8), intent(inout) :: b(:,:)
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
		call dgesv(m, 1, A(:,:,i), m, piv, b(:,i), m, err)
	end do
end subroutine

! Functions for solving large sets of small systems, along the first dimensions
subroutine solve_multi_c64(A, b)
	implicit none
	complex(4), intent(in)    :: A(:,:,:)
	complex(4), intent(inout) :: b(:,:)
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
		call sgesv(m, 1, A(:,:,i), m, piv, b(:,i), m, err)
	end do
end subroutine

subroutine solve_multi_c128(A, b)
	implicit none
	complex(8), intent(in)    :: A(:,:,:)
	complex(8), intent(inout) :: b(:,:)
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
		call zgesv(m, 1, A(:,:,i), m, piv, b(:,i), m, err)
	end do
end subroutine

subroutine solve_masked_32(A, b)
	implicit none
	real(4), intent(in)    :: A(:,:,:)
	real(4), intent(inout) :: b(:,:)
	! Work
	real(4)    :: Awork(size(A,1),size(A,2)), bwork(size(A,1)), work(size(A,1)*size(A,1))
	integer(4) :: x, ny, nx, i, j, ni, nj, nc, err, piv(size(A,1))
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
		call ssysv('L', nj, 1, Awork, nc, piv, bwork, nc, work, size(work), err)
		nj = 0
		do j = 1, nc
			if(A(j,j,x) .ne. 0) then
				nj = nj+1
				b(j,x) = bwork(nj)
			end if
		end do
	end do
end subroutine

subroutine solve_masked_64(A, b)
	implicit none
	real(8), intent(in)    :: A(:,:,:)
	real(8), intent(inout) :: b(:,:)
	! Work
	real(8)    :: Awork(size(A,1),size(A,2)), bwork(size(A,1)), work(size(A,1)*size(A,1))
	integer(4) :: x, ny, nx, i, j, ni, nj, nc, err, piv(size(A,1))
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
		call dsysv('L', nj, 1, Awork, nc, piv, bwork, nc, work, size(work), err)
		nj = 0
		do j = 1, nc
			if(A(j,j,x) .ne. 0) then
				nj = nj+1
				b(j,x) = bwork(nj)
			end if
		end do
	end do
end subroutine

subroutine solve_masked_c64(A, b)
	implicit none
	complex(4), intent(in)    :: A(:,:,:)
	complex(4), intent(inout) :: b(:,:)
	! Work
	complex(4)    :: Awork(size(A,1),size(A,2)), bwork(size(A,1)), work(size(A,1)*size(A,1))
	integer(4) :: x, ny, nx, i, j, ni, nj, nc, err, piv(size(A,1))
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
		call csysv('L', nj, 1, Awork, nc, piv, bwork, nc, work, size(work), err)
		nj = 0
		do j = 1, nc
			if(A(j,j,x) .ne. 0) then
				nj = nj+1
				b(j,x) = bwork(nj)
			end if
		end do
	end do
end subroutine

subroutine solve_masked_c128(A, b)
	implicit none
	complex(8), intent(in)    :: A(:,:,:)
	complex(8), intent(inout) :: b(:,:)
	! Work
	complex(8)    :: Awork(size(A,1),size(A,2)), bwork(size(A,1)), work(size(A,1)*size(A,1))
	integer(4) :: x, ny, nx, i, j, ni, nj, nc, err, piv(size(A,1))
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
		call zsysv('L', nj, 1, Awork, nc, piv, bwork, nc, work, size(work), err)
		nj = 0
		do j = 1, nc
			if(A(j,j,x) .ne. 0) then
				nj = nj+1
				b(j,x) = bwork(nj)
			end if
		end do
	end do
end subroutine

subroutine condition_number_multi_32(A, nums)
	implicit none
	real(4), intent(in)  :: A(:,:,:)
	real(4), intent(inout) :: nums(:)
	real(4)              :: eigs(size(A,1)), badval, tmp(1), Acopy(size(A,1),size(A,2))
	real(4), allocatable :: work(:)
	integer(4) :: i, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	call ssyev('N', 'U', m, A(:,:,1), m, eigs, tmp, -1, info)
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
			call ssyev('N', 'U', m, Acopy, m, eigs, work, lwork, info)
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

subroutine condition_number_multi_64(A, nums)
	implicit none
	real(8), intent(in)  :: A(:,:,:)
	real(8), intent(inout) :: nums(:)
	real(8)              :: eigs(size(A,1)), badval, tmp(1), Acopy(size(A,1),size(A,2))
	real(8), allocatable :: work(:)
	integer(4) :: i, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	call dsyev('N', 'U', m, A(:,:,1), m, eigs, tmp, -1, info)
	lwork  = int(tmp(1))
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
			call dsyev('N', 'U', m, Acopy, m, eigs, work, lwork, info)
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

subroutine eigpow_32(A, pow)
	implicit none
	real(4), intent(inout) :: A(:,:,:)
	real(4) :: eigs(size(A,1)), vecs(size(A,1),size(A,2))
	real(4) :: tmp(1), pow, tmp2(size(A,1),size(A,2))
	real(4), parameter   :: lim = 1e-6, lim0 = 1e-25
	real(4), allocatable :: work(:)
	integer(4) :: i, j, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	! Workspace query
	call ssyev('V', 'U', m, A(:,:,1), m, eigs, tmp, -1, info)
	lwork = int(tmp(1))
	allocate(work(lwork))
	do i = 1, n
		vecs = A(:,:,i)
		call ssyev('V', 'U', m, vecs, m, eigs, work, lwork, info)
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
			call sgemm('N','T', m, m, m, 1e0, tmp2, m, vecs, m, 0e0, A(:,:,i), m)
		end if
	end do
	deallocate(work)
end subroutine

subroutine eigpow_64(A, pow)
	implicit none
	real(8), intent(inout) :: A(:,:,:)
	real(8) :: eigs(size(A,1)), vecs(size(A,1),size(A,2))
	real(8) :: tmp(1), pow, tmp2(size(A,1),size(A,2))
	real(8), parameter   :: lim = 1d-9, lim0 = 1d-50
	real(8), allocatable :: work(:)
	integer(4) :: i, j, n, m, lwork, info
	n = size(A,3)
	m = size(A,1)
	! Workspace query
	call dsyev('V', 'U', m, A(:,:,1), m, eigs, tmp, -1, info)
	lwork = int(tmp(1))
	allocate(work(lwork))
	do i = 1, n
		vecs = A(:,:,i)
		call dsyev('V', 'U', m, vecs, m, eigs, work, lwork, info)
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
			call dgemm('N','T', m, m, m, 1d0, tmp2, m, vecs, m, 0d0, A(:,:,i), m)
		end if
	end do
	deallocate(work)
end subroutine

end module
