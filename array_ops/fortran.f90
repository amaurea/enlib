module array_ops

contains

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
		b(1,:) = b(1,:)/A(1,1,:)
		return
	end if
	err = 0
	do i = 1, n
		call dgesv(m, 1, A(:,:,i), m, piv, b(:,i), m, err)
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
	!$omp parallel do default(private) shared(A,b,nx,nc)
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
	!$omp parallel do default(private) shared(A,b,nx,nc)
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
	!$omp parallel default(private) shared(A, nums, n, m, badval, lwork)
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
	!$omp parallel default(private) shared(A, nums, n, m, badval, lwork)
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

end module
