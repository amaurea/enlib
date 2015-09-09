module fortran

	private :: map_border, calc_weights

contains

pure function sinc(x)
	implicit none
	real(_), intent(in) :: x
	real(_) :: sinc, y
	real(_), parameter :: cut = 1e-4, pi = 3.14159265358979323846d0
	y = abs(pi*x)
	if(y < cut) then
		sinc = 1 - y**2/6 + y**4/120
	else
		sinc = sin(y)/y
	end if
end function

! Port of scipy.ndimage's interpolation, for the purpose of adding transposes.
! It has two main components. One is the so-called spline filter, which I think
! computes relevant B-spline coefficients. It is a 1d operation applied along
! all axes one after another. For a single 1d array a, its action is:
!
! N = sum_i p^i a_i
! a_0 = N
! for i>0: a_i = a_i + p a_{i-1}
! a_{n-1} = p/(p^2-1) * (a_{n-1}+p a_{n-2})
! for i<n-1: a_i = p (a_{i+1}-a_i)
!
! The transpose of an operation of the type 1:n a_i = A a_{i-1} + B a_i
! is n-1::-1 a_i = A a_{i+1} + B a_i. So the transpose of this series of
! operations is:
!
! for i>0: a_i = p (a_{i-1}-a_i)
! a[-2],a[-1] = a[-2]+p/(p^2-1)*p*a[-1], p/(p^2-1)*a[-1]
! for i<n-1: a_i = a_i + p a_{i+1}
! for i>0: a_i = a_i + p^i a_0
!
! p^0 p^1 p^2 p^3 ...
! 0   1   0   0   ...
! 0   0   1   0   ...
! 0   0   0   1   ...
!
! Transpose is
!
! p^0 0   0   0   ...
! p^1 1   0   0   ...
! p^2 0   1   0   ...
! p^3 0   0   1   ...
!
! What about the other case, where max >= len? Here
!  N = (a_0 + p^{n-1} a_{n-1} + sum((p^i+p^{n-1-i})a_i,0<i<n-1))/(1-p^{2*(n-1)})
! This changes the left column of the transposed matrix from p^i to
! c_i, where c_0 = q, c_{n-1} = p^{n-1}*q, c_i = (p^i+p^{n-1-i})*q,
! with q = 1/(1-p^{2*(n-1)}).
!
! Looping through multidimensional D given shape along given axis,
! given a one-dimensional view d. We need 3 loops:
!  1. offsets off noff
!  2. blocks  b   nblock
!  3. samples s   n
! ind = off + (s+b*n)*stride
!
! do off = 0, noff-1
!  do block = 0, nblock-1
!   do samp = 0, nsamp-1
!    i = off + (samp+block*nsamp)*noff

! Apply a 1d spline filter of the given order along
! the given axis of a flat view data of an array
! with shape dims. This is copied from scipy, which
! seems to implement the algorithm described here:
! http://users.fmrib.ox.ac.uk/~jesper/papers/future_readgroups/unser9302.pdf
! It assumes *mirrored* boundary conditions, not cyclic!
subroutine spline_filter1d(data, dims, axis, order, border, trans)
	implicit none
	real(_), intent(inout) :: data(:)
	integer, intent(in)    :: dims(:), axis, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: a(:)
	real(_), parameter     :: tolerance = 1d-15
	real(_) :: pole(2), p, weight, pn, p2n, ip, q, v
	integer :: ndim, nblock, noff, n, oi, bi, i, pind, m, npole, xi, pi1, pi2, dpi

	ndim   = size(dims)
	n      = dims(axis+1)
	nblock = product(dims(1:axis))
	noff   = product(dims(axis+2:ndim))

	select case(order)
		case(2); npole = 1; pole(1) = sqrt(8d0)-3
		case(3); npole = 1; pole(1) = sqrt(3d0)-2
		case(4); npole = 2
			pole(1) = sqrt(664d0 - sqrt(438976d0)) + sqrt(304d0) - 19d0
			pole(2) = sqrt(664d0 + sqrt(438976d0)) - sqrt(304d0) - 19d0
		case(5); npole = 2
			pole(1) = sqrt(67.5d0 - sqrt(4436.25d0)) + sqrt(26.25d0) - 6.5d0
			pole(2) = sqrt(67.5d0 + sqrt(4436.25d0)) - sqrt(26.25d0) - 6.5d0
		case default; return
	end select
	weight = product((1-pole(1:npole))*(1-1/pole(1:npole)))
	if(.not. trans) then
		pi1 = 1; pi2 = npole; dpi = 1
	else
		pi1 = npole; pi2 = 1; dpi =-1;
	end if
	!$omp parallel private(oi, bi, a, pind, p, m, pn, i, p2n, ip, q, v)
	allocate(a(n))
	!$omp do collapse(2)
	do oi = 0, noff-1
		do bi = 0, nblock-1
			a = data(oi+bi*n*noff+1:oi+(bi+1)*n*noff:noff)
			a = a * weight
			do pind = pi1, pi2, dpi
				p = pole(pind)
				m = ceiling(log(tolerance)/log(abs(p)))
				! This is a bit cryptic. It is a port of
				! scipy ni_interpolation.c:273.
				if(.not. trans) then
					! Compute starting element. This is essentially
					! for(i=1;i<m;i++) a[0] += 0.5*(a[i]+a[-i]) * p**i
					! it only looks complicated due to boundary conditions.
					! The standard version used mirrored boundary conditions,
					! which give a[-i] = a[i], but we also want constant,
					! nearest and cyclic conditions.
					pn = 1
					v  = 0
					do i = 0, m-1
						xi = map_border(border, n, i)+1
						if(xi > 0) v = v + pn*a(xi)
						xi = map_border(border, n,-i)+1
						if(xi > 0) v = v + pn*a(xi)
						pn = pn*p
					end do
					a(1) = v/2
					! Update the rest of the array
					do i = 2, n
						a(i) = a(i) + p*a(i-1)
					end do
					a(n) = p/(p**2-1)*(a(n)+p*a(n-1))
					do i = n-1, 1, -1
						a(i) = p*(a(i+1)-a(i))
					end do
				else
					a(1) = -p*a(1)
					do i = 2, n-1
						a(i) = p*(a(i-1)-a(i))
					end do
					a(n) = a(n) - a(n-1)
					a(n-1) = a(n-1)+p/(p**2-1)*p*a(n)
					a(n)   = p/(p**2-1)*a(n)
					do i = n-1, 1, -1
						a(i) = a(i) + p*a(i+1)
					end do
					pn = 1
					! Boundary condition
					v  = a(1)/2
					a(1) = 0
					do i = 0, m-1
						xi = map_border(border, n, i)+1
						if(xi > 0) a(xi) = a(xi) + v * pn
						xi = map_border(border, n,-i)+1
						if(xi > 0) a(xi) = a(xi) + v * pn
						pn = pn*p
					end do
				end if
			end do
			data(oi+bi*n*noff+1:oi+(bi+1)*n*noff:noff) = a
		end do
	end do
	deallocate(a)
	!$omp end parallel
end subroutine

function get_weight_length(type, order) result(n)
	implicit none
	integer :: type, order, n
	n = 0
	select case(type)
		case(0) ! convolution
			select case(order)
				case(0); n = 1
				case(1); n = 2
				case(3); n = 4
			end select
		case(1) ! spline
			n = order+1
		case(2) ! lanczos
			n = max(1,2*order)
	end select
end function

pure subroutine calc_weights(type, order, p, weights, off)
	implicit none
	integer, intent(in)    :: type, order
	integer, intent(inout) :: off(:)
	real(_), intent(in)    :: p(:)
	real(_), intent(inout) :: weights(:,:)
	integer :: ndim, nw, i, j
	real(_) :: x
	ndim = size(weights, 2)
	nw   = size(weights, 1)
	! Speed up nearest neighbor
	if(order == 0) then
		off = nint(p); weights = 1; return
	end if
	do i = 1, ndim
		off(i) = floor(p(i)-(nw-2)*0.5d0)
		do j = 1, nw
			x = abs(p(i)-(j-1)-off(i))
			weights(j,i) = 0
			select case(type)
				case(0) ! convolution
					select case(order)
						case(0); if(x < 0.5) weights(j,i) = 1
						case(1); if(x < 1.0) weights(j,i) = 1-x
						case(3)
							if    (x < 1) then; weights(j,i) =  1.5*x**3 - 2.5*x**2 + 1
							elseif(x < 2) then; weights(j,i) = -0.5*x**3 + 2.5*x**2 - 4*x + 2; end if
					end select
				case(1) ! spline
					select case(order)
						case(0); if(x < 0.5) weights(j,i) = 1
						case(1); if(x < 1.0) weights(j,i) = 1-x
						case(2)
							if    (x < 0.5) then; weights(j,i) = 0.75-x**2
							elseif(x < 1.5) then; weights(j,i) = 0.50*(1.5-x)**2; end if
						case(3)
							if    (x < 1.0) then; weights(j,i) = (x*x*(x-2)*3+4)/6
							elseif(x < 2.0) then; weights(j,i) = (2-x)**3/6; end if
						case(4)
							if    (x < 0.5) then; weights(j,i) = x**2 * (x**2 * 0.25-0.625)+115d0/192
							elseif(x < 1.5) then; weights(j,i) = x*(x*(x*(5d0/6-x/6)-1.25)+5d0/24)+55d0/96
							elseif(x < 2.5) then; weights(j,i) = (x-2.5)**4/24; end if
						case(5)
							if    (x < 1.0) then; weights(j,i) = x**2*(x**2*(0.25d0-x/12)-0.5d0)+0.55d0
							elseif(x < 2.0) then; weights(j,i) = x*(x*(x*(x*(x/24-0.375d0)+1.25d0)-1.75d0)+0.625d0)+0.425d0
							elseif(x < 3.0) then; weights(j,i) = (3-x)**5/120d0; end if
					end select
				case(2) ! lanczos
					if(order == 0) then
						if(x < 0.5) weights = 1
					else
						if(x < order) weights(j,i) = sinc(x)*sinc(x/order)
					end if
			end select
		end do
	end do
end subroutine

pure function map_border(border, n, i) result(v)
	implicit none
	integer, intent(in) :: border, n, i
	integer :: v
	if(i < 0 .or. i >= n) then
		select case(border)
			case(0) ! constant value
				v = -1
			case(1) ! nearest
				v = max(0,min(n-1,i))
			case(2) ! cyclic
				v = modulo(i, n)
			case(3) ! mirrored
				v = modulo(i, 2*n-2)
				if(v >= n) v = 2*n-2-v
			case default
				v = -1
		end select
	else
		v = i
	end if
end function

! pos[ndim,nout] has indices into pre-flattened idata. nout = product(oshape)
! type indicates the type of interpolation to use. 0 is convolution, 1 is
! spline and 2 is lanczos. border indicates how to handle borders. 0 is
! constant zero value, 1 is nearest, 2 is cyclic and 3 is mirrored. trans indicates
! the transpose operation. It only makes sense when interpolate is a linear operation,
! which it is as long as one doesn't use constant boundary values.
subroutine interpol(idata, ishape, odata, pos, type, order, border, trans)
	implicit none
	real(_), intent(inout) :: idata(:,:), odata(:,:)
	real(_), intent(in)    :: pos(:,:)
	integer, intent(in)    :: ishape(:), type, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: weights(:,:)
	real(_) :: v(size(idata,1)), res(size(idata,1))
	integer :: off(size(pos,2)), inds(size(pos,2))
	integer :: xi, si, ci, i, j, dind, ndim, nsamp, nw, nsub, ncon, n

	ndim  = size(pos,2)
	nsamp = size(pos,1)
	nsub  = size(idata,1)
	nw    = get_weight_length(type, order)
	ncon  = nw**ndim
	if(trans) idata = 0
	!$omp parallel private(si,weights,off,res,inds,ci,dind,i,xi,v,j,n)
	allocate(weights(nw,ndim))
	!$omp do
	do si = 1, nsamp
		call calc_weights(type, order, pos(si,:), weights, off)
		! Multiply each interpolation weight with its corresponding
		! element in idata. For a 2d case with a non-flattened idata D
		! in C order, this would be
		!  D00*W00*W01 D01*W00*W11 D02*W00*W21
		!  D10*W10*W01 D11*W10*W11 D12*W10*W21
		!  D20*W20*W01 D21*W20*W11 D22*W20*W21
		! So loop through each cell of context
		if(.not. trans) res  = 0
		inds = 0
		cloop: do ci = 1, ncon
			! Get the value of this cell of context, taking into
			! account boundary conditions
			dind = 0
			do i = 1,ndim
				xi = inds(i) + off(i)
				n  = ishape(i)
				xi = map_border(border, ishape(i), xi)
				! If we don't map onto a valid point (because we use null-boundaries),
				! this cell doesn't contribute, so go to the next one
				if(xi < 0) cycle cloop
				dind = dind * n + xi
			end do
			if(.not. trans) then
				! Standard interpolation
				v = idata(:,dind+1)
				! Now multiply this value by all the relevant weights, one
				! for each dimension
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				res = res + v
			else
				! Transposed interpolation
				v = odata(:,si)
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				do j = 1, nsub
					!$omp atomic
					idata(j,dind+1) = idata(j,dind+1) + v(j)
				end do
			end if
			! Advance to next cell
			do i = ndim,1,-1
				inds(i) = inds(i) + 1
				if(inds(i) < nw) exit
				inds(i) = 0
			end do
		end do cloop
		if(.not. trans) odata(:,si) = res
	end do
	!$omp end parallel
end subroutine

!! Implementing the interpolation in fortran due to speed issues.
!! The numpy implementation was only 3 times faster than calling
!! slalib directly.
!!
!! I will try to keep the ordering memory efficient too, this time,
!! with x: (incomp,nsamp),  y: (oncomp,nsamp)
!!   xbox: (incomp,2),      n: (incomp)
!!  ygrid: (oncomp,ngrid), dy: (incomp,oncomp,ngrid)
!!
!! When collapsing dimensions, use C ordering, as it has no
!! impact on performance in this case.
!
!function ipol(x, xbox, n, ygrid, dygrid) result(y)
!  implicit none
!  real(8)       :: x(:,:), xbox(:,:), ygrid(:,:), dygrid(:,:,:)
!  real(8)       :: y(size(ygrid,1),size(x,2))
!  real(8)       :: x0(size(x,1)), idx(size(x,1))
!  real(8)       :: xrel(size(x,1))
!  integer(4)    :: n(:), xind(size(x,1))
!  integer(4)    :: incomp, oncomp, nsamp, ngrid, ic, oc, is, ig
!  integer(4)    :: steps(size(x,1))
!  incomp = size(x,1); oncomp = size(ygrid,1)
!  nsamp  = size(x,2); ngrid  = size(ygrid,2)
!
!  ! First build the nD to 1D translation
!  steps(incomp) = 1
!  do ic = incomp-1, 1, -1
!     steps(ic) = steps(ic+1)*n(ic+1)
!  end do
!  x0 = xbox(:,1); idx = (n-1)/(xbox(:,2)-xbox(:,1))
!  ! Then do the actual lookup
!  do is = 1, nsamp
!     xrel = (x(:,is)-x0)*idx
!     xind = floor(xrel+0.5)
!     xrel = xrel - xind
!     ig   = sum(xind*steps)+1
!     do oc = 1, oncomp
!        y(oc,is) = ygrid(oc,ig) + sum(dygrid(:,oc,ig)*xrel)
!     end do
!  end do
!end function

end module
