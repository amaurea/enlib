module nmat_core
	implicit none
! This module implements the core of the inverse noise matrix
! of the map maker, in order to make it openmp-able. In the
! previous version of the noise matrix, evaluation was not a
! real bottleneck, but it took about 1/3 of the time, I think.
! It would be nice to get that down to negligible amounts.
! As with pmat_core, this module handles one TOD at a time.
! Loop over tods in python, calling this for each.

! Because of the possibility of a variable number of noise
! basis vectors per bin, these are stored in a flattened format:
! (ndet,nvec)+(nbin), and similarly for the mode noise.

! For each frequency we will perform
!  N"d = (Nu+VEV')"d
! By the woodbury identity, this is
!  N"d = (Nu"-Nu"V(E"+V'Nu"V)"V'Nu")d
! d is pretty big, so we want as few operations with that as possible.
! On the other hand, we don't want to expand the full ndet*ndet matrices
!  Q' = sqrt((E"+V'Nu"V)")V'Nu"
! This results in only 3 multiplications rather than 5 as one
! would otherwise get. We therefore require Q rather than V and
! E as arguments. Q us (ndet,nvec), just like  is.

contains

	! ftod[d1,f] -> iN[d1,d2] ftod[d2,f], where iN = iNu - VEV'
	subroutine nmat_detvecs(ftod, bins, iNu, V, E, ebins)
		implicit none
		! Arguments
		complex(_), intent(inout) :: ftod(:,:)
		integer(4), intent(in)    :: bins(:,:), ebins(:,:)
		real(_),    intent(in)    :: iNu(:,:), V(:,:), E(:)
		! Work
		real(_),    allocatable   :: Q(:,:), Qd(:,:), orig(:,:), iNud(:,:)
		real(_)                   :: esign
		integer(4)                :: bi, nbin, nfreq, ndet, b1, b2, di, nv, vi, nf,v1,v2, si, nm, nmode
		nfreq = size(ftod,1)
		ndet  = size(ftod,2)
		nbin  = size(bins,2)
		nmode = 2*nfreq
		!!$omp parallel do private(bi,b1,b2,v1,v2,nf,nv,esign,Q,Qd,iNud,vi,di) schedule(dynamic)
		do bi = 1, nbin
			b1 = bins(1,bi)+1;   b2 = bins(2,bi)
			b1 = min(b1, nfreq); b2 = min(b2,nfreq)
			v1 = ebins(1,bi)+1;  v2 = ebins(2,bi)
			nf = b2-b1+1; nv = v2-v1+1; nm = 2*nf
			if(nf < 1) cycle ! Skip empty bins
			if(nv == 0) then
				do di = 1, ndet
					ftod(b1:b2,di) = ftod(b1:b2,di)*iNu(di,bi)
				end do
				cycle
			end if
			allocate(Q(ndet,nv), Qd(nm,nv))
			! Construct Q = VE**0.5 on the fly. This has practically no cost, so
			! it is worth the convenience of being able to send in the more general
			! V and E.
			esign = sign(1##D##0,E(v1))
			do vi = v1, v2
				Q(:,vi-v1+1) = V(:,vi)*abs(E(vi))**0.5
			end do
			! I usually don't use the flexibility of the leading dimension argument in
			! blas, but here it is actually useful, to avoid having to copy in and out
			! parts of ftod
			call S##gemm('N', 'N', nm, nv, ndet, 1##D##0, ftod(b1,1), nmode, Q, ndet, 0##D##0, Qd, nm)
			!!$omp parallel do
			do di = 1, ndet
				ftod(b1:b2,di) = ftod(b1:b2,di)*iNu(di,bi)
			end do
			call S##gemm('N', 'T', nm, ndet, nv, esign, Qd, nm, Q, ndet, 1##D##0, ftod(b1,1), nmode)
			deallocate(Qd, Q)
		end do
	end subroutine

	subroutine nmat_covs(ftod, bins, covs)
		implicit none
		! Arguments
		complex(_), intent(inout) :: ftod(:,:)
		integer(4), intent(in)    :: bins(:,:)
		real(_),    intent(in)    :: covs(:,:,:)
		! Work
		complex(_), allocatable   :: cC(:,:), orig(:,:)
		integer(4)                :: bi, nbin, nfreq, ndet, b1, b2, nf
		nfreq = size(ftod,1)
		ndet  = size(ftod,2)
		nbin  = size(bins,2)
		!$omp parallel do private(bi,b1,b2,nf,cC,orig) schedule(dynamic)
		do bi = nbin, 1, -1
			b1 = bins(1,bi)+1;   b2 = bins(2,bi)
			b1 = min(b1, nfreq); b2 = min(b2,nfreq)
			nf = b2-b1+1
			if(nf < 1) continue ! Skip empty bins
			allocate(cC(ndet,ndet))
			allocate(orig(nf,ndet))
			cC = covs(:,:,bi)
			orig = ftod(b1:b2,:)
			!C.T(ndet,ndet)*tod.T(ndet,nf) = tod.T(ndet,nf)
			! tod(nf,ndet) = tod(nf,ndet)*C(ndet,ndet)
			call C##gemm('N', 'N', nf, ndet, ndet, (1##D##0,0##D##0), orig, nf, cC, ndet, (0##D##0,0##D##0), ftod(b1:b2,:), nf)
			deallocate(cC, orig)
		end do
	end subroutine

	! ftod[d1,f] -> iNu[d1] ftod[d1,f]
	subroutine nmat_uncorr(ftod, bins, iNu)
		implicit none
		complex(_), intent(inout) :: ftod(:,:)
		integer(4), intent(in)    :: bins(:,:)
		real(_),    intent(in)    :: iNu(:,:)
		integer :: nfreq, ndet, nbin, di, bi, b1,b2
		nfreq = size(ftod,1)
		ndet  = size(ftod,2)
		nbin  = size(bins,2)
		do di = 1, ndet
			do bi = 1, nbin
				b1 = bins(1,bi)+1;   b2 = bins(2,bi)
				b1 = min(b1, nfreq); b2 = min(b2,nfreq)
				ftod(b1:b2,di) = ftod(b1:b2,di)*iNu(bi,di)
			end do
		end do
	end subroutine

	! V and E here are not the same as in the commend at the top.
	! Instead, they are defined as Q = VE**0.5
	subroutine nmat_detvecs_old(ftod, bins, iNu, V, E, ebins)
		implicit none
		! Arguments
		complex(_), intent(inout) :: ftod(:,:)
		integer(4), intent(in)    :: bins(:,:), ebins(:,:)
		real(_),    intent(in)    :: iNu(:,:), V(:,:), E(:)
		! Work
		complex(_), allocatable   :: Q(:,:), Qd(:,:), orig(:,:), iNud(:,:)
		complex(_)                :: esign
		integer(4)                :: bi, nbin, nfreq, ndet, b1, b2, di, nv, vi, nf,v1,v2
		nfreq = size(ftod,1)
		ndet  = size(ftod,2)
		nbin  = size(bins,2)

		!$omp parallel do private(bi,b1,b2,v1,v2,nf,nv,Q,Qd,orig,iNud,di,esign) schedule(dynamic)
		do bi = nbin, 1, -1
			b1 = bins(1,bi)+1;   b2 = bins(2,bi)
			b1 = min(b1, nfreq); b2 = min(b2,nfreq)
			v1 = ebins(1,bi)+1;  v2 = ebins(2,bi)
			nf = b2-b1+1; nv = v2-v1+1
			if(nf < 1) continue ! Skip empty bins
			allocate(Q(ndet,nv))
			allocate(Qd(nv,nf)) ! Q'd'
			allocate(orig(nf,ndet), iNud(nf,ndet))
			Qd = 0; iNud = 0
			orig = ftod(b1:b2,:)
			do di = 1, ndet
				iNud(:,di) = orig(:,di)*iNu(di,bi)
			end do
			if(nv > 0) then
				! Construct Q = VE**0.5 on the fly. This has practically no cost, so
				! it is worth the convenience of being able to send in the more general
				! V and E.
				esign = sign(1##D##0,E(v1))
				do vi = v1, v2
					Q(:,vi-v1+1) = V(:,vi)*abs(E(vi))**0.5
				end do
				! Q'd' = matmul(transpose(Q)(nvec,ndet),transpose(ftod)(ndet,nf))
				! QQ'd = matmul(Q(ndet,nvec),Qd(nvec,nf))
				! => (QQ'd)' = matmul(transpose(Qd)(nf,nvec),transpose(Q)(nvec,det))
				call C##gemm('T', 'T', nv, nf, ndet, (1##D##0,0##D##0), Q, ndet, orig, nf,   (0##D##0,0##D##0), Qd, nv)
				call C##gemm('T', 'T', nf, ndet, nv, esign,    Qd,   nv,    Q, ndet, (1##D##0,0##D##0), iNud, nf)
			end if
			do di = 1, ndet
				ftod(b1:b2,di) = iNud(:,di)
			end do
			deallocate(Qd, orig, iNud, Q)
		end do
	end subroutine

	!!! Windowing stuff !!!
	! Applies a cosine window of width samples
	! to the end of the provided tod. Provide
	! a negative width to invert the filter.
	subroutine apply_window(tod, width)
		implicit none
		real(_),    intent(inout) :: tod(:,:)
		integer(4), intent(in)    :: width
		real(_),    allocatable   :: window(:)
		real(_),    parameter     :: pi = 3.14159265358979323846d0
		integer(4) :: nsamp, ndet, di, si, w
		logical    :: invert
		invert = width < 0
		w      = abs(width)
		if(width == 0) return
		nsamp = size(tod,1)
		ndet  = size(tod,2)
		! First build window
		allocate(window(w))
		if(invert) then
			!$omp parallel do
			do si = 1, w
				window(si) = 1/(0.5d0-0.5d0*cos(pi*si/w))
			end do
		else
			!$omp parallel do
			do si = 1, w
				window(si) = 0.5d0-0.5d0*cos(pi*si/w)
			end do
		end if
		!$omp parallel do
		do di = 1, ndet
			! Apply window on each end
			tod(1:w,di) = tod(1:w,di)*window
			tod(nsamp-w+1:nsamp,di) = tod(nsamp-w+1:nsamp,di) * window(w:1:-1)
		end do
	end subroutine

end module
