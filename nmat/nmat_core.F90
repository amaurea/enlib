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

	subroutine nmat_detvecs(ftod, bins, iNu, V, E, vbins)
		implicit none
		! Arguments
		complex(_), intent(inout) :: ftod(:,:)
		integer(4), intent(in)    :: bins(:,:), vbins(:,:)
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
			v1 = vbins(1,bi)+1;  v2 = vbins(2,bi)
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
			call C##gemm('N', 'N', nf, ndet, ndet, (1##E##0,0##E##0), orig, nf, cC, ndet, (0##E##0,0##E##0), ftod(b1:b2,:), nf)
			deallocate(cC, orig)
		end do
	end subroutine

end module
