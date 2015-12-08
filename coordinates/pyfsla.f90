! These functions expect their inputs as right handed north polar coordinates!
subroutine aomulti(mjd, coord, ao, am)
	implicit none
	real(8), intent(in)    :: mjd(:), am(:)
	real(8), intent(inout) :: coord(:,:), ao(:)
	real(8)    :: ra, dec, pi
	integer(4) :: i, n
	n	= size(mjd)
	pi = 4*atan(1d0)
	!$omp parallel do private(i, ra, dec) firstprivate(ao)
	do i = 1, n
		call sla_aoppat(mjd(i), ao)
		call sla_oapqk("A", coord(i,1), pi/2-coord(i,2), ao, ra, dec)
		call sla_ampqk(ra, dec, am, coord(i,1), coord(i,2))
	end do
end subroutine

subroutine oamulti(mjd, coord, ao, am)
	implicit none
	real(8), intent(in)    :: mjd(:), am(:)
	real(8), intent(inout) :: coord(:,:), ao(:)
	real(8)    :: zen, hob, dob, rob, pi, ra, dec
	integer(4) :: i, n
	n  = size(mjd)
	pi = 4*atan(1d0)
	!$omp parallel do private(i, ra, dec, zen, hob,dob) firstprivate(ao)
	do i = 1, n
		call sla_aoppat(mjd(i), ao)
		call sla_mapqk(coord(i,1), coord(i,2), 0d0, 0d0, 0d0, 0d0, am, ra, dec)
		call sla_aopqk(ra, dec, ao, coord(i,1), zen, hob, dob, rob)
		coord(i,2) = pi/2-zen
	end do
end subroutine
