! These functions expect their inputs as right handed north polar coordinates!
function aomulti(mjd, icoord, ao, am) result(ocoord)
	implicit none
	real(8)    :: ao(:), am(:), mjd(:), icoord(:,:), ocoord(size(icoord,1),size(icoord,2))
	real(8)    :: ra, dec, pi
	integer(4) :: i, n
	n	= size(mjd)
	pi = 4*atan(1d0)
	do i = 1, n
		call sla_aoppat(mjd(i), ao)
		call sla_oapqk("A", icoord(1,i), pi/2-icoord(2,i), ao, ra, dec)
		call sla_ampqk(ra, dec, am, ocoord(1,i), ocoord(2,i))
	end do
end function

function oamulti(mjd, icoord, ao, am) result(ocoord)
	implicit none
	real(8)    :: ao(:), am(:), mjd(:), icoord(:,:), ocoord(size(icoord,1),size(icoord,2))
	real(8)    :: az, zen, hob, dob, rob, pi, ra, dec
	integer(4) :: i, n
	n  = size(mjd)
	pi = 4*atan(1d0)
	do i = 1, n
		call sla_aoppat(mjd(i), ao)
		call sla_mapqk(icoord(1,i), icoord(2,i), 0d0, 0d0, 0d0, 0d0, am, ra, dec)
		call sla_aopqk(ra, dec, ao, ocoord(1,i), zen, hob, dob, rob)
		ocoord(2,i) = pi/2-zen
	end do
end function
