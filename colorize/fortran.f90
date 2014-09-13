subroutine remap(a, res, vals, cols)
	implicit none
	real(8),    intent(in)    :: a(:), vals(:)
	integer(1), intent(in)    :: cols(:,:)
	integer(1), intent(inout) :: res(:,:)
	real(8) :: v, x(2), y(size(cols,1),2)
	integer :: i, j
	do i = 1, size(a)
		v = a(i)
		! Find location first greater value in vals
		do j = 1, size(vals)
			if(vals(j) > v) exit
		end do
		! Handle edge cases
		if(j <= 1) then
			res(:,i) = cols(:,j)
		elseif(j > size(vals)) then
			res(:,i) = cols(:,size(vals))
		else
			x = vals(j-1:j)
			y = cols(:,j-1:j)
			res(:,i) = min(max(0,nint(y(:,1) + (v-x(1))*(y(:,2)-y(:,1))/(x(2)-x(1)))),255)
		end if
	end do
end subroutine
