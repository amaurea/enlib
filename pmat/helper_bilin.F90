					xrel = (bore(:,si)+det_pos(:,di)-x0)*inv_dx
					xind = floor(xrel)
					xrel = xrel - xind
					ig   = sum(xind*steps)+1
					! Manual expansion of bilinear interpolation. Pretty bad memory
					! access pattern, sadly. But despite the huge number of operations
					! compared to gradient interpolation, it's about the same speed.
					work(:,1) = yvals(:,ig)*(1-xrel(1)) + yvals(:,ig+steps(1))*xrel(1)
					work(:,2) = yvals(:,ig+steps(2))*(1-xrel(1)) + yvals(:,ig+steps(2)+steps(1))*xrel(1)
					work(:,3) = yvals(:,ig+steps(3))*(1-xrel(1)) + yvals(:,ig+steps(3)+steps(1))*xrel(1)
					work(:,4) = yvals(:,ig+steps(2)+steps(3))*(1-xrel(1)) + yvals(:,ig+steps(2)+steps(3)+steps(1))*xrel(1)
					work(:,1) = work(:,1)*(1-xrel(2)) + work(:,2)*xrel(2)
					work(:,2) = work(:,3)*(1-xrel(2)) + work(:,4)*xrel(2)
					point = work(:,1)*(1-xrel(3)) + work(:,2)*xrel(3)
