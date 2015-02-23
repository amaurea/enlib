"""This module implements functions for drawing a coordinate grid and
coordinate axes on an image, for example for use with enmap."""
import numpy as np, time, os
from PIL import Image, ImageDraw, ImageFont

def calc_line_segs(pixs, steplim=10.):
	# Split on huge jumps
	lens = np.sum((pixs[1:]-pixs[:-1])**2,1)**0.5
	typical = np.median(lens)
	jump = np.where(lens > typical*steplim)[0]
	return np.split(pixs, jump+1)

class Gridinfo: pass

def calc_gridinfo(shape, wcs, steps=[2,2], nstep=[200,200]):
	"""Return an array of line segments representing a coordinate grid
	for the given shape and wcs. the steps argument describes the
	number of points to use along each meridian."""
	steps = np.zeros([2])+steps
	nstep = np.zeros([2],dtype=int)+nstep

	gridinfo = Gridinfo()
	ntheta = np.floor(180./steps[0])+1
	nphi   = np.floor(360./steps[1])+1

	gridinfo.lon = []
	gridinfo.lat = []
	# Draw lines of longitude
	for phi in np.arange(nphi)*steps[1]:
		pixs = np.array(wcs.wcs_world2pix(phi, np.linspace(-90,90,nstep[0],endpoint=True), 0)).T.astype(int)
		gridinfo.lon.append((phi,calc_line_segs(pixs)))

	# Draw lines of latitude
	for theta in np.arange(ntheta)*steps[0]-90:
		pixs = np.array(wcs.wcs_world2pix(np.linspace(0,360,nstep[1],endpoint=True), theta, 0)).T.astype(int)
		gridinfo.lat.append((theta,calc_line_segs(pixs)))
	return gridinfo

def draw_grid(img, gridinfo, color="00000020"):
	col = tuple([int(color[i:i+2],16) for i in range(0,len(color),2)])
	grid = Image.new("RGBA", img.size)
	draw = ImageDraw.Draw(grid, "RGBA")
	for cval, segs in gridinfo.lon:
		for seg in segs:
				draw.line([tuple(i) for i in seg], fill=col)
	for cval, segs in gridinfo.lat:
		for seg in segs:
			draw.line([tuple(i) for i in seg], fill=col)
	return Image.alpha_composite(img, grid)

def calc_label_pos(linesegs, shape):
	# For each grid line, identify where we enter and exit the
	# image. If these points exist, draw coorinates there. If they
	# do not, check if the 0 coordinate is in the image. If it is,
	# draw the coordinate there. Otherwise there is no point in
	# drawing.
	shape = np.array(shape)
	label_pos = []
	for cval, segs in linesegs:
		for seg in segs:
			# Check if we cross 0
			seg = np.array(seg)
			edges = np.array(np.where((seg[1:]*seg[:-1] < 0)|((seg[1:]-shape)*(seg[:-1]-shape) < 0)))
			# Mask those outside the image
			ocoord = edges.copy(); ocoord[1] = 1-ocoord[1]
			other = seg[tuple(ocoord)]
			outside = (other<0)|(other>shape[1-edges[1]])
			edges = edges[:,~outside]
			if edges.size > 0:
				# Ok, we cross an edge. Interpolate the position for each
				for ei,ec in edges.T:
					x = seg[([ei,ei+1],[ec,ec])]
					y = seg[([ei,ei+1],[1-ec,1-ec])]
					xcross = float(0 if x[0]*x[1] <= 0 else shape[ec])
					ycross = y[0] + (y[1]-y[0])*(xcross-x[0])/(x[1]-x[0])
					entry  = [cval,0,0]
					entry[2-ec] = ycross
					entry[1+ec] = xcross
					label_pos.append(entry)
			else:
				# No edge crossing. But perhaps that's because everything is
				# happening inside the image. If so, the first point should
				# be inside.
				if np.all(seg[0]>=0) and np.all(seg[0]<shape):
					label_pos.append([cval,seg[0,0],seg[0,1]])
	return label_pos

def calc_bounds(boxes, size):
	"""Compute bounding box for a set of boxes [:,{from,to},{x,y}].
	The result will no less than ((0,0),size)."""
	return np.array([np.minimum((0,0),np.min(boxes[:,0],0)),np.maximum(size,np.max(boxes[:,1],0))])

def expand_image(img, bounds):
	res = Image.new("RGBA", bounds[1]-bounds[0])
	res.paste(img, tuple(-bounds[0]))
	return res

def draw_labels(img, label_pos, fname="arial.ttf", fsize=16, fmt="%.0f", color="000000"):
	# For each label, determine the size the text would be, and
	# displace it left, right, up or down depending on which edge
	# of the image it is at
	col = tuple([int(color[i:i+2],16) for i in range(0,len(color),2)])
	try:
		font = ImageFont.truetype(font=fname, size=fsize)
	except IOError:
		# Load fallback font
		font = ImageFont.truetype(font="arial.ttf", size=fsize, filename=os.path.join(os.path.dirname(__file__), "arial.ttf"))
	labels = []
	boxes  = []
	for cval, x, y in label_pos:
		pos   = np.array([x,y])
		label = fmt % cval
		lsize = np.array(font.getsize(label))
		if   x == 0:           box = np.array([pos-[lsize[0],lsize[1]/2],pos+[0,lsize[1]/2]])
		elif x == img.size[0]: box = np.array([pos-[0,lsize[1]/2],pos+[lsize[0],lsize[1]/2]])
		elif y == 0:           box = np.array([pos-[lsize[0]/2,lsize[1]],pos+[lsize[0]/2,0]])
		elif y == img.size[1]: box = np.array([pos-[lsize[0]/2,0],pos+[lsize[0]/2,lsize[1]]])
		else: continue
		labels.append(label)
		boxes.append(box)
	boxes  = np.array(boxes).astype(int)
	# Pad image to be large enough to hold the displaced labels
	bounds = calc_bounds(boxes, img.size)
	img    = expand_image(img, bounds)
	boxes -= bounds[0]
	# And draw the text
	draw = ImageDraw.Draw(img)
	for label, box in zip(labels, boxes):
		draw.text(box[0], label, col, font=font)
	return img
