"""This module implements functions for drawing a coordinate grid and
coordinate axes on an image, for example for use with enmap."""
import numpy as np, PIL.Image, PIL.ImageDraw, time
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
from wand.display import display

def calc_line_segs(pixs, steplim=10.):
	# Split on huge jumps
	lens = np.sum((pixs[1:]-pixs[:-1])**2,1)**0.5
	typical = np.median(lens)
	jump = np.where(lens > typical*steplim)[0]
	return np.split(pixs, jump+1)

def calc_grid_segs(shape, wcs, steps=[2,2], nstep=[100,100]):
	"""Return an array of line segments representing a coordinate grid
	for the given shape and wcs. the steps argument describes the
	number of points to use along each meridian."""
	ntheta = np.floor(180./steps[0])+1
	nphi   = np.floor(360./steps[1])+1

	segs = []
	# Draw lines of longitude
	for phi in np.arange(nphi)*steps[1]:
		pixs = np.array(wcs.wcs_world2pix(phi, np.linspace(-90,90,nstep[0],endpoint=True), 0)).T.astype(int)
		segs += calc_line_segs(pixs)

	# Draw lines of latitude
	for theta in np.arange(ntheta)*steps[0]-90:
		pixs = np.array(wcs.wcs_world2pix(np.linspace(0,360,nstep[1],endpoint=True), theta, 0)).T.astype(int)
		segs += calc_line_segs(pixs)
	return segs

def draw_grid_segs_wand(segs, shape):
	with Drawing() as draw:
		draw.stroke_antialias = False
		draw.stroke_dash_array = [1,3]
		draw.stroke_color = Color("gray")
		draw.fill_opacity = 0
		for seg in segs: draw.polyline([list(p) for p in seg])
		with Image(width=shape[-1], height=shape[-2]) as img:
			t1 = time.time()
			draw(img)
			t2 = time.time()
			print "time:", t2-t1
			return np.fromstring(img.make_blob("RGBA"),np.uint8).reshape(img.height, img.width, 4)

def draw_grid_segs_pil(segs, shape):
	img  = PIL.Image.new("RGBA", shape[1::-1])
	draw = PIL.ImageDraw.Draw(img)
	t1 = time.time()
	for seg in segs:
		draw.line([tuple(i) for i in seg], fill=(0,0,0,32))
	t2 = time.time()
	print "time:", t2-t1
	return np.array(img)

def combine(images):
	def toimg(arr): return PIL.Image.fromarray(arr).convert('RGBA')
	res = toimg(images[0])
	for img in images[1:]:
		res = PIL.Image.alpha_composite(res, toimg(img))
	return np.array(res)
