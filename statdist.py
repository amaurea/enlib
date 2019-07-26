from __future__ import division
import numpy as np, scipy.stats, scipy.integrate, scipy.optimize

def rint(f, a, b, e):   return scipy.integrate.romb(f(np.linspace(a,b,2**e+1)), (b-a)/2**e)
def find_root(f, a, b):
	fa, fb = f(a), f(b)
	if fa*fb < 0: return scipy.optimize.brentq(f, a, b)
	else: return a if abs(fa) <= abs(fb) else b

def maxgauss_cdf(x, n): return scipy.stats.norm.cdf(x)**n
def maxgauss_sf (x, n): return mingauss_cdf(-x,n)
def maxgauss_pdf(x, n): return n*maxgauss_cdf(x,n-1)*scipy.stats.norm.pdf(x)
def maxgauss_mean(n):   return rint(lambda x: x*maxgauss_pdf(x, n), -30, 30, 10)
def maxgauss_var(n):    return rint(lambda x: x**2*maxgauss_pdf(x,n), -30, 30, 10)-maxgauss_mean(n)**2
def maxgauss_std(n):    return maxgauss_var(n)**0.5
def maxgauss_quant(p, n): return find_root(lambda x: maxgauss_cdf(x, n)-p, -30, 30)
def maxgauss_n(mean):   return 10**find_root(lambda logn: maxgauss_mean(10**logn)-mean, 0, 10)

def mingauss_cdf(x, n):
	exact = 1 - scipy.stats.norm.sf(x)**n
	approx = scipy.stats.norm.cdf(x)*n
	return np.where(approx < 1e-6, approx, exact)
def mingauss_sf (x, n): return maxgauss_cdf(-x,n)
def mingauss_pdf(x, n): return maxgauss_pdf(-x,n)
def mingauss_mean(n):   return rint(lambda x: x*mingauss_pdf(x, n), -30, 30, 10)
def mingauss_var(n):    return rint(lambda x: x**2*mingauss_pdf(x,n), -30, 30, 10)-mingauss_mean(n)**2
def mingauss_std(n):    return mingauss_var(n)**0.5
def mingauss_quant(p, n): return find_root(lambda x: mingauss_cdf(x, n)-p, -30, 30)
def mingauss_n(mean):   return 10**find_root(lambda logn: mingauss_mean(10**logn)-mean, 0, 10)
