"""This module provides an interface for noise matrices.
Individual experiments can inherit from this - other functions
in enlib will work as long as this interface is followed.
For performance and memory reasons, the noise matrix
overwrites its input array."""

# Dilemma: Most noise matrices will act in fourier space.
# The fourier version of the array is potentially already
# available in the calling function, so taking a real array
# as input and fourier-transforming it internally may be
# wasteful.
#
# But on the other hand, if we take a fourier
# array as input.we force the cost of a fourier transform
# even for the cases where it isn't needed, such as
# the case of white noise.
#
# If we provide both a real and fourier interface, then
# the calling code will need to know which one the
# noise matrix prefers.
#
# In practice, I can't think of any case where the
# fourier array would be available on the outside
# in the first place. So let's go with a real interface,
# and worry about wasted fourier transforms later,
# if that actually ever happens.

class NoiseMatrix:
	def apply(self, tod):
		"""Apply the full noise matrix to tod. tod is overwritten,
		but also returned for convenience."""
		return tod
	def white(self, tod):
		"""Apply a fast, uncorrelated (white noise) approximation
		of the noise matrix to tod. tod is overwritten, but also
		returned for convenience."""
		return tod
	
