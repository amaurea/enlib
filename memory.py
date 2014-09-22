# Get information about the process's memory usage.
import resource, os

def current():
	with open("/proc/self/statm","r") as f:
		return int(f.readline().split()[0])*resource.getpagesize()

def resident():
	with open("/proc/self/status","r") as f:
		for line in f:
			toks = line.split()
			if toks[0] == "VmRSS:":
				return int(toks[1])*1024

def max():
	with open("/proc/self/status","r") as f:
		for line in f:
			toks = line.split()
			if toks[0] == "VmPeak:":
				return int(toks[1])*1024
