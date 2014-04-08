%module iers
%{
#define SWIG_FILE_WITH_INIT
#include "iers.h"
%}
%rename(lookup) iers_lookup;
%include "iers.h"
