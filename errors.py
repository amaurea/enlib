"""This module provides a set of exceptions specific to data analysis."""
class DataError(Exception): pass
class DataMissing(DataError): pass
class DataInvalid(DataError): pass
