# For each exported WIT interface (e.g. "cacheback"), componentize-py expects
# a Python module with the same name (cacheback.py) that provides the resource
# implementations. This file just imports that module so it is discovered.
import cacheback
