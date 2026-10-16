"""Install only the opt-in source hook; never import torch at startup."""
from step5_fp8kv.hook import install

install()
