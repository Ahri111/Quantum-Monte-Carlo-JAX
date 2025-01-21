import os
import jax
import jax.numpy as jnp

os.environ['JAX_PLATFORMS'] = 'METAL'
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'