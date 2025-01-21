from setuptools import setup, find_packages

setup(
   name="qmc",
   version="0.1.0",
   author="Your Name",
   author_email="your.email@example.com",
   description="Quantum Monte Carlo implementation with JAX",
   packages=find_packages(),
   install_requires=[
       "pyscf==2.7.0",
       "pyqmc==0.6.0", 
       "numpy==2.0.2",
       "jax==0.4.30",
       "jax-metal==0.1.0",
       "jaxlib==0.4.30",
       "h5py==3.12.1",
   ],
   python_requires=">=3.9.20",
   classifiers=[
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 3.9",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent",
   ],
)