# Installation

## Python Package

```bash
python -m pip install tract7dt
```

`tract7dt` depends on several scientific Python packages (declared in package metadata). Tractor and astrometry dependencies must be installed separately.

## Recommended Tractor Setup

### 1) Create environment

```bash
conda create -n <envname> python=3.11 -y
conda activate <envname>
```

### 2) Install system dependencies

```bash
sudo apt update
sudo apt install -y build-essential python3-dev git pkg-config \
    libcfitsio-dev libeigen3-dev swig libceres-dev \
    libgoogle-glog-dev libgflags-dev libsuitesparse-dev
```

### 3) Install Python build stack

```bash
pip install -U pip setuptools wheel cython numpy scipy fitsio emcee matplotlib
```

### 4) Install astrometry.net

```bash
conda install -c conda-forge astrometry -y
```

### 5) Build/install Tractor

```bash
git clone https://github.com/dstndstn/tractor.git
cd tractor
python setup.py build_ext --inplace --with-ceres --with-cython
pip install . --no-build-isolation
```

### 6) Verify

```bash
python -c "import tractor; print(tractor.__version__)"
python -c "from tractor.ceres_optimizer import CeresOptimizer; print('CeresOptimizer OK')"
```
