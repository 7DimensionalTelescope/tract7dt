# Tract7DT

[![Docs](https://github.com/seoldh99/tract7dt/actions/workflows/docs.yml/badge.svg)](https://github.com/seoldh99/tract7dt/actions/workflows/docs.yml)

7DT image-specific Tractor photometry pipeline.  
This distribution is licensed under MIT.

## Documentation

- Live documentation: https://seoldh99.github.io/tract7dt/

## Installation

1) Install Tractor (see https://github.com/dstndstn/tractor; however the upstream README seems outdated; use the recommended steps below).  
2) Install this package:

```bash
python -m pip install tract7dt

# To update an existing installation:
python -m pip install -U tract7dt
```

`astrometry.net` and Tractor must be installed in advance. Python prerequisites such as `numpy`, `scipy`, `astropy`, `photutils`, `sep`, `matplotlib`, `fitsio`, `emcee`, and `PyYAML` are declared as dependencies.

Check your installed version:

```bash
tract7dt --version
```

## Tractor installation (recommended)

### 1) Conda environment

```bash
conda create -n {envname} python=3.11 -y
conda activate {envname}
```

### 2) System dependencies (apt)

```bash
sudo apt update
sudo apt install -y build-essential python3-dev git pkg-config \
    libcfitsio-dev libeigen3-dev swig libceres-dev \
    libgoogle-glog-dev libgflags-dev libsuitesparse-dev
```

### 3) Python build tools + scientific stack

```bash
pip install -U pip setuptools wheel cython numpy scipy fitsio emcee matplotlib
```

### 4) astrometry.net (required by Tractor)

```bash
conda install -c conda-forge astrometry -y
```

### 5) Tractor build (Cython + Ceres enabled)

```bash
git clone https://github.com/dstndstn/tractor.git
cd tractor
python setup.py build_ext --inplace --with-ceres --with-cython
pip install . --no-build-isolation
```

### 6) Verify install

```bash
python -c "import tractor; print(tractor.__version__)"
python -c "from tractor.ceres_optimizer import CeresOptimizer; print('CeresOptimizer OK')"
python examples/tractor-sdss-synth.py
```

## Quick start

### 1) Create a YAML config

```bash
# Writes sample_config.yaml in the current directory by default.
tract7dt dump-config

# Or write to a specific path.
tract7dt dump-config --out sample_config.yaml

# Use --force to overwrite an existing file.
tract7dt dump-config --force
```

### 2) Download sample image data for test runs

```bash
# Download under current directory with original dataset folder name.
tract7dt download-sample

# Or set an explicit dataset directory path.
tract7dt download-sample --dir "/path/to/sample_data_dir"

# Overwrite an existing target dataset directory.
tract7dt download-sample --dir "/path/to/sample_data_dir" --force
```

Without `--dir`, the command creates the dataset under the current directory using the archive's original dataset folder name.  
With `--dir /path/to/sample_data_dir`, the dataset directory is created at that exact path.  
In both cases, FITS names are standardized to `7DT_UDS_m***.fits` and `image_list.txt` is generated with absolute paths.

### 3) Edit required config paths

At minimum, set:
- `inputs.tile_name`
- `inputs.input_catalog`
- `inputs.image_list_file`
- `inputs.gaiaxp_synphot_csv`
- `outputs.work_dir`

### 4) Run the full pipeline

```bash
tract7dt run --config /path/to/config.yaml
```

### Role of the config file

- The YAML config centralizes runtime decisions (inputs, outputs, checks, ePSF behavior, patch options, merge behavior, ZP calibration, logging, performance knobs).
- `tract7dt run --config ...` reads this single file and executes the full pipeline (load, ePSF, patches, fitting, merge, and optionally Gaia augmentation + ZP calibration).
- Subcommands share the same config model, so one file can drive full runs and partial reruns consistently.

For full command/config/behavior reference, see the docs in `docs/` or the hosted site on GitHub Pages.
