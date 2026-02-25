# Commands

## Pipeline Commands

All pipeline commands require `--config /path/to/config.yaml` and execute one or more pipeline stages.

| Command | Stages executed | Description |
|---------|----------------|-------------|
| `run` | all stages | Full pipeline: load → [augment Gaia] → ePSF → patches → patch inputs → run patches → merge → [compute ZP]. Stages in brackets run when `zp.enabled: true`. |
| `run-epsf` | load + ePSF | Load inputs, validate, and build ePSF products. Useful for iterating on ePSF parameters. |
| `build-patches` | load + patches | Load inputs and generate patch geometry definitions. |
| `build-patch-inputs` | load + patches + payloads | Load inputs, build patches, and write per-patch data payloads. |
| `run-patches` | run patches only | Run Tractor fitting subprocesses on **existing** patch payloads. Does not reload inputs. |
| `merge` | merge only | Merge **existing** patch fit outputs into the final catalog. Does not reload inputs. |
| `compute-zp` | ZP only | Compute zero-point calibration on the **existing** merged catalog. Adds `MAG_*_fit` and `MAGERR_*_fit` columns. |

### Usage

```bash
tract7dt run --config /path/to/config.yaml
tract7dt run-epsf --config /path/to/config.yaml
tract7dt build-patches --config /path/to/config.yaml
tract7dt build-patch-inputs --config /path/to/config.yaml
tract7dt run-patches --config /path/to/config.yaml
tract7dt merge --config /path/to/config.yaml
tract7dt compute-zp --config /path/to/config.yaml
```

### Stage Dependencies

Some commands depend on outputs from earlier stages:

| Command | Requires existing... |
|---------|---------------------|
| `run` | Nothing (runs everything). |
| `run-epsf` | Nothing (loads inputs fresh). |
| `build-patches` | Nothing (loads inputs fresh). |
| `build-patch-inputs` | Nothing (loads inputs fresh; also builds patches). |
| `run-patches` | Patch payloads in `outputs.patch_inputs_dir` and ePSF products in `outputs.epsf_dir`. |
| `merge` | Per-patch fit CSVs in `outputs.tractor_out_dir`. |
| `compute-zp` | Merged catalog at `outputs.final_catalog` and GaiaXP CSV at `inputs.gaiaxp_synphot_csv`. |

### Typical Iteration Patterns

**Full run from scratch:**

```bash
tract7dt run --config config.yaml
```

**Re-run just the fitting after changing patch_run parameters:**

```bash
# Edit config.yaml (e.g. change gal_model, n_opt_iters, psf_model)
tract7dt run-patches --config config.yaml
tract7dt merge --config config.yaml
```

**Re-merge after a partial re-run:**

```bash
tract7dt merge --config config.yaml
```

**Re-compute ZP with different clipping parameters:**

```bash
# Edit config.yaml (e.g. change zp.clip_sigma)
tract7dt compute-zp --config config.yaml
```

**Test ePSF quality before running the full pipeline:**

```bash
tract7dt run-epsf --config config.yaml
# Inspect EPSFs/ directory, then proceed
tract7dt run --config config.yaml
```

## Utility Commands

### `dump-config`

Write the latest sample config template to a file.

```bash
tract7dt dump-config                     # → sample_config.yaml
tract7dt dump-config --out my_config.yaml
tract7dt dump-config --force             # overwrite existing
```

The generated file contains all parameters with comments. It is the recommended starting point for new configurations.

### `download-sample`

Download and prepare a sample image dataset for testing.

```bash
tract7dt download-sample                       # → default dir under cwd
tract7dt download-sample --dir /path/to/data
tract7dt download-sample --dir /path/to/data --force
```

See [Sample Data](sample-data.md) for download behavior, progress display, and usage terms.

## Logging Behavior

- **Pipeline commands** (`run`, `run-epsf`, `build-patches`, `build-patch-inputs`, `run-patches`, `merge`, `compute-zp`) apply the `logging.*` section from the YAML config after loading.
- **`download-sample`** does not read `--config`; it uses an independent logging setup.
- **`dump-config`** does not read `--config`; it only writes the template file.
