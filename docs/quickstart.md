# Quick Start

## 1) Generate a config template

```bash
# write sample_config.yaml in current directory
tract7dt dump-config

# write to explicit path
tract7dt dump-config --out sample_config.yaml

# overwrite existing file
tract7dt dump-config --force
```

## 2) Edit required fields

At minimum, update:

- `inputs.input_catalog`
- `inputs.image_list_file`
- `inputs.gaiaxp_synphot_csv` (if used)
- `outputs.work_dir`

## 3) Run full pipeline

```bash
tract7dt run --config /path/to/config.yaml
```

## 4) Useful partial commands

```bash
tract7dt run-epsf --config /path/to/config.yaml
tract7dt build-patches --config /path/to/config.yaml
tract7dt build-patch-inputs --config /path/to/config.yaml
tract7dt run-patches --config /path/to/config.yaml
tract7dt merge --config /path/to/config.yaml
```

## 5) Sample data for test runs

```bash
# default target folder under cwd
tract7dt download-sample-data

# explicit target folder
tract7dt download-sample-data --dir /path/to/dataset

# overwrite existing target folder
tract7dt download-sample-data --dir /path/to/dataset --force
```

See [Sample Data](sample-data.md) for download behavior and terms.
