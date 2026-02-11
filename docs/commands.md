# Commands

## Pipeline Commands

- `run`: full pipeline (load -> ePSF -> patches -> patch inputs -> run patches -> merge)
- `run-epsf`: load inputs + build ePSF only
- `build-patches`: load inputs + build patches only
- `build-patch-inputs`: load inputs + build patches + build patch payloads
- `run-patches`: run Tractor subprocesses on existing patch payloads
- `merge`: merge existing patch outputs into final catalog

All pipeline commands require:

```bash
--config /path/to/config.yaml
```

## Utility Commands

- `dump-config`: write the sample config template
- `download-sample-data`: download and unpack sample image data

## Important Logging Note

- Pipeline commands (`run*`, `merge`) apply logging from YAML (`logging.*`) after config load.
- `download-sample-data` does not read `--config`; its logging setup is independent of YAML.
