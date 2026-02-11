# Sample Data

`download-sample-data` downloads and prepares a sample dataset package.

## Command

```bash
tract7dt download-sample-data
tract7dt download-sample-data --dir /path/to/dataset
tract7dt download-sample-data --dir /path/to/dataset --force
```

## Behavior

- Downloads release asset from configured source.
- Verifies SHA256 (if configured).
- Extracts archive.
- Standardizes FITS filenames to `7DT_UDS_m***.fits`.
- Writes `image_list.txt` with absolute paths.

## Archive Handling

Current behavior treats the downloaded archive as temporary:

- existing archive path is removed before download
- archive is removed after extraction/preparation

So users do not retain `.tar.gz` leftovers by default.

## Progress Display

- TTY: one-line in-place progress bar with percent and MiB.
- non-TTY: periodic progress logs.

## Logging Scope Caveat

`download-sample-data` does not take `--config`, so pipeline YAML logging settings are not applied directly to this command.

## Source Configuration

Default source can be set via environment variables:

- `TRACT7DT_SAMPLE_REPO`
- `TRACT7DT_SAMPLE_TAG`
- `TRACT7DT_SAMPLE_ASSET`
- `TRACT7DT_SAMPLE_SHA256` (optional)

## Usage Terms

Sample data distribution has separate usage restrictions from code license.
See `LICENSE_SAMPLE_DATA.txt` in the sample package.
