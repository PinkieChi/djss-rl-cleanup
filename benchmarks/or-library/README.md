# OR-Library Job-Shop Benchmarks

This directory stores the raw OR-Library `jobshop1.txt` source file and extracted JSPLIB-style single-instance files used for benchmark-derived DJSS experiments.

Source: <https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html>

The OR-Library page states that `jobshop1` contains 82 job-shop test instances commonly cited in the literature, with machines numbered from 0 in each operation row. The preparation script converts selected instances to this project's 1-based machine IDs and generated dynamic due-date/arrival settings.

Regenerate the converted benchmark datasets with:

```bash
python3 scripts/prepare_or_library_benchmarks.py \
  --source benchmarks/or-library/raw/jobshop1.txt \
  --output-dir outputs/or-library-benchmark-derived-20260721/datasets
```
