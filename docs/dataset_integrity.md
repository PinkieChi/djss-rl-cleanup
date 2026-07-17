# Dataset Integrity Note

The cleaned project now contains a restored valid copy of `Dataset 50_0.5_0.02.ini`.

The original copy found inside `djss-rl G` was 203,909 bytes of null bytes. A separate good copy from `Downloads/Dataset 50_0.5_0.02.ini` was inspected and copied into this project.

Current integrity check:

- Size: 184,213 bytes
- Encoding: UTF-8
- Null bytes: 0
- INI sections: `[world]`
- Parsed world entries: 679

The notebook now includes `Scenario.make_world_from_dataset(...)` and `make_env(dataset_path=...)`, so it can load this restored dataset. The default validation/evaluation setup uses `Dataset 50_0.5_0.02.ini` unless `DJSS_DATASET_PATH` points elsewhere.

One limitation remains: the original dataset writer serialized compatible machines as Python object memory addresses, for example `<__main__.Machine object at 0x...>`, rather than stable machine IDs or names. The loader maps those opaque address tokens deterministically to the restored machine list, preserving operation processing times and routing cardinality. Exact original machine identity cannot be guaranteed unless the dataset is re-exported with stable machine identifiers.

Smoke-test result after adding the loader:

- Loaded jobs: 50
- Loaded machines: 15
- Loaded operations: 406
- Observation dimension: 14
- Action dimension: 9
