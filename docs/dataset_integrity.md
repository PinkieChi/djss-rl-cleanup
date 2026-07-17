# Dataset Integrity Note

The cleaned project now contains a restored valid copy of `Dataset 50_0.5_0.02.ini`.

The original copy found inside `djss-rl G` was 203,909 bytes of null bytes. A separate good copy from `Downloads/Dataset 50_0.5_0.02.ini` was inspected and copied into this project.

Current integrity check:

- Size: 184,213 bytes
- Encoding: UTF-8
- Null bytes: 0
- INI sections: `[world]`
- Parsed world entries: 679

The notebook currently generates scheduling instances in `Scenario.make_world()` and writes a dataset when `validation=True`; it does not read this `.ini` file back. For reproducible validation, add a loader that reconstructs the same world from this `.ini` file.
