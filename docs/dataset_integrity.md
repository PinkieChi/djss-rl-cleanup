# Dataset Integrity Note

`Dataset 50_0.5_0.02.ini` was included in the original project files, but the inspected copy is 203,909 bytes of null bytes.

That means it is probably corrupt, empty, or accidentally overwritten. The notebook currently generates scheduling instances in `Scenario.make_world()` and writes a dataset when `validation=True`; it does not read this `.ini` file back.

For reproducible validation, regenerate this dataset from a fixed seed or restore it from a known-good backup, then add a loader that reconstructs the same world from the `.ini` file.
