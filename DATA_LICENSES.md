# Data sources and reuse

`data/msc_sessions123.npz` contains prepared Pearson-connectivity fields from the
Midnight Scan Club, [OpenNeuro ds000224, version 1.0.4](https://doi.org/10.18112/openneuro.ds000224.v1.0.4),
distributed under CC0. Retain the source attribution when reusing these fields.
The array includes ten participants, their first three sessions, five processing
paths, and 79,800 connectivity edges.

`data/locked_adni_mean_history.npy`, `data/locked_adni_mean_target.npy`, and
`data/fixed_coefficients.json` contain population means and shared model
coefficients from 69 ADNI training participants. They contain no participant axis
or participant identifiers. ADNI raw data and participant-level derived records
are not included. Obtain those data through [ADNI](https://adni.loni.usc.edu/)
and follow the applicable data-use agreement when supplying local inputs.

The synthetic example and original aggregate tables are distributed under the
repository's [MIT license](LICENSE). The data files retain the values in the
v1.0.0 release. Their SHA-256 checksums are in [data/SHA256SUMS](data/SHA256SUMS).
