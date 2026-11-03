# Data sources and access

The public repository contains code and runtime generators. Prepared arrays,
trained coefficients, population means, saved results, and saved simulation
outputs are in the private
[yury-rd/fmri-transfer-data](https://github.com/yury-rd/fmri-transfer-data)
repository. The optional `data/` submodule requires authorized GitHub access.
See the [README](README.md#use-the-private-data-checkout) for setup.

`msc_sessions123.npz` contains prepared Pearson-connectivity fields from the
Midnight Scan Club, [OpenNeuro ds000224, version 1.0.4](https://doi.org/10.18112/openneuro.ds000224.v1.0.4),
distributed under CC0. Retain the dataset attribution when reusing those fields.
Keeping this prepared derivative private does not change the underlying dataset's
license.

`locked_adni_mean_history.npy`, `locked_adni_mean_target.npy`, and
`fixed_coefficients.json` contain ADNI population means and shared model
coefficients. Obtain raw ADNI data and participant-level derived records through
[ADNI](https://adni.loni.usc.edu/) under the applicable data-use agreement.
Private repository access does not replace that agreement.

The software and synthetic generators are distributed under the repository's
[MIT license](LICENSE). The companion repository retains data attribution,
checksums, and the applicable terms for saved aggregate and synthetic outputs.
Numerical data are excluded from public wheel and source distributions.
