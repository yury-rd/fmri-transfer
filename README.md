# Personalized Prediction of Differences in fMRI Connectivity Estimates Across Preprocessing Pipelines

Yury Maximov, Aleksandr Volkov, Tamoghna Chattopadhyay, Paul M. Thompson, and Ildar Gabitov

Under review at the [2027 IEEE International Symposium on Biomedical Imaging (ISBI 2027)](https://biomedicalimaging.org/2027/).

This public repository contains the code for the paper. Prepared observations,
trained coefficients, population means, saved results, and saved simulation
outputs are stored separately in the private
[yury-rd/fmri-transfer-data](https://github.com/yury-rd/fmri-transfer-data)
repository. The public repository references that private checkout as the
`data/` Git submodule; an ordinary clone does not download it.

The model uses four coefficients shared across functional connectivity (FC)
edges to predict a participant's FA-minus-BASE response at the third visit.
Its inputs are that visit's BASE connectivity and the A, F, and AF responses
from the first two visits. The target scan's BASE connectivity is required:
the model predicts a preprocessing difference for an available scan.

## Run without private data

Use Python 3.11 or later:

```sh
git clone https://github.com/yury-rd/fmri-transfer.git
cd fmri-transfer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --output results/demo.json
python -m unittest discover -s tests -v
```

The default mode generates a small synthetic example at runtime with seed
`20260918`. Neither this command nor the public tests need private data or
GitHub credentials. Tests generate their own inputs and temporary result
fixtures. Omit `--output` to print the result as JSON.

## Use the private data checkout

Access to the companion repository must be granted separately. With GitHub
credentials that have that access, run:

```sh
git submodule update --init data
gh release download v1.0.0 --repo yury-rd/fmri-transfer-data --pattern msc_sessions123.npz --dir data
python run.py --mode msc --output results/msc_transfer.json
python run.py --mode summary --output results/summary.json
```

The submodule is pinned to a particular private data commit. Updating the public
checkout alone does not change or grant access to the private data. If access
is unavailable, the synthetic example and all public tests remain usable.
The prepared MSC array is stored as a release asset in the private companion,
separately from its Git data branch. Its download uses the authenticated GitHub
CLI (`gh`) and is needed only for MSC mode; summary mode needs only the submodule.

An existing authorized data checkout can be selected explicitly:

```sh
python run.py --mode msc --data-dir /path/to/fmri-transfer-data --output results/msc_transfer.json
python run.py --mode summary --data-dir /path/to/fmri-transfer-data --output results/summary.json
```

This directory must contain the data files directly, as on the companion's
`data` branch. Paths passed to `--input`, `--output`, and `--data-dir` are
relative to the current working directory. The default `data/` path is relative
to `run.py`, so the project can be moved or renamed.
For a separate checkout, use that path as the private release download's
`--dir`, or pass an MSC NPZ stored elsewhere with `--input`.

MSC mode applies the private ADNI coefficients to prepared Midnight Scan Club
fields without fitting on MSC. Its required files are `msc_sessions123.npz`,
`fixed_coefficients.json`, `locked_adni_mean_history.npy`,
`locked_adni_mean_target.npy`, and `msc_expected.json`. It reports the largest
mean-loss difference from the private saved reference. The replay tolerance
is `1e-12` in float64 arithmetic. MSC was used during earlier method development;
this is a retrospective transfer with fixed coefficients.

Summary mode reads exactly `adni_aggregate.json`, `fixed_coefficients.json`,
and `msc_expected.json`. It does not fit a model or regenerate the study
results. Missing private files produce an error explaining how to initialize
the submodule or select `--data-dir`; an absent data checkout is never treated
as an empty successful summary.

## Local inputs

The `responses` array has axes `participant × visit × node × edge`, with node
order `BASE, A, F, AF, FA`. The study's FC coordinates are the lower-triangle
entries selected by `np.tril_indices(400, -1)`, stored as Pearson correlations
without a Fisher z transform. A local analysis may use another edge count.

BASE is the motion-24 residual field. A removes the effective five-component
anatomical nuisance subspace recovered from the baseline/joint-regression
difference. F is the finite-record DCT-II high-pass projection at 0.01 Hz.
AF applies F after A, and FA applies A after F. The effective A basis stays
fixed when applied after F.

Provide a local NPZ containing `responses`, `reliability`, and `design`:

```sh
python run.py --input local-data/local_fields.npz --output results/local_response.json
```

Use at least five participants, exactly three visits, the five nodes above,
and at least two edges. `reliability` and `design` must be finite
participant-by-feature matrices. They supply the reliability and operator-design
controls; specify their construction in the study protocol and keep target
outcomes out of them. The synthetic example generates these controls randomly.
Local demo mode requires only the supplied NPZ, with no private checkout.

## Code layout and tests

```text
run.py                 Command-line runner
fmri_response/         Prediction model, controls, and input handling
data/                  Optional private Git submodule
tests/                 Tests using runtime-generated synthetic inputs
```

`fmri_response/response.py` implements the four-coefficient model, participant
holdout, repeated-BASE control, and early-visit agreement.
`fmri_response/controls.py` contains the seven-control comparison and the
participant-specific control envelope. Input checks and the centered ridge
solver are in `fmri_response/_utils.py`.

Tests compare holdout predictions with direct fits and check input validation,
feature lineage, relocated execution, explicit data paths, and helpful errors
without a private checkout. A synthetic fixed-coefficient fixture exercises
MSC mode without using study observations or saved coefficients. Package builds
exclude the private submodule and numerical data files.

Keep local inputs in the ignored `local-data/` directory and generated outputs
in `results/`. Do not commit either to the public code repository. Data sources
and access terms are described in [DATA_LICENSES.md](DATA_LICENSES.md).

## Citation and license

See [CITATION.md](CITATION.md) for the manuscript citation. Public code is
available from [yury-rd/fmri-transfer](https://github.com/yury-rd/fmri-transfer)
and distributed under the [MIT license](LICENSE). Private data access remains
subject to its applicable terms; access to the code does not grant access to data.
