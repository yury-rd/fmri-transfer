# Personalized Prediction of Differences in fMRI Connectivity Estimates Across Preprocessing Pipelines

Yury Maximov, Aleksandr Volkov, Tamoghna Chattopadhyay, Paul M. Thompson, and Ildar Gabitov

Under review at the [2027 IEEE International Symposium on Biomedical Imaging (ISBI 2027)](https://biomedicalimaging.org/2027/).

This repository contains the code and released data for the paper. The model uses
four coefficients shared across functional connectivity (FC) edges to predict a
participant's FA-minus-BASE response at the third visit. Its inputs are that
visit's BASE connectivity and the A, F, and AF responses from the first two visits.
The target scan's BASE connectivity is therefore required: the model predicts a
preprocessing difference for an available scan.

## Running the examples

Use Python 3.11 or later. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

curl --fail --location \
  https://github.com/yury-rd/fmri-transfer/releases/download/v1.0.0/msc_sessions123.npz \
  --output data/msc_sessions123.npz

python run.py --output results/demo.json
python run.py --mode msc --output results/msc_transfer.json
python run.py --mode summary --output results/summary.json
```

The default mode runs a small synthetic example with seed `20260918`. The `msc`
mode applies the released ADNI coefficients to the included Midnight Scan Club
(MSC) fields without fitting on MSC. The `summary` mode reads the saved result
tables and coefficients. Omit `--output` to print the result as JSON.

The MSC array is distributed as a
[release asset](https://github.com/yury-rd/fmri-transfer/releases/download/v1.0.0/msc_sessions123.npz).
The download above is needed for `--mode msc` and the full test suite. The
[complete project ZIP](https://github.com/yury-rd/fmri-transfer/releases/download/v1.0.0/fmri-transfer-v1.0.0.zip)
already includes it. All other data files are in the repository.

The directory can be moved or renamed; `run.py` locates its data relative to its
own location. Paths passed to `--input` and `--output` are relative to the current
working directory.

## Layout

```text
run.py                 Command-line runner
fmri_response/         Prediction model, controls, and input handling
data/                  MSC fields, ADNI population means, coefficients, and results
tests/                 Numerical and command-line regression checks
```

`fmri_response/response.py` implements the four-coefficient model, participant
holdout, repeated-BASE control, and early-visit agreement. `fmri_response/controls.py`
contains the seven-control comparison and the participant-specific control
envelope. Input checks and the centered ridge solver are in
`fmri_response/_utils.py`.

## Input data

The `responses` array has axes `participant × visit × node × edge`. Node order is
`BASE, A, F, AF, FA`. FC coordinates are the 79,800 lower-triangle entries selected
by `np.tril_indices(400, -1)`, stored as Pearson correlations without a Fisher z
transform. The MSC array has shape `(10, 3, 5, 79800)`, with participants MSC01
through MSC10 and sessions 1 through 3 in order. It retains the release's float64
values.

BASE is the motion-24 residual field. A removes the effective five-component
anatomical nuisance subspace recovered from the baseline/joint-regression
difference. F is the finite-record DCT-II high-pass projection at 0.01 Hz.
AF applies F after A, and FA applies A after F. The effective A basis stays fixed
when it is applied after F.

For a local analysis, provide an NPZ file containing `responses`, `reliability`,
and `design`:

```sh
python run.py --input local_fields.npz --output results/local_response.json
```

Use at least five participants, exactly three visits, and the five nodes above.
The edge count can differ for a local analysis. The `reliability` and `design`
arrays must be finite participant-by-feature matrices. They supply the reliability
and operator-design controls; specify their construction in the study's measurement
protocol and keep target outcomes out of them. The synthetic example generates
these controls randomly.

## Released results

The MSC mean squared errors are:

| Predictor | Mean squared error |
| --- | ---: |
| Personalized model | 0.00433179 |
| Repeated BASE | 0.00457413 |
| Fixed ADNI mean history | 0.00476962 |
| ADNI mean target | 0.00532586 |

`data/msc_expected.json` records full-precision results and per-participant losses.
The MSC run reports the largest difference from the saved mean losses. MSC was
used during earlier method development, so this analysis is a retrospective
transfer with fixed coefficients, not an independent validation cohort.

`data/adni_aggregate.json` contains the seven-control comparison.
`data/fixed_coefficients.json` holds the `personal4` and `base3` coefficient
vectors. The two NPY files contain ADNI population means from 69 training
participants, with no participant axis. ADNI participant-level records are not
included. Data sources and reuse terms are in [DATA_LICENSES.md](DATA_LICENSES.md).

## Tests

```sh
python -m unittest discover -s tests -v
```

The tests check participant holdout, the three run modes, and paths after moving
the project. The MSC regression tolerance is an absolute error of `1e-12` in
float64 arithmetic.

## Source and citation

The code and prepared data are available from
[yury-rd/fmri-transfer](https://github.com/yury-rd/fmri-transfer).
The [v1.0.0 release](https://github.com/yury-rd/fmri-transfer/releases/tag/v1.0.0)
includes a complete project ZIP and the MSC array. Data checksums are listed in
[data/SHA256SUMS](data/SHA256SUMS).

See [CITATION.md](CITATION.md) for the manuscript citation. Code is distributed
under the [MIT license](LICENSE).
