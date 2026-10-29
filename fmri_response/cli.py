"""Command-line examples for personalized connectivity prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import response
from .controls import qualify_complete_participant_operator_atlas


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def load_responses(source, *, controls=False):
    """Read finite fields in participant, visit, node, edge order."""
    with np.load(source, allow_pickle=False) as data:
        fields = np.asarray(data["responses"], dtype=np.float64)
        reliability = data["reliability"] if controls else None
        design = data["design"] if controls else None
    minimum_participants = 5 if controls else 1
    if (
        fields.ndim != 4
        or fields.shape[0] < minimum_participants
        or fields.shape[1:3] != (3, 5)
        or fields.shape[3] < 2
        or not np.isfinite(fields).all()
    ):
        raise ValueError(
            "responses must be finite with shape (participants, 3, 5, edges), "
            f"at least {minimum_participants} participants, and at least two edges"
        )
    if controls:
        for name, values in (("reliability", reliability), ("design", design)):
            if (
                values.ndim != 2
                or values.shape[0] != len(fields)
                or values.shape[1] < 1
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"{name} must be a finite participant-by-feature matrix")
    return fields, reliability, design


def msc_transfer(data_dir, input_path=None):
    """Apply the released coefficients without fitting on MSC."""
    source = input_path if input_path is not None else data_dir / "msc_sessions123.npz"
    fields, _, _ = load_responses(source)
    parameters = json.loads((data_dir / "fixed_coefficients.json").read_text())
    x, base, y, _ = response.fields(fields)
    personal = response.losses(x @ np.array(parameters["personal4"]), y)
    repeated_base = response.losses(base @ np.array(parameters["base3"]), y)
    mean_query = x.copy()
    mean_query[:, :, 1:] = np.load(data_dir / "locked_adni_mean_history.npy", allow_pickle=False)
    fixed_history = response.losses(mean_query @ np.array(parameters["personal4"]), y)
    mean_target = np.load(data_dir / "locked_adni_mean_target.npy", allow_pickle=False)
    universal = response.losses(np.broadcast_to(mean_target, y.shape), y)
    result = {
        "n": len(fields),
        "personal_vs_base3": response.contrast(personal, repeated_base),
        "personal_vs_fixed_adni_mean_history": response.contrast(personal, fixed_history),
        "personal_vs_adni_mean_target": response.contrast(personal, universal),
    }
    # The saved reference describes only the bundled MSC participants and sessions.
    if source.resolve() == (data_dir / "msc_sessions123.npz").resolve():
        expected = json.loads((data_dir / "msc_expected.json").read_text())
        result["maximum_mean_loss_replay_error"] = max(
            abs(result[key][metric] - expected[key][metric])
            for key in result if key.startswith("personal_")
            for metric in ("candidate_mean_loss", "reference_mean_loss")
        )
    return result


def response_demo(input_path=None):
    """Compare the personalized model with the controls under participant holdout."""
    if input_path is not None:
        fields, reliability, design = load_responses(input_path, controls=True)
        scope = "Local user-supplied response fields"
    else:
        rng = np.random.default_rng(20260918)
        fields = rng.normal(size=(7, 3, 5, 13))
        reliability = rng.normal(size=(7, 2))
        design = rng.normal(size=(7, 3))
        scope = "Synthetic software example; these are not paper measurements"
    atlas = qualify_complete_participant_operator_atlas(
        fields,
        early_visits=(0, 1),
        future_visits=(2,),
        base_node=0,
        source_nodes=(1, 2, 3),
        target_node=4,
        additive_nodes=(1, 2),
        seen_path_node=3,
        reliability_features=reliability,
        design_summary_features=design,
        ridge=1.0,
    )
    x, base, y, _ = response.fields(fields)
    prediction, _ = response.lopo(x, y)
    baseline, _ = response.lopo(base, y)
    return {
        "scope": scope,
        "participants": len(fields),
        "personal_loss": atlas.atlas_loss,
        "seven_control_losses": atlas.admissible_baseline_losses,
        "participant_oracle_envelope_loss": atlas.best_admissible_baseline_loss,
        "personal_vs_repeated_base": response.contrast(
            response.losses(prediction, y), response.losses(baseline, y)
        ),
    }


def main(project_dir, argv=None):
    data_dir = Path(project_dir).resolve() / "data"
    parser = argparse.ArgumentParser(
        description="Personalized prediction of fMRI connectivity differences."
    )
    parser.add_argument(
        "--mode", choices=("demo", "msc", "summary"), default="demo",
        help="synthetic or local holdout analysis, fixed-coefficient MSC transfer, or saved tables",
    )
    parser.add_argument("--input", type=Path, help="local NPZ fields; see README for array shapes")
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    args = parser.parse_args(argv)
    if args.mode == "summary" and args.input is not None:
        parser.error("--input applies to demo and msc modes")
    try:
        if args.mode == "summary":
            result = {
                path.name: json.loads(path.read_text())
                for path in sorted(data_dir.glob("*.json"))
            }
        else:
            with threadpool_limits(limits=1):
                result = (
                    msc_transfer(data_dir, args.input)
                    if args.mode == "msc"
                    else response_demo(args.input)
                )
        payload = json.dumps(plain(result), indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
            print(json.dumps({"output": str(args.output), "mode": args.mode}))
        else:
            print(payload, end="")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
