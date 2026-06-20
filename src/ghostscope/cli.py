"""ghostscope command-line interface.

Subcommands:
    gen-data   generate a synthetic labeled flow dataset
    train      train a detector on flow records
    detect     score new flow records and emit explainable alerts
    evaluate   score labeled records and report detection metrics
    info       inspect a trained model bundle
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .detector import VALID_MODELS, Detector, train_detector
from .features import FeaturePipeline
from .logging_utils import configure_logging
from .metrics import evaluate as evaluate_metrics
from .schema import FEATURE_COLUMNS, LABEL_COLUMN, SchemaError

app = typer.Typer(
    add_completion=False,
    help="ghostscope: AI-based network intrusion detection for authorized "
    "defensive monitoring.",
    no_args_is_help=True,
)
console = Console()


def _fail(message: str) -> None:
    console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code=1)


@app.command()
def gen_data(
    out: str = typer.Option("data/flows.csv", "--out", "-o", help="Output CSV path."),
    n_normal: int = typer.Option(2000, help="Number of normal flow records."),
    n_port_scan: int = typer.Option(80, help="Number of port-scan anomalies."),
    n_ddos: int = typer.Option(80, help="Number of DDoS-burst anomalies."),
    n_exfil: int = typer.Option(40, help="Number of exfiltration anomalies."),
    seed: int = typer.Option(1337, help="RNG seed for deterministic output."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate a deterministic synthetic labeled flow dataset."""
    import os

    from .datagen import write_dataset

    configure_logging(verbose)
    directory = os.path.dirname(out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df = write_dataset(
        out,
        n_normal=n_normal,
        n_port_scan=n_port_scan,
        n_ddos=n_ddos,
        n_exfil=n_exfil,
        seed=seed,
    )
    n_anom = int(df[LABEL_COLUMN].sum())
    console.print(
        f"[green]wrote[/green] {len(df)} records to {out} "
        f"({n_anom} anomalies, {len(df) - n_anom} normal, seed={seed})"
    )


@app.command()
def train(
    data: str = typer.Argument(..., help="Training CSV of flow records."),
    model_dir: str = typer.Option(
        "models/ghostscope", "--model-dir", "-m", help="Output bundle directory."
    ),
    model: str = typer.Option(
        "iforest", "--model", help=f"Detector backend, one of {VALID_MODELS}."
    ),
    contamination: float = typer.Option(
        0.02, help="Expected anomaly fraction (IsolationForest)."
    ),
    percentile: float = typer.Option(
        99.0, help="Percentile of training scores used as the alert threshold."
    ),
    epochs: int = typer.Option(40, help="Autoencoder training epochs."),
    seed: int = typer.Option(1337, help="Global seed for determinism."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train an anomaly detector and persist the model bundle."""
    configure_logging(verbose)
    if model not in VALID_MODELS:
        _fail(f"unknown model '{model}', choose one of {VALID_MODELS}")
    try:
        df = FeaturePipeline.read_csv(data)
        detector = train_detector(
            df,
            model_type=model,
            contamination=contamination,
            percentile=percentile,
            seed=seed,
            epochs=epochs,
        )
    except SchemaError as exc:
        _fail(str(exc))
    except ImportError as exc:
        _fail(str(exc))
    detector.save(model_dir)
    console.print(
        f"[green]trained[/green] {model} detector on {len(df)} records "
        f"-> {model_dir} (threshold={detector.threshold:.4f}, "
        f"features={len(detector.metadata.feature_names)})"
    )


def _alerts_table(
    df, scores, flags, explanations, schema_label: str, limit: int = 0
) -> Table:
    """Build the alerts table. A positive ``limit`` truncates printed rows."""
    has_label = schema_label in df.columns
    table = Table(title="ghostscope alerts", show_lines=False)
    table.add_column("row", justify="right", style="cyan")
    table.add_column("score", justify="right")
    table.add_column("top contributing features", style="yellow")
    if has_label:
        table.add_column("label", justify="right")

    shown = 0
    for i in range(len(df)):
        if not flags[i]:
            continue
        if limit > 0 and shown >= limit:
            break
        contribs = ", ".join(
            f"{c['feature']}={c['value']}" for c in explanations[i]
        )
        cells = [str(i), f"{scores[i]:.4f}", contribs]
        if has_label:
            cells.append(str(int(df.iloc[i][schema_label])))
        table.add_row(*cells)
        shown += 1
    return table


@app.command()
def detect(
    data: str = typer.Argument(..., help="CSV of flow records to score."),
    model_dir: str = typer.Option(
        "models/ghostscope", "--model-dir", "-m", help="Trained bundle directory."
    ),
    top_k: int = typer.Option(3, help="Top contributing features per alert."),
    json_out: str | None = typer.Option(
        None, "--json-out", help="Write alerts as JSON to this path."
    ),
    limit: int = typer.Option(
        50, help="Max alerts to print in the table (0 = all)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score flow records and emit explainable anomaly alerts."""
    configure_logging(verbose)
    try:
        detector = Detector.load(model_dir)
        df = FeaturePipeline.read_csv(data)
        scores, flags = detector.predict(df)
        explanations = detector.explain(df, top_k=top_k)
    except (SchemaError, FileNotFoundError) as exc:
        _fail(str(exc))
    except ImportError as exc:
        _fail(str(exc))

    n_alerts = int(flags.sum())
    table = _alerts_table(df, scores, flags, explanations, LABEL_COLUMN, limit=limit)
    console.print(table)
    console.print(
        f"[bold]{n_alerts}[/bold] alert(s) of {len(df)} records "
        f"(threshold={detector.threshold:.4f}, model={detector.model_type})"
    )

    if json_out:
        alerts = []
        for i in range(len(df)):
            if not flags[i]:
                continue
            alerts.append(
                {
                    "row": i,
                    "score": float(scores[i]),
                    "contributors": explanations[i],
                }
            )
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_type": detector.model_type,
                    "threshold": detector.threshold,
                    "n_records": len(df),
                    "n_alerts": n_alerts,
                    "alerts": alerts,
                },
                fh,
                indent=2,
            )
        console.print(f"[green]wrote[/green] alerts JSON to {json_out}")


@app.command()
def evaluate(
    data: str = typer.Argument(..., help="Labeled CSV (must include 'label')."),
    model_dir: str = typer.Option(
        "models/ghostscope", "--model-dir", "-m", help="Trained bundle directory."
    ),
    json_out: str | None = typer.Option(
        None, "--json-out", help="Write metrics as JSON to this path."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a trained detector against labeled flow records."""
    configure_logging(verbose)
    try:
        detector = Detector.load(model_dir)
        df = FeaturePipeline.read_csv(data)
        if LABEL_COLUMN not in df.columns:
            _fail(
                f"evaluate requires a '{LABEL_COLUMN}' column "
                "(0 = normal, 1 = anomaly)."
            )
        scores, flags = detector.predict(df)
    except (SchemaError, FileNotFoundError) as exc:
        _fail(str(exc))
    except ImportError as exc:
        _fail(str(exc))

    y_true = df[LABEL_COLUMN].to_numpy(dtype=int)
    report = evaluate_metrics(y_true, flags, scores)

    table = Table(title="ghostscope evaluation", show_header=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("precision", f"{report.precision:.4f}")
    table.add_row("recall", f"{report.recall:.4f}")
    table.add_row("f1", f"{report.f1:.4f}")
    table.add_row(
        "roc_auc", "n/a" if report.roc_auc is None else f"{report.roc_auc:.4f}"
    )
    console.print(table)

    cm = Table(title="confusion matrix")
    cm.add_column("", style="cyan")
    cm.add_column("pred normal", justify="right")
    cm.add_column("pred anomaly", justify="right")
    cm.add_row("true normal", str(report.true_negatives), str(report.false_positives))
    cm.add_row("true anomaly", str(report.false_negatives), str(report.true_positives))
    console.print(cm)

    if json_out:
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        console.print(f"[green]wrote[/green] metrics JSON to {json_out}")


@app.command()
def info(
    model_dir: str = typer.Option(
        "models/ghostscope", "--model-dir", "-m", help="Trained bundle directory."
    ),
) -> None:
    """Show metadata about a trained model bundle, or the tool version."""
    console.print(f"ghostscope version {__version__}")
    console.print(f"expected feature columns: {', '.join(FEATURE_COLUMNS)}")
    try:
        detector = Detector.load(model_dir)
    except FileNotFoundError:
        console.print(f"[yellow]no model bundle at {model_dir}[/yellow]")
        return
    meta = detector.metadata
    table = Table(title=f"model bundle: {model_dir}", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("model_type", meta.model_type)
    table.add_row("features", ", ".join(meta.feature_names))
    table.add_row("threshold", f"{meta.threshold:.6f}")
    table.add_row("percentile", str(meta.percentile))
    table.add_row("contamination", str(meta.contamination))
    table.add_row("n_train", str(meta.n_train))
    table.add_row("seed", str(meta.seed))
    table.add_row("trained_with", f"ghostscope {meta.ghostscope_version}")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
