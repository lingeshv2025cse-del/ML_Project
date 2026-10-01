from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from config import HIGH_CONFIDENCE_THRESHOLD, MATCH_THRESHOLD, TOP_K_CANDIDATES
from src.entity_matching import resolve_entities
from src.evaluation import detect_label_columns, evaluate_predictions
from src.preprocessing import prepare_source_dataframe, read_csv_file

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

app = Flask(__name__)
app.config["SECRET_KEY"] = "entity-resolution-demo"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(DATA_DIR)

RESULTS_FILE = OUTPUT_DIR / "matches.csv"
UNMATCHED_FILE = OUTPUT_DIR / "unmatched.csv"
MANUAL_REVIEW_FILE = OUTPUT_DIR / "manual_review.csv"
MODEL_PATH = MODELS_DIR / "entity_resolution_model.pkl"


def ensure_project_folders() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


ensure_project_folders()


def _format_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_status(probability: float) -> str:
    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH_CONFIDENCE"
    if probability >= MATCH_THRESHOLD:
        return "POSSIBLE_MATCH"
    return "NO_MATCH"


def _save_results_dataframe(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if df.empty:
        empty_df = pd.DataFrame(columns=[
            "source1_id", "source2_id", "source3_id", "source1_name",
            "source2_name", "source3_name", "name_similarity",
            "address_similarity", "phone_similarity", "email_similarity",
            "website_similarity", "overall_probability", "match_status"
        ])
        empty_df.to_csv(RESULTS_FILE, index=False)
        return
    df.to_csv(RESULTS_FILE, index=False)


def _save_unmatched_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        empty_df = pd.DataFrame(columns=["source1_id", "source1_name", "reason"]) 
        empty_df.to_csv(UNMATCHED_FILE, index=False)
        return
    df.to_csv(UNMATCHED_FILE, index=False)


def _load_latest_results() -> pd.DataFrame:
    if RESULTS_FILE.exists():
        try:
            return pd.read_csv(RESULTS_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _determine_training_label_source(source1_df: pd.DataFrame,
                                    source2_df: pd.DataFrame,
                                    source3_df: pd.DataFrame) -> Tuple[bool, str]:
    label_sources = [source1_df, source2_df, source3_df]
    for df in label_sources:
        for col in df.columns:
            normalized = str(col).strip().lower()
            if normalized in {"match", "is_match", "label", "target", "match_label"}:
                return True, normalized
    return False, "weak_labels_generated"


def _read_source_files(source1_path: str, source2_path: str, source3_path: str):
    source1 = prepare_source_dataframe(read_csv_file(source1_path), "source1")
    source2 = prepare_source_dataframe(read_csv_file(source2_path), "source2")
    source3 = prepare_source_dataframe(read_csv_file(source3_path), "source3")
    return source1, source2, source3


def process_datasets(source1_path: str, source2_path: str, source3_path: str) -> Dict[str, Any]:
    source1_df, source2_df, source3_df = _read_source_files(source1_path, source2_path, source3_path)

    # Detect whether the uploaded data already contains a supervised label column.
    has_label, label_source = _determine_training_label_source(source1_df, source2_df, source3_df)

    results, unmatched, model, notes, metrics = resolve_entities(
        source1_df,
        source2_df,
        source3_df,
        model_path=MODEL_PATH,
        use_supervised_labels=has_label,
        label_source=label_source,
    )

    if results.empty:
        empty_df = pd.DataFrame(columns=[
            "source1_id", "source2_id", "source3_id", "source1_name", "source2_name",
            "source3_name", "name_similarity", "address_similarity", "phone_similarity",
            "email_similarity", "website_similarity", "overall_probability", "match_status"
        ])
        results = empty_df

    if unmatched.empty:
        unmatched = pd.DataFrame(columns=["source1_id", "source1_name", "reason"])

    _save_results_dataframe(results)
    _save_unmatched_dataframe(unmatched)

    analytics_payload = {
        "has_label": has_label,
        "label_source": label_source,
        "training_notes": notes,
        "metrics": metrics,
        "records": {
            "source1": len(source1_df),
            "source2": len(source2_df),
            "source3": len(source3_df),
        },
        "match_stats": {
            "high_confidence": int((results["match_status"] == "HIGH_CONFIDENCE").sum()) if not results.empty else 0,
            "possible": int((results["match_status"] == "POSSIBLE_MATCH").sum()) if not results.empty else 0,
            "unmatched": len(unmatched) if not unmatched.empty else 0,
        },
        "similarity_stats": {
            "avg_name_similarity": _format_float(results["name_similarity"].mean()) if not results.empty else 0.0,
            "avg_address_similarity": _format_float(results["address_similarity"].mean()) if not results.empty else 0.0,
            "avg_phone_similarity": _format_float(results["phone_similarity"].mean()) if not results.empty else 0.0,
            "avg_email_similarity": _format_float(results["email_similarity"].mean()) if not results.empty else 0.0,
        },
    }

    with open(BASE_DIR / "output" / "analytics.json", "w", encoding="utf-8") as handle:
        json.dump(analytics_payload, handle, indent=2)

    return {
        "results": results,
        "unmatched": unmatched,
        "records": {
            "source1": len(source1_df),
            "source2": len(source2_df),
            "source3": len(source3_df),
        },
        "training_note": notes,
        "metrics": metrics,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        uploaded_files = {
            "source1": request.files.get("source1_file"),
            "source2": request.files.get("source2_file"),
            "source3": request.files.get("source3_file"),
        }

        for key, file_obj in uploaded_files.items():
            if file_obj is None or file_obj.filename == "":
                flash(f"Please upload a CSV file for {key.replace('_', ' ').upper()}.", "error")
                return render_template("upload.html")

            if not file_obj.filename.lower().endswith(".csv"):
                flash(f"Invalid file type for {key.replace('_', ' ').upper()}. Please upload a CSV file.", "error")
                return render_template("upload.html")

        save_paths = {}
        for name, file_obj in uploaded_files.items():
            destination = DATA_DIR / f"{name}.csv"
            file_obj.save(destination)
            save_paths[name] = str(destination)

        try:
            summary = process_datasets(save_paths["source1"], save_paths["source2"], save_paths["source3"])
            return render_template("results.html", summary=summary, total_rows=0)
        except Exception as exc:
            flash(f"Entity resolution failed: {exc}", "error")
            return render_template("error.html", message=str(exc))

    return render_template("upload.html")


@app.route("/run-sample")
def run_sample_datasets():
    try:
        sample_paths = {
            "source1": str(DATA_DIR / "source1.csv"),
            "source2": str(DATA_DIR / "source2.csv"),
            "source3": str(DATA_DIR / "source3.csv"),
        }
        summary = process_datasets(sample_paths["source1"], sample_paths["source2"], sample_paths["source3"])
        return render_template("results.html", summary=summary, total_rows=0)
    except Exception as exc:
        flash(f"Sample dataset failed to run: {exc}", "error")
        return render_template("error.html", message=str(exc))


@app.route("/results")
def results_page():
    results_df = _load_latest_results()
    summary = {
        "results": results_df,
        "records": {
            "source1": int(results_df["source1_id"].nunique()) if not results_df.empty else 0,
            "source2": int(results_df["source2_id"].nunique()) if not results_df.empty else 0,
            "source3": int(results_df["source3_id"].nunique()) if not results_df.empty else 0,
        },
        "training_note": "No results have been generated yet.",
        "metrics": {},
    }

    if (BASE_DIR / "output" / "analytics.json").exists():
        try:
            with open(BASE_DIR / "output" / "analytics.json", "r", encoding="utf-8") as handle:
                summary_payload = json.load(handle)
                summary["records"] = summary_payload.get("records", summary["records"])
                summary["training_note"] = summary_payload.get("training_notes", "")
                summary["metrics"] = summary_payload.get("metrics", {})
        except Exception:
            pass

    return render_template("results.html", summary=summary, total_rows=len(results_df))


@app.route("/analytics")
def analytics_page():
    results_df = _load_latest_results()
    payload = {
        "count": len(results_df),
        "source1_records": int(results_df["source1_id"].nunique()) if not results_df.empty else 0,
        "source2_records": int(results_df["source2_id"].nunique()) if not results_df.empty else 0,
        "source3_records": int(results_df["source3_id"].nunique()) if not results_df.empty else 0,
        "high_confidence": int((results_df["match_status"] == "HIGH_CONFIDENCE").sum()) if not results_df.empty else 0,
        "possible_matches": int((results_df["match_status"] == "POSSIBLE_MATCH").sum()) if not results_df.empty else 0,
        "unmatched": int((results_df["match_status"] == "NO_MATCH").sum()) if not results_df.empty else 0,
    }

    if (BASE_DIR / "output" / "analytics.json").exists():
        try:
            with open(BASE_DIR / "output" / "analytics.json", "r", encoding="utf-8") as handle:
                payload.update(json.load(handle))
        except Exception:
            pass

    return render_template("analytics.html", analytics=payload)


@app.route("/download-results")
def download_results():
    if not RESULTS_FILE.exists():
        flash("No matching results file exists yet.", "error")
        return redirect(url_for("index"))
    return send_file(RESULTS_FILE, as_attachment=True, download_name="results.csv")


@app.route("/download-unmatched")
def download_unmatched():
    if not UNMATCHED_FILE.exists():
        flash("No unmatched file exists yet.", "error")
        return redirect(url_for("index"))
    return send_file(UNMATCHED_FILE, as_attachment=True, download_name="unmatched.csv")


@app.route("/manual-review", methods=["GET", "POST"])
def manual_review():
    if request.method == "POST":
        review_rows = []
        decisions = request.form.getlist("decision")
        candidate_ids = request.form.getlist("candidate_id")
        for candidate_id, decision in zip(candidate_ids, decisions):
            review_rows.append({"candidate_id": candidate_id, "decision": decision, "reviewed_at": pd.Timestamp.now().isoformat()})
        if review_rows:
            review_df = pd.DataFrame(review_rows)
            review_df.to_csv(MANUAL_REVIEW_FILE, index=False)
            flash("Manual review decisions saved successfully.", "success")
        return redirect(url_for("manual_review"))

    results_df = _load_latest_results()
    possible = results_df[results_df["match_status"].isin(["POSSIBLE_MATCH"])].copy() if not results_df.empty else pd.DataFrame()
    return render_template("results.html", summary={"results": possible, "manual_review": True}, total_rows=len(possible))


@app.route("/error")
def generic_error():
    return render_template("error.html", message="An unexpected application error occurred.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
