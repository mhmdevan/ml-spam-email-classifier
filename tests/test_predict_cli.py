# tests/test_predict_cli.py

import sys
import subprocess
from pathlib import Path

from src import train_spam_classifier


def run_cli(text: str) -> str:
    """
    Run the CLI prediction script and return its stdout as text.
    """
    project_root: Path = train_spam_classifier.PROJECT_ROOT
    cmd = [
        sys.executable,
        "-m",
        "src.predict_spam",
        "--text",
        text,
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_predict_cli_spam():
    spam_text = "Congratulations, you have won a free iPhone. Click here now!"
    stdout = run_cli(spam_text)
    # Expect it to be classified as SPAM
    assert "Predicted label: SPAM" in stdout


def test_predict_cli_ham():
    ham_text = "Hi John, can we reschedule our meeting to tomorrow afternoon?"
    stdout = run_cli(ham_text)
    # Expect it to be classified as HAM
    assert "Predicted label: HAM" in stdout
