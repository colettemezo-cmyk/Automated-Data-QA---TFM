"""Repo and project paths shared by all workflows."""

from pathlib import Path

from bootstrap import ensure_app_on_path

_APP = ensure_app_on_path(__file__)
PROJECT_ROOT = _APP.parent
