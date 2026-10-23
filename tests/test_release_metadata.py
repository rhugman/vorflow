import ast
import os
from pathlib import Path
import re
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_version_is_valid_pep440():
    version = Version(_pyproject()["project"]["version"])
    assert version > Version("0.0.1")


def test_licence_uses_pep639_metadata():
    data = _pyproject()
    project = data["project"]
    assert project["license"] == "MIT"
    for name in project["license-files"]:
        assert (ROOT / name).is_file()
    # PEP 639: a licence expression must not be combined with licence classifiers.
    assert not any(c.startswith("License ::") for c in project["classifiers"])
    # setuptools only writes License-Expression from 77.0.3 onward.
    setuptools = next(
        Requirement(r) for r in data["build-system"]["requires"]
        if Requirement(r).name == "setuptools"
    )
    assert not setuptools.specifier.contains("77.0.2")


def test_people_have_names_and_no_placeholder_emails():
    project = _pyproject()["project"]
    for person in project["authors"] + project["maintainers"]:
        assert person.get("name")
        assert "example.com" not in person.get("email", "")


def test_project_urls_point_to_upstream():
    upstream = ("https://github.com/rhugman/vorflow", "https://rhugman.github.io/vorflow/")
    for url in _pyproject()["project"]["urls"].values():
        assert url.startswith(upstream)


def test_minimum_dependency_job_pins_every_runtime_floor():
    workflow = (ROOT / ".github" / "workflows" / "python-app.yml").read_text(
        encoding="utf-8"
    )
    pins = dict(re.findall(r"^\s+([A-Za-z0-9_.-]+)==(\S+)$", workflow, re.M))
    for value in _pyproject()["project"]["dependencies"]:
        requirement = Requirement(value)
        assert requirement.name in pins, f"{requirement.name} not pinned in CI"
        assert requirement.specifier.contains(pins[requirement.name])


def test_conda_environment_lists_every_runtime_dependency():
    environment = (ROOT / "etc" / "environment.yml").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\s+-\s+([A-Za-z0-9_.-]+)", environment, re.M))
    for value in _pyproject()["project"]["dependencies"]:
        assert Requirement(value).name in listed


def test_release_script_imports_are_declared():
    """Third-party imports in scripts/ must be installable from the dev extra."""
    project = _pyproject()["project"]
    declared = {
        Requirement(r).name
        for r in project["dependencies"] + project["optional-dependencies"]["dev"]
    }
    local_modules = {script.stem for script in (ROOT / "scripts").glob("*.py")}
    local_modules.update(package.name for package in (ROOT / "src").iterdir()
                         if (package / "__init__.py").is_file())
    for script in (ROOT / "scripts").glob("*.py"):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        # An import in a try body with an ImportError handler is optional; its
        # fallback in the handler (e.g. tomli for tomllib) is still checked.
        optional = {
            id(statement)
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(
                isinstance(h.type, ast.Name)
                and h.type.id in {"ImportError", "ModuleNotFoundError"}
                for h in node.handlers
            )
            for statement in node.body
        }
        for node in ast.walk(tree):
            if id(node) in optional:
                continue
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top == "__future__" or top in local_modules:
                    continue
                assert top in declared, f"{script.name} imports undeclared {top}"


def test_public_docs_link_to_upstream():
    for name in ("README.md", "CHANGELOG.md"):
        content = (ROOT / name).read_text(encoding="utf-8")
        assert "https://github.com/oscarfasanchez/vorflow_os" not in content
        assert "https://github.com/rhugman/vorflow" in content


def test_source_fallback_is_not_a_duplicate_release_version():
    source = (ROOT / "src" / "vorflow" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert '__version__ = "0+unknown"' in source
    assert '__version__ = "0.0.2"' not in source


def test_basic_usage_script_runs_from_a_clean_directory(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "basic_usage.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Generated " in result.stdout
    assert " Voronoi cells" in result.stdout
