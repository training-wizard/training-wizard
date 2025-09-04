"""This module provides functionality for saving information necessary for reproducing a training-wizard run."""

import os
import platform
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from shlex import split
from shutil import copyfile


def sh(command: str) -> str:
    """Run a shell command.

    Args:
        command: The shell command to run.

    Returns:
        The standard output of the command.
    """
    res = subprocess.run(split(command), check=True, capture_output=True, encoding="utf-8")
    return res.stdout.strip()


def save_reproducibility_information(reproducibility_dir: str | Path = Path()):
    """Save information needed for the reproducibility of a python call in a folder.

    The information includes the date and time, project name, version, repository, commit ID, command,
    host, working directory, and Python interpreter as well as any configuration TOML files that were used.

    Args:
        reproducibility_dir: The directory in which to save the reproducibility information.
    """
    date = sh('date -u +"%Y-%m-%dT%H:%M:%SZ"')
    if Path("pyproject.toml").exists():
        with open("pyproject.toml", "rb") as toml_content:
            pyproject_toml = tomllib.load(toml_content)
            project_name = pyproject_toml["project"]["name"]
            project_version = pyproject_toml["project"]["version"]
        project_info = f"""
        Project = "{project_name}"
        Version = "{project_version}"\
        """
    else:
        project_info = ""
    is_repo = sh("git rev-parse --is-inside-work-tree").lower() == "true"
    if is_repo:
        repository = sh("git remote get-url origin").strip("@gitlab.com:")
        commit_id = sh("git rev-parse HEAD")
        git_info = f"""
        Repository = "{repository}"
        Commit = "{commit_id}"\
        """
    else:
        git_info = ""
    code_info = "" if project_info == git_info == "" else f"""\n        [code]{project_info}{git_info}        """
    command = " ".join(sys.argv)
    hostname = platform.node()
    working_directory = Path.cwd()
    python_interpreter = sys.executable

    main_toml = textwrap.dedent(
        f"""\
        Date = "{date}"{code_info}
        [call]
        Command = "{command}"
        Host = "{hostname}"
        WorkingDirectory = "{working_directory}"
        PythonInterpreter = "{python_interpreter}"
        """
    )

    destination_directory = Path(reproducibility_dir) / "reproducibility"
    os.makedirs(destination_directory, exist_ok=True)
    # Copy the configuration TOML and .py files to the current directory if it is not already there.

    # Source-Destination pairs
    python_copy: list[tuple[Path, str]] = []
    toml_copy: list[tuple[Path, str]] = [(Path(arg), Path(arg).name) for arg in split(command) if arg.endswith(".toml")]

    try:
        for path, _ in toml_copy:
            found: list[tuple[Path, str]] = []
            while len(found) == 0 and working_directory.resolve() in path.resolve().parents:
                path = path.parent
                found = [(file, str(file.relative_to(path))) for file in path.rglob("**/*.py")]
            if len(found) > 100:
                print(f"WARNING: Skipping {path.parent}/*.py because it contains {len(found)} python files.")
            else:
                python_copy.extend(found)
    except Exception as e:
        print(f"WARNING: Reproducibility failed to save python files: {e}")

    for source_path, dest_suffix in [*toml_copy, *python_copy]:
        destination_path = destination_directory / dest_suffix
        assert source_path != destination_path, FileExistsError(f"{source_path} already exists in {destination_path}.")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source_path, destination_path, follow_symlinks=True)

    # Save the main reproducibility information to a file.
    with open(destination_directory / "main.toml", "w") as f:
        f.write(main_toml)
