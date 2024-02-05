"""Module to run pre commit checks for staged files."""
from subprocess import check_call
import os
import fire
import git

PRECOMMIT_CHECK_EXTENSIONS = [".py"]
PRE_COMMIT_HOOKS = [
    "check-case-conflict",
    "check-merge-conflict",
    "end-of-file-fixer",
    "trailing-whitespace",
    "prettier",
    "black",
    "pylint",
]


def _run_pre_commit_check(hook: str, files: str) -> None:
    """Run pre commit check on selected files

    Args:
        hook (_type_): name of hook.
        files (_type_): selected files.
    """
    check_call(["pre-commit", "run", hook, "-v", "--files", *files])


class Command:
    """Commands to be run by fire lib."""

    def run(self):
        """run linter check by running all PRECOMMIT HOOKS."""
        repo_path = "."

        repo = git.Repo(repo_path)

        # Get the list of changed or staged files
        changed_or_staged_files = [item.a_path for item in repo.index.diff(None)]
        changed_or_staged_files.extend([item.a_path for item in repo.index.diff("HEAD")])

        res = [item for item in changed_or_staged_files if os.path.splitext(item)[1] in PRECOMMIT_CHECK_EXTENSIONS]

        for hook in PRE_COMMIT_HOOKS:
            # for cur_file in res:
            _run_pre_commit_check(hook, res)


if __name__ == "__main__":
    fire.Fire(Command)
    print("All looks good. 👍 ")
