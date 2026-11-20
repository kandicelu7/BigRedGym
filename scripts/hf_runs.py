"""Sync training runs through a private Hugging Face model repo.

The molab notebook uploads each finished run; your PC fetches them into logs/.
Needs a Hugging Face write token in HF_TOKEN (or a prior `hf auth login`).
Run without installing anything into the project:

    uv run --no-project --with huggingface_hub python scripts/hf_runs.py check
    uv run --no-project --with huggingface_hub python scripts/hf_runs.py sync --iterations 1500
    uv run --no-project --with huggingface_hub python scripts/hf_runs.py list
    uv run --no-project --with huggingface_hub python scripts/hf_runs.py fetch

Runs are stored as <experiment>/<run>/ in the repo, exactly as under logs/,
so `fetch` drops them straight into place. Only every --every-th checkpoint
(plus the last) is uploaded, along with the run's configs and vitals.
"""

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "bigredgym-runs"


def iteration(path):
    return int(re.search(r"model_(\d+)\.pt", path.name).group(1))


def repo_id(api, name):
    return name if "/" in name else f"{api.whoami()['name']}/{name}"


def remote_runs(api, repo):
    """{'experiment/run': {checkpoint iterations}} for runs already uploaded."""
    runs = {}
    if not api.repo_exists(repo):
        return runs
    for f in api.list_repo_files(repo):
        m = re.fullmatch(r"([^/]+/[^/]+)/model_(\d+)\.pt", f)
        if m:
            runs.setdefault(m.group(1), set()).add(int(m.group(2)))
    return runs


def local_runs(logs, iterations):
    """Run dirs under logs/ that reached the final checkpoint."""
    return sorted(p.parent for p in logs.glob(f"*/*/model_{iterations}.pt"))


def upload(api, repo, run, every):
    ckpts = sorted(run.glob("model_*.pt"), key=iteration)
    skip = [p.name for p in ckpts[:-1] if iteration(p) % every]
    api.upload_folder(
        repo_id=repo,
        folder_path=run,
        path_in_repo=f"{run.parent.name}/{run.name}",
        ignore_patterns=skip,
        commit_message=f"{run.parent.name}/{run.name}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["check", "sync", "list", "fetch"])
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="repo name, or user/name (default: <you>/bigredgym-runs)")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs")
    parser.add_argument("--iterations", type=int, default=1500,
                        help="sync: final iteration a finished run has reached")
    parser.add_argument("--every", type=int, default=250,
                        help="sync: upload checkpoints at multiples of this")
    parser.add_argument("--prefix", default="",
                        help="fetch: only experiments starting with this, e.g. go2osc_wide_")
    args = parser.parse_args()

    api = HfApi()
    repo = repo_id(api, args.repo)

    if args.command == "check":
        api.create_repo(repo, private=True, exist_ok=True)
        print(f"ok: logged in as {api.whoami()['name']}, private repo {repo} ready, "
              f"{len(remote_runs(api, repo))} run(s) stored")
    elif args.command == "list":
        for run, its in sorted(remote_runs(api, repo).items()):
            print(f"{run}  (iters {min(its)}..{max(its)})")
    elif args.command == "sync":
        api.create_repo(repo, private=True, exist_ok=True)
        stored = remote_runs(api, repo)
        failed = 0
        for run in local_runs(args.logs, args.iterations):
            name = f"{run.parent.name}/{run.name}"
            if args.iterations in stored.get(name, ()):
                continue
            try:
                upload(api, repo, run, args.every)
                print(f"uploaded {name}", flush=True)
            except Exception as e:  # keep going; the next sync retries it
                failed += 1
                print(f"!!! upload failed for {name}: {e}", flush=True)
        sys.exit(1 if failed else 0)
    elif args.command == "fetch":
        patterns = [f"{args.prefix}*"] if args.prefix else None
        before = set(local_runs(args.logs, args.iterations))
        snapshot_download(repo, local_dir=args.logs, allow_patterns=patterns)
        new = sorted(set(local_runs(args.logs, args.iterations)) - before)
        print(f"{len(new)} new run(s) in {args.logs}:")
        for run in new:
            print(f"  {run.parent.name}/{run.name}")


if __name__ == "__main__":
    main()
