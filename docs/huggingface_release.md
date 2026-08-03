# Hugging Face Release

The supported Hugging Face target is a dataset repository containing the same
sanitized public supplement that `deep-research export-public` produces. This is
a code-and-data paper supplement, not a model-weight release.

## Prepare Locally

```bash
python -m pip install -c constraints-public.txt -e ".[api,paper,dev,publish]"
python scripts/publish_huggingface.py \
  --repo-id <hf-namespace>/deep-research-paper-supplement \
  --dry-run \
  --force
```

The dry run builds a manifest-based export, creates a Hugging Face upload folder
with the dataset card at `README.md`, refreshes `PUBLIC_EXPORT_REPORT.json`, and
runs the public release audit on the upload folder.

For release publishing, run from a clean committed git tree. The helper refuses
dirty-tree exports unless `--allow-dirty` is passed for local inspection.

## Publish

Set the token through the environment or an existing `huggingface-cli login`;
do not put tokens in command-line arguments or checked-in files.

```bash
export HF_TOKEN=<your-token>
python scripts/publish_huggingface.py \
  --repo-id <hf-namespace>/deep-research-paper-supplement \
  --force
```

The script creates or updates a Hugging Face `dataset` repo, uploads the audited
folder, and prints the dataset URL. Use `--private` for a private staging repo,
then rerun without `--private` when the GitHub PR has landed and the dataset
card has been inspected.

## What Goes To Hugging Face

- Public source code, tests, docs, and CI metadata.
- Compact query/rubric manifests and derived analysis tables.
- Paper A rebuild source, generated tables/figures, and final PDF.
- `PUBLIC_MANIFEST.json` and `PUBLIC_EXPORT_REPORT.json` with file hashes.

The same exclusions apply as GitHub: no filled `.env`, no raw generated report
forests, no raw judge-verdict directories, no caches, no model weights, no
checkpoints, no private notes, no drafts, and no submission bundles.
