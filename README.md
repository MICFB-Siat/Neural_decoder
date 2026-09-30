# Neural decoder

Code for eigenmode-based brain decoding and encoding, including an inference demo and analysis workflows for the paper figures.

## Repository structure

| Directory | Contents |
| --- | --- |
| [demo/](demo/) | Inference demo, sample data, and optional encoder pretraining |
| [full/](full/) | Analysis workflows for Figures 2–6 and Extended Data S3–S8 |

The workflow directories and entry points are listed in [full/workflows.json](full/workflows.json).

## Getting started

For the inference demo:

```bash
cd demo
pip install -r requirements.txt
python download_assets.py
python verify_assets.py
python run_inference.py --task classification
```

For the analysis workflows, run the resource downloader from the repository root using Python 3.11 or later:

```bash
pip install 'huggingface_hub>=0.34'
python full/download_assets.py --list
```

Download the resources you need with `--scope <resource-prefix>`, install the dependencies for the selected workflow, and use its entry point's `--help` to view the available options. Configure external data paths for workflows that require your own preprocessed inputs.

## Data and model weights

Model weights and packaged analysis resources are available on [Hugging Face](https://huggingface.co/SSp1ash/Eigenbrain). The download scripts place resources in the directories expected by the code. [full/asset_manifest.json](full/asset_manifest.json) records the analysis resource paths, sizes, checksums, and repository revision.
