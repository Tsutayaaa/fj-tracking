# fj-tracking: Bacterial Tracking Pipeline

**English** | [简体中文](README.md)

A pipeline for segmenting, tracking, and measuring bacterial cells in phase-contrast microscopy videos, developed for the analysis of *Flavobacterium johnsoniae* motility. Two segmentation backends are available: **Omnipose** and **MiSiC**.

The repository includes source code, installation configuration, and both model weight files. It does not include sample videos, experimental datasets, generated results, or installed runtime environments. Users provide their own videos.

## Get the code

The pipeline is available on the `tracking1.0` branch:

```bash
git clone --branch tracking1.0 https://github.com/Tsutayaaa/fj-tracking.git
cd fj-tracking
```

## Install

Install Miniconda first and make sure `conda` is available in your terminal. From the repository root, run:

```bash
python install.py
```

The installer prepares two separate Python 3.10 environments:

| Environment | Purpose |
|---|---|
| `omnipose` | Omnipose segmentation and all downstream processing |
| `misic` | MiSiC segmentation with TensorFlow |

The default installation uses CPU inference. On Linux or Windows with a compatible NVIDIA GPU and driver, install the CUDA 12.1 build of PyTorch instead:

```bash
python install.py --cuda
```

To install only one segmentation backend, select it explicitly:

```bash
python install.py omni
```

or:

```bash
python install.py misic
```

A MiSiC-only installation still prepares the downstream processing dependencies in the `omnipose` environment, but does not install the Omnipose inference dependencies.

The installer reuses existing environments with the selected names and installs the required dependency versions. Existing environments must use Python 3.10. To use different environment names, set `OMNI_ENV` and `MISIC_ENV` before installation and keep the same settings when running the pipeline.

### Dependency configuration

- `environment.yml` and `environment-misic.yml` define the base Python environments.
- `requirements.txt` defines the shared downstream dependencies.
- `requirements-omni.txt` adds Omnipose, PyTorch, and compatible image-reading dependencies.
- `requirements-misic.txt` defines the TensorFlow runtime and MiSiC preprocessing dependencies.

Use `install.py` to complete installation; creating the base conda environments alone does not install the full pipeline.

MiSiC 1.1.5 declares an older TensorFlow 2.4.1 dependency. The installer explicitly installs TensorFlow 2.10, then installs MiSiC with `--no-deps` to avoid downgrading that runtime. Apple Silicon uses `tensorflow-macos`; other supported platforms use `tensorflow`.

Internet access is required to install dependencies. Platforms without compatible TensorFlow 2.10 distributions, including Linux ARM, are outside the intended installation targets of this configuration.

## Check the installation

The installer finishes by checking imports and loading the local weights for the selected segmentation backends. You can repeat this check at any time:

```bash
python main.py --check
```

To check only one backend:

```bash
python main.py --check --check-mode omni
```

or:

```bash
python main.py --check --check-mode misic
```

## Run

Command syntax:

```text
python main.py <video_path> [both|omni|misic] [--device auto|cpu|cuda] [--output-dir <output_directory>]
```

The input video may be stored at any readable path. Quote paths containing spaces. If the backend argument is omitted, both segmentation pipelines run.

| Option | Behavior |
|---|---|
| `both` | Run Omnipose and MiSiC; this is the default |
| `omni` | Run only the Omnipose pipeline |
| `misic` | Run only the MiSiC pipeline |
| `--device auto` | Use CUDA for Omnipose when available; otherwise use CPU |
| `--device cpu` | Run Omnipose on CPU |
| `--device cuda` | Require an available CUDA device for Omnipose |
| `--output-dir` | Set the complete results directory for this video |

MiSiC runs on CPU. The launcher uses the Python standard library and dispatches processing to the configured conda environments, so you do not need to activate a processing environment before running `main.py`.

Without `--output-dir`, results are written to `outputs/<video_stem>/` under the repository root. For all command-line options, run:

```bash
python main.py --help
```

## Processing stages

**Omnipose:** segmentation → temporal splitting of merged instances → mask cleaning → tracking → feature extraction → animation.

**MiSiC:** segmentation → mask cleaning → tracking → feature extraction → animation.

The tracker associates cell centroids using Hungarian assignment and retains unmatched tracks for a configurable number of frames. The animation displays cell masks and centroid trajectories, with trajectory lines interrupted across detection gaps.

## Outputs

Each selected backend produces the following files. `{backend}` is either `omni` or `misic`.

| File | Contents |
|---|---|
| `tracks_animation_{backend}.mp4` | Video with colored cell masks and growing centroid trajectories |
| `tracks_{backend}.csv` | Per-detection track IDs, frame indices, labels, positions, and morphology |
| `tracks_features_{backend}.csv` | Per-track motion and morphology summaries |
| `frames_features_{backend}.csv` | Per-detection morphology, velocity, direction, and validity flags |
| `run_manifest.json` | Input, parameter, and model fingerprints used to protect resumed runs |

Intermediate instance masks are also stored in the results directory. If no cells pass the area filter, the pipeline writes empty tables with fixed headers and a video without cell overlays.

The detection table contains:

```text
track_id, frame, label, area, centroid_x, centroid_y, major_axis_length, minor_axis_length
```

For compatibility with existing outputs, **`centroid_x` is the row coordinate (image y), and `centroid_y` is the column coordinate (image x)**.

Track summaries include speed statistics, length and width statistics, net displacement, path length, persistence, reversal counts, reversal rate, and the number of outlier steps.

## Resume an interrupted run

Rerunning the same input with the same parameters and model weights reuses complete segmentation masks and recomputes downstream results.

The run manifest checks the video content, recorded parameters, and model files. If these differ from an existing run, the launcher requires a new output directory to prevent reuse of incompatible masks.

Masks are written through temporary files to avoid leaving a partially written final mask after an interruption. If an existing mask is corrupted or has an incompatible shape or data type, the pipeline reports its path. Delete the reported mask and rerun segmentation.

## Included models

Both segmentation backends load the weights directly from this repository:

| File | Purpose |
|---|---|
| `models/bact_phase_omnitorch_0` | Omnipose bacterial phase-contrast segmentation |
| `models/MiSiCv2.h5` | MiSiC segmentation |
| `models/SHA256SUMS` | SHA-256 checksums of the bundled weight files |

No manual copying into a user model cache or an installed package directory is required. Download the actual model files rather than saving their GitHub HTML pages.

## Parameters and measurement conventions

Default pixel-based parameters are scaled relative to a reference video width of **1824 pixels**. These include the expected cell width, minimum cell area, maximum matching distance, and speed outlier threshold. This assumes proportional resizing under the same imaging conditions. Cropping or changing microscope magnification requires parameter recalibration. Environment-variable settings for individual processing stages are documented in their source files.

| Measurement | Unit or interpretation |
|---|---|
| Speed | Pixels per frame |
| Length and width | Pixels; derived from equivalent-ellipse axes |
| Area | Square pixels |
| Duration | Frames, including gaps between the first and last detection |
| Reversal rate | Reversals per frame |

The pipeline does not automatically convert measurements to micrometers or seconds, and it does not use individual video-frame timestamps. Uneven acquisition intervals require additional time calibration before comparing physical velocities.

### Path length and persistence

Path length is the sum of displacement magnitudes between valid successive observations. The calculation accounts for detection gaps by summing displacements rather than normalized speeds.

Across a gap, only the straight-line displacement between the observed endpoints is available; motion within the missing interval cannot be reconstructed.

Speed outliers are excluded from path length, while net displacement still uses the first and last positions of the full track. Consequently, `persistence = net_displacement / path_length` can exceed 1 when outlier steps are present. Consider `n_outlier` when filtering or interpreting these results.

Reversal counts use the original direction-smoothing and threshold method. They have not been validated against manually annotated reversal events. Likewise, `major_axis_length` is an equivalent-ellipse measurement, not the centerline length of a curved cell.

## Validation scope

Packaging checks were performed in temporary Python 3.10 environments on macOS Apple Silicon. They covered dependency imports, loading both bundled models, displacement-based path calculation, empty feature tables, mask frame-index checks, and rejection of cached results after an input change.

The full video analysis pipeline was not executed during these checks. Linux, Windows, and CUDA were not tested on actual hardware. Temporary test environments and test inputs are not included in the repository. Segmentation, splitting, tracking accuracy, and downstream biological measurements still require validation for the intended data.

## Upstream projects

- [Omnipose](https://github.com/kevinjohncutler/omnipose) and its [model-loading documentation](https://omnipose.readthedocs.io/settings.html)
- [MiSiC 1.1.5](https://pypi.org/project/misic/1.1.5/)
- [PyTorch installation instructions for previous releases](https://pytorch.org/get-started/previous-versions/)
