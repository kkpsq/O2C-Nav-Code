# O2C-Nav

O2C-Nav is a zero-shot vision-and-language navigation (VLN) system for embodied agents. The agent combines language-model reasoning with semantic mapping and waypoint planning, and is evaluated in the Habitat VLN-CE environment on R2R and RxR-style datasets.

This repository contains the research code and evaluation scripts for O2C-Nav. Datasets, simulator assets, model checkpoints, and API credentials are not included in the repository.

## Contents

- `vlnce_baselines/`: O2C-Nav agent, policy, mapping, and Habitat integration.
- `habitat_extensions/`: VLN-CE datasets, tasks, sensors, and measurements.
- `eval_scripts/`: ready-to-run R2R and RxR evaluation entry points.
- `run_mp.py`: multi-process evaluation launcher.
- `requirements.txt`: Python dependencies.

## Installation

### 1. Create the environment

```bash
conda create -n o2cnav python=3.9
conda activate o2cnav
```

### 2. Install Habitat-Sim

The commands below build Habitat-Sim locally. A compatible pre-built package may be used instead if available in your environment.

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout tags/v0.1.7
pip install -r requirements.txt
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" python setup.py install --headless
cd ..
```

### 3. Install Habitat-Lab

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/v0.1.7
```

Before installing Habitat-Lab, remove `tensorflow==1.13.1` from `habitat_baselines/rl/requirements.txt` to avoid conflicts with the rest of the environment. Then install the matching PyTorch build and Habitat:

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0 torchaudio==2.1.0+cu121 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python setup.py develop --all
cd ..
```

### 4. Install GroundingDINO and remaining dependencies

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout 57535c5a79791cb76e36fdb64975271354f10251
pip install -e . --no-build-isolation
pip install git+https://github.com/facebookresearch/segment-anything.git
cd ..

pip install setuptools==58.5.3 meson-python ninja
pip install -r requirements.txt --use-pep517 --no-build-isolation
pip install nltk
```

If the GroundingDINO build fails, retry after upgrading or reinstalling `setuptools` and `wheel` in the active environment.

### **Phrase-to-Class Mapping Optimization**

For more robust matching between GroundingDINO phrases and semantic classes, the project can use the edit-distance implementation described in [CA-Nav](https://github.com/Chenkehan21/CA-Nav-code). Replace the `phrases2classes` method in `GroundingDINO/groundingdino/util/inference.py` with the following after installing `nltk`:

```python
from nltk.metrics import edit_distance

@staticmethod
def phrases2classes(phrases, classes):
    class_ids = []
    for phrase in phrases:
        if phrase in classes:
            class_ids.append(classes.index(phrase))
        else:
            distances = [edit_distance(phrase, c) for c in classes]
            class_ids.append(int(np.argmin(distances)))
    return np.array(class_ids)
```

## Checkpoints and data

Download the Grounded-SAM checkpoints from the [provided Google Drive folder](https://drive.google.com/drive/folders/1RvB3z8wi19saplpFYw07NwTdgVkBbH2G). Place the files under `data/grounded_sam/` with these names:

```text
data/
├── grounded_sam/
│   ├── GroundingDINO_SwinT_OGC.py
│   ├── groundingdino_swint_ogc.pth
│   ├── repvit_sam.pt
│   └── sam_vit_h_4b8939.pth
├── datasets/
│   └── R2R_VLNCE_v1-3_preprocessed/
└── scene_datasets/
    └── mp3d/
```

Populate `data/datasets/` with the R2R/RxR VLN-CE files and `data/scene_datasets/mp3d/` with the Matterport3D scene files required by Habitat. These assets are distributed by their respective dataset and benchmark owners and are not redistributed here.

## Configuration and API credentials

Evaluation configuration is stored in `vlnce_baselines/config/`. The default R2R entry point is `vlnce_baselines/config/r2r.yaml`; change the split, GPU IDs, process count, and checkpoint/output paths there or in the evaluation script as needed.

Set credentials in the shell before running an evaluation. Never commit a real key to Git:

```bash
export LA_API_KEY="your-api-key"
export LA_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LA_MODEL_NAME="qwen3-vl-235b-a22b-instruct"
```

`LA_BASE_URL` and `LA_MODEL_NAME` may be changed to any endpoint and model that support the OpenAI-compatible chat-completions interface.

## Evaluation

From the repository root, edit `CUDA_VISIBLE_DEVICES` and `--nprocesses` in the script you want to run, then execute:

```bash
bash eval_scripts/r2r.sh
bash eval_scripts/rxr.sh
```

The provided scripts are configured for the project's benchmark episodes. Runtime depends on the GPU, number of processes, model endpoint, and selected split; on an RTX 4090 with six processes, the OpenNav-100 evaluation takes approximately one hour.

