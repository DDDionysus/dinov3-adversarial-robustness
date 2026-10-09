# Evaluating the Robustness of Self-Supervised Vision Transformers to Adversarial Perturbations: A Study of Gram Anchoring in DINOv3


## Overview 

This project evaluates the adversarial robustness of self-supervised Vision Transformers by comparing DINOv2 and 
DINOv3. Robustness is assessed at both representation-level using CLS-token and patch-token cosine similarity, and the 
downstream-task level through semantic segmentation and monocular depth estimation. Fast Gradient Sign Method (FGSM) 
and Projected Gradient Descent (PGD) attacks are used to study model behaviour under adversarial perturbations. 


## Project Structure 
```text
code/
└── checkpoints/            # local only; not included in the repository
│   └── backbones/          # pretrained DINO model checkpoints
└── dataset                 # local only; not included in the repository
│   ├── ImageNet/           # ImageNet validation images
│   ├── NYU_Depth_v2/       # NYU Depth V2 RGD/Depth data
│   └── VOC2012/            # Pascal VOC 2012 images and masks
└── output/                 # experiment outputs 
└── src/
    ├── attacks/            # FGSM and PGD implementations
    ├── datasets/           # dataset loading and preprocessing 
    ├── dev/                # early development, sanity-check, exploratory scripts
    ├── downstream
    │   ├── depth/          # monocular depth estimation pipeline
    │   └── segmentation/   # semantic segmentation pipeline 
    ├── evaluation/         # robustness evaluation utilities
    ├── experiments/        # experiment scripts
    ├── features/           # DINO feature extraction
    ├── models/             # model loading 
    ├── utils/              # helper functions
    └── visualisation/      # result visualisation, plotting
```

Note that `dataset/` and `checkpoints/` directories are not included in the repository because the datasets and 
pretrained weights are distributed separately. Create these directories locally and place the downloaded files in the 
corresponding subdirectories. If different directory names are used, update the dataset/checkpoint paths in the 
experiment scripts accordingly. 


## Models 
This project uses pretrained DINOv2 and DINOv3 variants. 

<table style="margin: auto">
    <thead>
        <tr>
            <th>Generation</th>
            <th>Model</th>
            <th>Parameters</th>
            <th>Download</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>DINOv2</td>
            <td>ViT-L/14 distilled </td>
            <td align="right">300M</td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">[link]</a></td>
        </tr>
        <tr>
            <td>DINOv3</td>
            <td>ViT-L/16 distilled</td>
            <td align="right">300M</td>
            <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
        </tr>
    </tbody>
</table>


## Datasets 

<table style="margin: auto">
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Download</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ImageNet</td>
            <td align="center"><a href="https://image-net.org/download.php">[link]</a></td>
        </tr>
        <tr>
            <td>Pascal VOC 2012</td>
            <td align="center"><a href="https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar">[link]</a></td>
        </tr>
        <tr>
            <td>NYU Depth V2</td>
            <td align="center"><a href="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat">[link]</a></td>
        </tr>
    </tbody>
</table>

Note that for the access to ImageNet, registration and approval are required. 

After access is granted: 
- Go to the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012),
- Download Validation images (all tasks) 6.3GB


## Environment / Dependencies 

This project was developed using Python with PyTorch. 

Install the required Python packages with: 

```bash 
pip install -r requirements.txt
```

The main dependencies are: 
- PyTorch 
- torchvision 
- Pillow 
- h5py 
- tqdm 
- pandas 
- matplotlib 

For GPU acceleration, install a PyTorch build compatible with the CUDA version available on your system. See the 
official PyTorch installation instructions for the appropriate command. 


## Running the Experiments 
### Representation-level Evaluation 

For running representation-level evaluation locally: 

- Go to `code/src/experiments/other_datasets.py`, 
- Set variable `dataset_names` to each of the datasets: `imagenet`, `voc`, or `nyu` separately, or include them all in 
the list to run them in one go, 
- Run `code/src/experiments/other_datasets.py`.

For experiments in an HPC environment: 

- Go to `code/src/experiments/hpc_other_datasets.py`, 
- Set variable `dataset_names` to each of the datasets: `imagenet`, `voc`, or `nyu` separately, or include them all in 
the list to run them in one go, 
- Run `code/src/experiments/hpc_other_datasets.py`.

#### Output 

The representation-level evaluation produces CSV files containing the robustness metrics for each perturbation budget.

The main metrics are:

- **CLS similarity**: cosine similarity between clean and adversarial CLS-token representations.
- **Patch mean**: mean cosine similarity between corresponding clean and adversarial patch-token representations.
- **Patch standard deviation**: variation in cosine similarity across patch locations within an image.

Higher cosine similarity indicates that the adversarial perturbation causes a smaller change in the extracted 
representation and therefore indicates stronger representation stability.

Results are generated separately for FGSM and PGD across the configured perturbation budgets.

### Semantic Segmentation 

For semantic segmentation evaluation: 

- Go to `code/src/downstream/segmentation/train.py`, 
- Set variable `model_name` to the required model variant, 
- Run `code/src/downstream/segmentation/train.py` to train the corresponding prediction head, 
- Go to `code/src/downstream/segmentation/evaluate_attack.py`, 
- Set variable `model_name` to the corresponding model variant, and `attack` to the required adversarial attack, 
- Run `code/src/downstream/segmentation/evaluate_attack.py` to evaluate robustness under the selected attack. 

The evaluation reports semantic segmentation performance under clean and adversarial inputs using mean Intersection 
over Union (mIoU).

### Monocular Depth Estimation 

For monocular depth estimation evaluation: 

- Go to `code/src/downstream/depth/train.py`, 
- Set variable `model_name` to the required model variant, 
- Run `code/src/downstream/depth/train.py` to train the corresponding prediction head, 
- Go to `code/src/downstream/depth/evaluate_attack.py`, 
- Set variable `model_name` to the corresponding model variant, and `attack` to the required adversarial attack, 
- Run `code/src/downstream/depth/evaluate_attack.py` to evaluate robustness under the selected attack. 

The evaluation reports monocular depth estimation performance under clean and adversarial inputs using Root Mean Square 
Error (RMSE) and Absolute Relative Error (AbsRel).

### HPC Execution

HPC job scripts are provided in `code/src/hpc_scripts/`:

- `run_fgsm.sh` runs the ImageNet representation-level FGSM experiment.
- `run_pgd.sh` runs the ImageNet representation-level PGD experiment.
- `run_nyu.sh` runs the configured representation-level experiment on NYU Depth V2.

The scripts use SLURM job arrays to divide the dataset across multiple GPU jobs. Before submitting a job, update 
the project, dataset, checkpoint, and output paths in the corresponding Python and shell scripts to match the HPC 
environment.

The required Python and PyTorch modules are loaded inside each job script. The exact module versions may need to be 
changed depending on the software available on the target HPC system.


## DINOv3-7B Representation Robustness Experiment 

This section describes the new DINOv3-7B experiment pipeline. Adversarial perturbation generation and representation 
evaluation are separated so that the exact generated inputs can be reused without regenerating the attacks. 

### Required Files 

Before running the experiment, prepare: 

- the ImageNet validation subset: https://image-net.org/download.php; 
- the DINOv3 ViT-7B/16 pretrained checkpoint; 
- a writable output directory, preferably in HPC scratch storage. 

The experiment uses the following files: 

```text
code/src/extension/representation/dinov3_7b_experiment.py
code/src/extension/representation/run_7b.sh
```

- `dinov3_7b_experiment.py` implements dataset partitioning, perturbation generation, and evaluation. 
- `run_7b.sh` configures the HPC environment and runs the selected experiment stage. 

The same Python runner is used for all experiment stages. 

### HPC Configuration 

Open the provided DINOv3-7B SLURM script `run_7b.sh` and configure the following paths: 

```bash
export IMAGENET_VAL_PATH="path to imagenet dataset"
export DINOV3_7B_CHECKPOINT="path to dinov3 7b checkpoint"
export EXPERIMENT_OUTPUT_DIR="path to output"
```

The script assumes that the repository is located at `/scratch_tide/$USER/dinov3-adversarial-robustness`. Update the `cd` command if the
repository is stored elsewhere.

The submission script assumes that a working Python and PyTorch environment is available on the HPC system. The module 
versions or environment setup may need to be adjusted according to the available HPC configuration. 

The main Python dependencies are: 

- PyTorch 
- torchvision 
- Pillow
- tqdm 

The model loader uses `torch.hub.load()`. Therefore, the DINOv3 repository code must be accessible through network 
access, the Torch Hub cache, or a local repository checkout. 

### Experimental Stages 

The experiment has four stages: 

```text
generate_rfgsm
generate_pgd
evaluate_rfgsm
evaluate_pgd
```

The generation stages use SLURM job arrays to divide the dataset between workers. The evaluation stages must run as 
ordinary single jobs after the corresponding generation stage has completed. 

### Running R-FGSM Generation 

```bash
sbatch --array=0-49 code/src/extension/representation/run_7b.sh generate_rfgsm
```

### Running PGD Generation 

```bash
sbatch --array=0-49 code/src/extension/representation/run_7b.sh generate_pgd 
```

Each array task processes a separate subset of the dataset. The generated files contain:

- the clean input tensor;
- the exact adversarial input tensor;
- image ID;
- model name;
- attack type;
- attack objective;
- epsilon;
- random seed;
- alpha and number of steps, where applicable.

### Evaluating the Saved R-FGSM Inputs 

```bash
sbatch code/src/extension/representation/run_7b.sh evaluate_rfgsm
```

### Evaluating the Saved PGD Inputs 

```bash
sbatch code/src/extension/representation/run_7b.sh evaluate_pgd
```

Do not use `--array` for either evaluation stage.

The evaluation stages load the saved clean and adversarial tensors and calculate:

- CLS-token cosine similarity;
- patch-token cosine similarity mean;
- patch-token cosine similarity standard deviation;
- Gram-matrix mean squared error;
- normalized off-diagonal Gram difference.

The attacks are not regenerated during evaluation.

### Output Structure

The expected output structure is:

```text
EXPERIMENT_OUTPUT_DIR/
├── perturbations/
│   ├── rfgsm/
│   │   └── dinov3_vit7b16/
│   └── pgd/
│       └── dinov3_vit7b16/
└── results/
    ├── rfgsm_dinov3.csv
    └── pgd_dinov3.csv
```

Before launching the full experiment, it is recommended to submit a single array task first:

```bash
sbatch --array=0-0 code/src/extension/representation/run_7b.sh generate_rfgsm
```

This can be used to verify the Python environment, model loading, GPU memory usage, execution time, dataset path, 
checkpoint path, and output location.
