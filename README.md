# RHS Wisley Bug Watch

This machine learning pipeline was designed as a step towards classifying insects like thrips and pirate bugs. It's purpose is to confirm or deny the presence of an insect in a given image. 

In summary, the pipeline saves filtered citizen-sourced data via Zooniverse that is then used as input data for the model's training and testing. The ratio of images with bugs to images with no bugs was very low. The data was both filtered and augmented to ensure balance.

The project uses a ResNet18 Convolutional Neural Network (CNN) and logs all experiments, model weights, and graphs to Weights & Biases (W&B).

## Setup

1. Install Dependencies

Ensure you have Python 3.9+ and `uv` installed on your machine.

To set up your environment and install all required dependencies, run `uv sync`.

2. Weights & Biases Setup

This project relies on W&B to track experiments.
Ensure you have an account at wandb.
Log in with ``wandb login`` and paste your API key when prompted.

3. Data Preparation
Place all the images from Zooniverse into the ``images`` folder.
Ensure the filtered CSV (i.e., Filtered_Bugs_Min4_Frac0.55_Gold-x.csv) is in the root directory. The script expects columns containing Filenames. You must manually edit the csv to have the correct labels (ground truth) indicating Insect or Other/Unknown.


## Project Structure

### Scripts to Run
`main.py`: This is the only script you need to run. It contains the EXPERIMENT_CONFIG dictionary where you can tweak the parameters. You might need to `cd` into the `insect_detection` folder first.

*Alternatively, with the HEC:*

`submitjobs.sh`: This becomes the only script you need to run presuming you have cloned your repo on the HEC. You will need to add the `git` module and install `uv`.

### High End Compute Commands and Scripts
- Login with `ssh <username>@wayland-2022.hec.lancs.ac.uk`.
- Create a shell script with `nano submitjobs.sh`.
- Run with `sbatch submitjobs.sh`.
- View status with `sacct`.

Copy and paste this code into the shell script:

```bash
#!/bin/bash
#SBATCH --job-name=pests
#SBATCH -p gpu-short
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module load cuda/12.9
nvidia-smi

source /etc/profile
source $HOME/.bashrc

uv sync
uv run main.py
```

It essentially specifies the specs for which the HEC will run the code and submits the job.

### Modules

- `data_prep.py`: Loads the dataset, first splits into training and testing, and then augments images with bugs in them (both test and train datasets).

- `initialise_model.py`: Initialises the ResNet-18 model and calculates class weights.

- `training_eval.py`: The training loop evaluates the model, outputs ROC curves, saves .pth weight files, and uploads prediction tables to W&B that are then used to plot more graphs like probability distributions.

- `images` folder: Put all images from Zooniverse here.

- `model_results` folder: Local folder where weights, graphs, and classification reports are saved.

## Running an Experiment
1. Edit the configurations 
```python
EXPERIMENT_CONFIG = {
    "learning_rate": 1e-4,   # Speed of learning
    "epochs": 25,            # Total training cycles
    "batch_size": 32,        # Images processed simultaneously
    "num_augmentations": 6,  # Number of copies generated per insect
    "threshold": 0.5,        # Confidence required to predict "Insect" (0.0 to 1.0)
    "frozen_layers": True,   # Set False to ...
    "notes": "Testing baseline learning rate."
}
```

2. Run the script
```bash
uv run main.py
```

3. Observe how the changes affect the results
Once the script starts, it will print a link in your terminal. Click it to open your W&B dashboard.

## Building W&B Custom Charts
You might need to create custom charts to observe certain kinds of metrics in specific ways.
1. Click Add Panel > Custom Chart.
2. Set the Query to Summary Table > `Evaluation_Data`.
3. Use the `Is Correct`, `Insect Probability`, and `Image Type` columns mapped by the Python script to generate your custom visualisations.