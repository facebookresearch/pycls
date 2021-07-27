# Sweeps and Design Spaces

The *sweep* code in pycls provides support for studying *design spaces* and more generally *population statistics* of models. This idea is that instead of planning a single pycls job (e.g., testing a specific model configuration), one can study the behavior of an entire population of models. This allows for quite powerful and succinct experimental design, and elevates the study of individual model behavior to the study of the behavior of model populations.

This doc is organized as follows:
- [Introduction and background](#introduction-and-background)
- [Sweep prerequisites](#sweep-prerequisites)
- [Sweep overview](#sweep-overview)
- [Sweep examples](#sweep-examples)

## Introduction and background

The concept of network *design spaces* was introduced in  [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) and [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678). A design space is a large, possibly infinite, population of model architectures. The core insight is that we can sample models from a design space, giving rise to a model distribution, and turn to tools from classical statistics to analyze the design space. Instead of studying the behavior of an individual model, we study the *population statistics* of a collection of models. For example, when studying network design, we can aim to find a single best model under a specific setting (as in model search), or we can aim to study a diverse population of models to understand more general design principles that make for effective models. Typically, the latter allows us to learn and *generalize* to new settings, and makes for more robust findings that are likely to hold under more diverse settings. We recommend reading the above mentioned papers for further motivation and details.

We operationalize the study of design spaces and population statistics by introducing a very flexible notion of *sweeps*. Simply put, a sweep is a population level experiment consisting of a number of individual pycls jobs. As mentioned, studying the *population statistics* of models can be more informative than designing and testing an individual model. A sweep can be as simple as a grid or random search over a hyperparameter (e.g., model depth, learning rate, etc.). However, a sweep can be far more general than a hyperparameter search, and can be used to study the behavior of diverse populations of models simultaneously varying along many dimensions.

Just like a single training job in pycls is defined by a config, a sweep is likewise defined by a *sweep config*. This meta-level sweep config is an extremely powerful concept and elevates experimental design to the sweep level. So, rather than creating a large number of individual pycls configs, we can create a single sweep config to generate and study a population of models. A sweep config defines everything about a sweep, including: (1) the sweep setup options (defines how to sample pycls configs), (2) sweep launch options (how to launch the sweep to a cluster), and (3) sweep analysis options (to  generate an analysis of the model population statistics). Rather than going into more detail here, we suggest studying the examples below and looking at the code documentation.

## Sweep prerequisites

Before beginning with the sweep code, please make sure to complete the following steps:

- Please ensure that you can run individual pycls jobs by following the steps in [`GETTING_STARTED.md`](GETTING_STARTED.md). You should be able to successfully run an individual pycls job (e.g., training a model) prior to running a sweep.

- Instructions and tools mentioned here were designed to work on *SLURM managed clusters*. While most of the code could easily be adopted to other clusters (in particular only  [`sweep_launch.py`](../tools/sweep_launch.py) and [`sweep_launch_job.py`](../tools/sweep_launch_job.py) would likely need to be altered for a non-SLURM managed cluster), the pycls code only supports SLURM managed clusters out of the box.

- For simplicity and to successfully run the examples below, we recommend changing all the files in ./tools to be executable by the user (for example: `chmod 744 ./tools/*.py`). Of course, as an alternative, one can instead execute any of the python scripts invoking python explicitly.

## Sweep overview

### The sweep config

A sweep config consists of three main parts (usage described in more detail shortly):
- `SETUP` options: used to specify a base pycls config along with samplers
- `COLLECT` options: used to specify options for collecting the sweep results
- `ANALYSIS` options: used to specify option for analyzing the sweep results

In addition to these parts, there are a few top-level options that should be set, including:
- `ROOT_DIR`: root directory where all sweep output subdirectories will be placed
- `NAME`: the sweep name must be unique and defines the output subdirectory(s)
- `RUN_MODE`: mode to launch the sweep with (`train`, `test` or `time`)

For full documentation see: [`sweep/config.py`](../pycls/sweep/config.py). It is easier to get started by looking at the example sweeps at the end of this doc prior to looking at the full documentation.

### Setting up a sweep

[`sweep_setup.py`](../tools/sweep_setup.py): Setting up a sweep generates the individual pycls job configs necessary to launch the sweep. This make take some time (many minutes) if sampling many configs or if it is difficult to find configs that generate the sampling constraints. Once the sweep config is defined, the sweep can be set up via:
```
SWEEP_CFG=path/to/config.yaml
./tools/sweep_setup.py --sweep-cfg $SWEEP_CFG
```
In this and following examples we assume the sweep config is stored at `path/to/config.yaml`.

The following files are created in the output directory:
```
ROOT_DIR/NAME/cfgs/??????.yaml  # numbered configs for individual pycls jobs
ROOT_DIR/NAME/cfgs_summary.yaml  # summary of the generated cfgs
ROOT_DIR/NAME/sweep_cfg.yaml  # copy of the original sweep configuration
```
Here ROOT_DIR and NAME are the fields specified in the sweep config. Note that before launching the sweep, you should spot check some of the generated configs and the cfgs_summary.yaml to see if the generated pycls configs look reasonable. You can run the sweep_setup command repeatedly so long as you have not launched the sweep.

### Launching a sweep

[`sweep_launch.py`](../tools/sweep_launch.py): Launching a sweep sends the individual pycls jobs to a SLURM managed cluster using the options in LAUNCH of the sweep config. The launch is fairly quick, although obviously the individual pycls jobs may run for long periods of time. The sweep can be launched via:
```
./tools/sweep_launch.py --sweep-cfg $SWEEP_CFG
```
The following files are created in the output directory:
```
ROOT_DIR/NAME/logs/??????/*  # results of each individual pycls job
ROOT_DIR/NAME/logs/sbatch/*  # SLURM log files fo reach pycls job
ROOT_DIR/NAME/pycls/*  # copy of pycls code for basic job isolation
```
A sweep should only be launched once. If the sweep is fully stopped, you can resume it by calling the sweep_launch command again. While the sweep is running, you can monitor individual jobs by looking in the individual pycls log output directories, or by collecting the sweep (described next).

Note that standard SLURM commands can be used to monitor the sweep and individual jobs, cancel a sweep, requeue it, etc. See the [SLURM documentation](https://slurm.schedmd.com/documentation.html) for more information. Common useful SLURM commands include:
```
squeue --me
scontrol requeue JOBID
scancel JOBID
sinfo -o '%f %A %N %m %G' | column -t
```

### Collecting a sweep

[`sweep_collect.py`](../tools/sweep_collect.py): Collecting a sweep gathers core information from each individual pycls job and places it into a single large json file. Note that collecting a sweep is also a great way to see the *status* of a sweep, and the sweep collection can be run an unlimited number of times. The command for this is:
```
./tools/sweep_collect.py --sweep-cfg $SWEEP_CFG
```
The following files are created in the output directory:
```
ROOT_DIR/NAME/sweep.json  # output file with all of the sweep information
```

### Analyzing a sweep

[`sweep_analyze.py`](../tools/sweep_analyze.py): Analyzing a sweep is the final step in the life cycle of a sweep. Note that the analysis can be run as soon as partial results are collected (via sweep_collect.py) and does not require the sweep to be finished or even for any individual pycls jobs to be finished. The analysis depends on the options in ANALYSIS part of the sweep config; note that it is typical to reanalyze the data multiple times while altering the ANALYSIS options. The command for analysis is:
```
./tools/sweep_analyze.py --sweep-cfg $SWEEP_CFG
```
The following files are created in the output directory:
```
ROOT_DIR/NAME/analysis.html  # html file containing the sweep analysis
```
After generating the analysis, the analysis.html file can be viewed in any browser. As it’s a fully self-contained html file (with embedded vector images), it can also be easily shared.

## Sweep examples

We provide three example sweeps (along with their output):
- Sweep config: [`cifar_optim`](../configs/sweeps/cifar/cifar_optim.yaml) | Analysis: [cifar_optim_analysis](https://dl.fbaipublicfiles.com/pycls/sweeps/cifar/cifar_optim_analysis.html)
- Sweep config: [`cifar_regnet`](../configs/sweeps/cifar/cifar_regnet.yaml) | Analysis: [cifar_regnet_analysis](https://dl.fbaipublicfiles.com/pycls/sweeps/cifar/cifar_regnet_analysis.html)
- Sweep config: [`cifar_best`](../configs/sweeps/cifar/cifar_best.yaml) | Analysis: [cifar_best_analysis](https://dl.fbaipublicfiles.com/pycls/sweeps/cifar/cifar_best_analysis.html)

We suggest looking at each example config carefully to understand the setup process. We will go through the [`cifar_optim`](../configs/sweeps/cifar/cifar_optim.yaml) config next in more detail. The other two example configs can serve as additional reference point and demonstrate various simple use cases.

The [`cifar_optim`](../configs/sweeps/cifar/cifar_optim.yaml) config starts with a `DESC` and `NAME` field which are self-explanatory. Next comes the `SETUP` section. In `SETUP`, the  `NUM_CONFIGS: 64` field indicates that we will sample 64 individual pycls configs. Next are the `SAMPLERS`:
```
SAMPLERS:
  OPTIM.BASE_LR:
    TYPE: float_sampler
    RAND_TYPE: log_uniform
    RANGE: [0.25, 5.0]
    QUANTIZE: 1.0e-10
  OPTIM.WEIGHT_DECAY:
    TYPE: float_sampler
    RAND_TYPE: log_uniform
    RANGE: [5.0e-5, 1.0e-3]
    QUANTIZE: 1.0e-10
```
There are two `SAMPLERS`, one for the `OPTIM.BASE_LR` and the other for `OPTIM.WEIGHT_DECAY`. Both of these sample floats using a log-uniform distribution with the ranges and quantization as specified. This means that for every sampled config, these two corresponding fields will be sampled from these distributions. The `SAMPLERS` are typically a critical aspect of the sweep config and control how individual pycls configs are generated. There is a lot of flexibility in the type of sampler to use and the distribution from which to sample, see the `SAMPLERS` section of the [`sweep/config.py`](../pycls/sweep/config.py) for more details. Note that one can sample any parameter that is part of the base pycls config. In addition , one can put `CONSTRAINTS` on the sampled configs, this functionality is not used in this example but is used in the [`cifar_regnet`](../configs/sweeps/cifar/cifar_regnet.yaml) example.

Next, after the samplers comes the `BASE_CFG`. The `BASE_CFG` is simply a standard pycls config. Every sampled config will be the `BASE_CFG` with values specified by the `SAMPLERS` (like `OPTIM.BASE_LR`) overwritten. In this example, the `BASE_CFG` is simply ResNet-56 with some strong data augmentation. Note, however, that the epoch length (`OPTIM.MAX_EPOCH`) is set to a fairly short 50 epochs to allow for each individual job to be fast. Typically, when generating a sweep, we keep the epochs per model low and focus not on absolute performance but on observed trends.

Next comes the `LAUNCH` options. Note that these may need to be customized to different cluster setups (e.g., the `PARTITION` field will likely need to be changed for different clusters). Finally the `ANALYZE` options control the generated analysis html file. For example, in this case we plot `METRICS: [lr, wd, lr_wd]`, meaning we plot the learning rate, weight decay, and the product of the two. These are simply shortcuts to the corresponding fields in the config; these shortcuts are defined in [`sweep/analysis.py`](../pycls/sweep/analysis.py).

Take a look at the generated [cifar_optim_analysis](https://dl.fbaipublicfiles.com/pycls/sweeps/cifar/cifar_optim_analysis.html). First, there are Error Distribution Functions (EDF) for the trained models, see [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214). Next there are plots showing error versus learning rate, weight decay, and the product of the two. An interesting observation is that the product of learning rate and weight decay are most predictive of error. Next are plots of error versus various complexity metrics (note, however, that the model is fixed in all the configs so the complexity metrics don’t vary). Finally, training and testing curves are shown for the best three models. As discussed, the analysis is fully customizable (see the options in `ANALYSIS`).

Finally, for reference, to run the sweep in its entirety, the steps are:
```
# from within the pycls root directory:
SWEEP_CFG=configs/sweeps/cifar/cifar_optim.yaml
./tools/sweep_setup.py --sweep-cfg $SWEEP_CFG
./tools/sweep_launch.py --sweep-cfg $SWEEP_CFG
./tools/sweep_collect.py --sweep-cfg $SWEEP_CFG
./tools/sweep_analyze.py --sweep-cfg $SWEEP_CFG
```

The best way to learn more about sweeps is to set up your own sweeps. The sweep config system is quite powerful and allows for many interesting experiments.
