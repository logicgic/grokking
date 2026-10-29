## GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS
### unofficial re-implementation of [this paper](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) by Power et al.
### code written by Charlie Snell

pull and install:
```
git clone https://github.com/Sea-Snell/grokking.git
cd grokking/
pip install -r requirements.txt
```

To roughly re-create Figure 1 in the paper run:

```
export PYTHONPATH=$(pwd)/grokk_replica/
cd scripts/
python train_grokk.py
```

![](grokk.png)
###### Running the above command should give curves like this.

Try different operations or learning / architectural hparams by modifying configurations in the `config/` directory. I use [Hydra](https://hydra.cc/docs/intro) to handle the configs (see their documentation to learn how to change configs in the commandline etc...).

Training uses [Weights And Biases](https://wandb.ai/home) by default to generate plots in realtime. If you would not like to use wandb, just set `wandb.use_wandb=False` in `config/train_grokk.yaml` or as an argument when calling `train_grokk.py`


### Checkpoints and resuming

Training saves a full `checkpoints/latest.pt` every 10,000 updates and at the
end of training. Paths are relative to the Hydra run directory unless absolute.
`train.save_every=0` disables periodic saving but still saves at completion.
The latest checkpoint is atomically replaced, so storage does not grow with steps.
A separate `checkpoints/model-final.pt` contains the final model state dict.
SIGINT/Ctrl+C and SIGTERM request a stop after the current update/evaluation;
then the script saves `latest.pt` and `model-interrupted.pt`. SIGKILL, power loss,
and unhandled errors can only be recovered from the last completed checkpoint.

Resume using an absolute checkpoint path (a relative resume path is resolved
against the directory where the command was launched):

```bash
export PYTHONPATH="$PWD/grokk_replica${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/train_grokk.py \
  dataset=mod_polynomial_dataset dataset.p=97 dataset.frac_train=0.5 \
  train.resume_from=/absolute/path/to/checkpoints/latest.pt \
  train.max_steps=5000000 wandb.use_wandb=false
```

`max_steps` is the total target step, not the number of additional updates.
Keep the same dataset, architecture, optimizer settings, batch size, evaluation
schedule, and `num_workers=0`; incompatible settings are rejected. Full checkpoints
restore model, optimizer, scheduler, step, dataset split/token mapping, and Python,
PyTorch and CUDA random states. Exact trajectories across hardware or PyTorch
versions are not guaranteed. Only load checkpoints from trusted sources.

For inference-only weight loading, use `model.checkpoint_path=/absolute/path/to/model-final.pt`;
this starts a new optimizer and is not a training resume. Full `latest.pt` files
must instead be supplied through `train.resume_from`.
