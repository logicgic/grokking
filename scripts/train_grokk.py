import random
import signal
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from datasets import AbstractDataset
from utils import combine_logs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item
from checkpointing import atomic_save, load_checkpoint, save_checkpoint

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def train(config):
    print('using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']
    # Worker prefetch queues cannot currently be checkpointed exactly.
    if train_cfg['num_workers'] != 0:
        raise ValueError('Checkpointable training requires train.num_workers=0')
    save_every = train_cfg.get('save_every', 10000)
    if not isinstance(save_every, int) or save_every < 0:
        raise ValueError('train.save_every must be a nonnegative integer')
    checkpoint_dir = Path(train_cfg.get('checkpoint_dir', 'checkpoints')).resolve()
    resume_from = train_cfg.get('resume_from')
    if resume_from and config['model'].get('checkpoint_path'):
        raise ValueError('Use train.resume_from or model.checkpoint_path, not both')

    seed = train_cfg.get('seed')
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], 
                              betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1))
    # Create the training iterator before restoring global RNG state: DataLoader
    # iterator creation consumes a torch RNG draw, but no samples with workers=0.
    train_iterator = iter(train_dataloader)
    step = 0
    if resume_from:
        # Relative resume paths refer to the launch directory, not Hydra's output.
        resume_path = Path(hydra.utils.to_absolute_path(resume_from))
        step = load_checkpoint(resume_path, model, optim, lr_schedule, config, dataset)

    stop_requested = []

    def request_stop(signum, frame):
        stop_requested.append(signum)

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    budget = train_cfg['max_steps']
    progress = tqdm(total=int(budget) if budget is not None else None, initial=step)
    try:
        while (budget is None or step < budget) and not stop_requested:
            x, y = next(train_iterator)
            loss, logs = model.get_loss(x.to(device), y.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()
            step += 1
            progress.update(1)
            if step % train_cfg['eval_every'] == 0:
                model.eval()
                with torch.no_grad():
                    all_val_logs = []
                    val_iterator = iter(val_dataloader)
                    for _ in range(train_cfg['eval_batches']):
                        val_x, val_y = next(val_iterator)
                        _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                        all_val_logs.append(val_logs)
                out_log = {'val': combine_logs(all_val_logs), 'train': combine_logs([logs]),
                           'step': step, 'lr': float(lr_schedule.get_last_lr()[0])}
                print(out_log, flush=True)
                if wandb_cfg['use_wandb']:
                    wandb.log(out_log)
                model.train()
            if save_every and step % save_every == 0:
                save_checkpoint(checkpoint_dir, model, optim, lr_schedule, step, config, dataset)
        save_checkpoint(checkpoint_dir, model, optim, lr_schedule, step, config, dataset)
        weight_name = 'model-interrupted.pt' if stop_requested else 'model-final.pt'
        atomic_save(model.state_dict(), checkpoint_dir / weight_name)
        status = 'Interrupted safely' if stop_requested else 'Training completed'
        print(f'{status} at step {step}; weights: {checkpoint_dir / weight_name}', flush=True)
    finally:
        progress.close()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if wandb_cfg['use_wandb']:
            wandb.finish()


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()

