# set up environment
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)),
        tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=70)

    # Add these checks
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        # print(f"Sample {i} type: {type(sample)}")
        if isinstance(sample, tuple):
            # print(f"  Tuple length: {len(sample)}")
            pass
            for j, item in enumerate(sample):
                # print(f"    Item {j} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
                pass
        elif isinstance(sample, dict):
            pass
            # print(f"  Dict keys: {sample.keys()}")
            for key, value in sample.items():
                pass
                # print(f"    {key} type: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        else:
            pass
            # print(f"  Unexpected type: {type(sample)}")
        
        # Assuming the first element is the image and the second is the label
        image = sample[0] if isinstance(sample, tuple) else sample['image']
        label = sample[1] if isinstance(sample, tuple) else sample['label']
        
        # print(f"  Image stats: min={image.min().item():.3f}, max={image.max().item():.3f}, mean={image.mean().item():.3f}")
        # print(f"  Label stats: min={label.min().item():.3f}, max={label.max().item():.3f}, mean={label.mean().item():.3f}")
        
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"  WARNING: NaN or Inf values found in image {i}")

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader

class BaseTrainer:
    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint(join(self.args.work_dir, self.args.task_name, './ckpt/sam_med3d.pth'))
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        
    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()}, # , 'lr': self.args.lr * 0.1},
            {'params': sam_model.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if self.args.multi_gpu:
                self.model.module.load_state_dict(last_ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))
    
    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        # print(f"low_res_masks from decoder: min={low_res_masks.min().item()}, max={low_res_masks.max().item()}, mean={low_res_masks.mean().item()}")

        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        # print(f"prev_masks after interpolation: min={prev_masks.min().item()}, max={prev_masks.max().item()}, mean={prev_masks.mean().item()}")

        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            # print(f"Click {num_click + 1}/{num_clicks}")

            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
            # print(f"Click {num_click + 1} loss: {loss.item()}")




        return prev_masks, return_loss
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            intersection = (mask_pred & mask_gt).sum().float()
            union = mask_pred.sum() + mask_gt.sum()
            if union == 0:
                return 0.0
            return (2.0 * intersection / union).item()

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        # print(f"pred_masks unique values: {torch.unique(pred_masks)}")
        # print(f"true_masks unique values: {torch.unique(true_masks)}")
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice = compute_dice(pred_masks[i], true_masks[i])
            print(f"Sample {i} dice: {dice}")
            dice_list.append(dice)
        avg_dice = sum(dice_list) / len(dice_list)
        print(f"Average dice: {avg_dice}")
        return avg_dice
    def get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    def calculate_dice_score(self, pred_masks, gt_masks):
        pred_masks = (pred_masks > 0.5).float()
        gt_masks = (gt_masks > 0).float()
        
        intersection = (pred_masks * gt_masks).sum()
        union = pred_masks.sum() + gt_masks.sum()
        
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return dice.item()
    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_dice_sum = 0
        epoch_sample_count = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        for step, data in enumerate(tbar):
            # Unpack data
            if isinstance(data, (tuple, list)):
                image3D, gt3D = data
            elif isinstance(data, dict):
                image3D, gt3D = data['image'], data['label']
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

            # Check if the image dimensions are correct (allowing for variable batch size)
            if len(image3D.shape) != 5 or image3D.shape[1:] != (1, 128, 128, 128):
                print(f"Skipping batch at step {step} due to unexpected shape: {image3D.shape}")
                torch.save({
                    'image3D': image3D,
                    'gt3D': gt3D,
                    'step': step,
                    'epoch': epoch
                }, f'skipped_batch_epoch{epoch}_step{step}.pt')
                continue

            batch_size = image3D.shape[0]

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():
                try:
                    image3D = self.norm_transform(image3D.squeeze(dim=1))  # (B, C, W, H, D)
                    image3D = image3D.unsqueeze(dim=1)
                    
                    if torch.isnan(image3D).any() or torch.isinf(image3D).any():
                        raise ValueError("NaN or Inf values found after normalization")
                    
                    image3D = image3D.to(device)
                    gt3D = gt3D.to(device).type(torch.long)
                    
                    with amp.autocast():
                        image_embedding = sam_model.image_encoder(image3D)
                        if torch.isnan(image_embedding).any() or torch.isinf(image_embedding).any():
                            raise ValueError("NaN or Inf values in image embedding")
                        
                        self.click_points = []
                        self.click_labels = []

                        prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=num_clicks)                

                    epoch_loss += loss.item()
                    cur_loss = loss.item()
                    loss /= self.args.accumulation_steps
                    
                    self.scaler.scale(loss).backward()

                    # Calculate dice score for each sample in the batch
                    for i in range(batch_size):
                        dice_score = self.calculate_dice_score(prev_masks[i:i+1], gt3D[i:i+1])
                        epoch_dice_sum += dice_score
                        epoch_sample_count += 1

                        # Update best dice score
                        if dice_score > self.step_best_dice:
                            self.step_best_dice = dice_score
                            if dice_score > 0.9:
                                self.save_checkpoint(
                                    epoch,
                                    sam_model.state_dict(),
                                    describe=f'{epoch}_step_dice:{dice_score:.4f}_best'
                                )

                except ValueError as e:
                    print(f"Error in step {step}, epoch {epoch}")
                    print(f"Error message: {str(e)}")
                    
                    torch.save({
                        'image3D': image3D,
                        'gt3D': gt3D,
                        'step': step,
                        'epoch': epoch
                    }, f'error_batch_epoch{epoch}_step{step}.pt')

                    continue

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                
                if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss:.4f}, Best Dice: {self.step_best_dice:.4f}')
                    
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
            else:
                step_loss += cur_loss

        epoch_loss /= (step + 1)
        epoch_dice = epoch_dice_sum / epoch_sample_count if epoch_sample_count > 0 else 0
        print(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}, Average Dice: {epoch_dice:.4f}, Best Dice: {self.step_best_dice:.4f}")

        return epoch_loss, 0, epoch_dice, []
        def eval_epoch(self, epoch, num_clicks):
            return 0
        
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )
                
                # save train dice best checkpoint
                if epoch_dice > self.best_dice: 
                    self.best_dice = epoch_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                # self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
