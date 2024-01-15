# import sys
# sys.path.append('/home/nouri-sepehr-ad/.local/lib/python3.8/site-packages')

# import ptvsd
# ptvsd.enable_attach(address=('172.16.10.44', 5678))
# ptvsd.wait_for_attach()

from argparse import ArgumentParser
import torch
import optuna

from models.trainer import *
import os
import random

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def train(args):
    seed()
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def network_summary(args):
    seed()
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.summarize_network(args=args)


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()

def objective(trial):
    # patch_size = trial.suggest_categorical('patch_size', [2,4,8,16])
    global index

    patch_size_list = [2,4,8,16]
    patch_size = patch_size_list[index]
    args.patch_size = patch_size
    print(f"Optuna Hyperparameter [Patch Size: {patch_size}]")
    index += 1

    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

    validation_acc = model.validation_acc
    return validation_acc

def seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed()

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: 6,7 0,1 2,3,4,5 -1 for CPU')
    parser.add_argument('--project_name', default='./base.model/SYSU/TE34_1', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='./output_visuals', type=str)

    # data
    parser.add_argument('--num_workers', default=32, type=int)       # origianl: 8
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='SYSU', type=str)
    parser.add_argument('--batch_size', default=32, type=int)        # original: 16
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)        # original: 512
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=bool)
    parser.add_argument('--multi_scale_infer', default=False, type=bool)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])
    parser.add_argument('--net_G', default='ScratchFormer', type=str, help='ScratchFormer')
    parser.add_argument('--loss', default='ce', type=str)

    # decoder
    parser.add_argument('--patch_size', default=4, type=int, help= 'recommended: decoder(A,B)=2, decoderC=4')
    parser.add_argument('--decoder_type', default='decoderA', type=str, help='base,decoder(A,B,C)')

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.00035, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str, help='linear | step')
    parser.add_argument('--lr_decay_iters', default=[100], type=int)

    # optuna
    parser.add_argument('--num_trials', default=0, type=int)
    
    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # optimization
    # study = optuna.create_study(direction='maximize')
    # index = 0
    # study.optimize(objective, n_trials = 4)

    # best_params = study.best_params
    # best_patch_size = best_params['patch_size']

    # print(f"Best Patch Size: {best_patch_size}")
    torch.cuda.empty_cache()
    #network_summary(args)
    train(args)

    #test(args)
