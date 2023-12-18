import argparse
from escnn import gspaces
import pytorch_lightning as pl
from datasets.dataset_qm9 import QM9DataModule
from models.qm9.resnet import SteerableCNN_QM9
from models.qm9.lightning import QM9Regressor


parser = argparse.ArgumentParser()
# training
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=100)
# model
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_features', type=int, default=8)
parser.add_argument('--num_inv_features', type=int, default=32)
parser.add_argument('--L', type=int, default=1)
# MLP params
parser.add_argument('--mlp_n_channels', type=int, default=8)
parser.add_argument('--mlp_n_layers', type=int, default=3)
args = parser.parse_args()


dataset = QM9DataModule('./datasets/qm9/', batch_size=args.batch_size, name='QM9', target='homo').setup()
gspace = gspaces.flipRot3dOnR3() # O(3), see https://quva-lab.github.io/escnn/api/escnn.gspaces.html

model = SteerableCNN_QM9(
    gspace=gspace,
    num_layers=args.num_layers,
    num_features=args.num_features,
    num_inv_features=args.num_inv_features,
    L=args.L,
    mlp_params={'n_channels': args.mlp_n_channels, 'n_layers': args.mlp_n_layers},
    edge_distr=[None,None]
)

module = QM9Regressor(model, args.lr, dataset.target_mean, dataset.target_std, dataset.pos_mean, dataset.pos_std)

trainer = pl.Trainer(accelerator='auto', max_epochs=args.max_epochs)
trainer.fit(module, dataset)