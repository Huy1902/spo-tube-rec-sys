import torch
import os
import numpy as np
import pandas as pd
from rl_agent.utils import set_random_seed, wrap_batch
from reader.MDPDataReader import MDPDataReader
from model.MDPUserRespond import MDPUserResponse
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

path_to_data = "dataset/mdp/"
path_to_output = "output/mdp/"


cuda = 0
if cuda >= 0 and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    torch.cuda.set_device(cuda)
    device = f"cuda:{cuda}"
else:
    device = "cpu"

print(device)

item_info = np.load(os.path.join(path_to_data, "item_info.npy"))
item_ids = np.load(os.path.join(path_to_data, "item_ids.npy"))
train = pd.read_csv(os.path.join(path_to_data, "all.csv"), sep="@")
test = pd.read_csv(os.path.join(path_to_data, "test.csv"), sep="@")

params = dict()
params['train'] = train
params['val'] = test
params['item_meta'] = item_info
params['item_ids'] = item_ids
params['n_worker'] = 0
params['max_seq_len'] = 50

params['loss_type'] = 'bce'
params['device'] = device
params['l2_coef'] = 0.001
params['lr'] = 0.0003
params['feature_dim'] = 16
params['hidden_dims'] = [256]
params['attn_n_head'] = 2
params['batch_size'] = 128
params['seed'] = 26
params['epoch'] = 1
params['dropout_rate'] = 0.2
params['model_path'] = os.path.join(path_to_output,
                          f"env/mdp_user_env_lr{params['lr']}_reg{params['l2_coef']}_eval.model")
set_random_seed(params['seed'])

# @title Train user response
reader = MDPDataReader(params)
model = MDPUserResponse(reader, params).to(device)

# reader = RL4RSDataReader(params)
# model = RL4RSUserResponse(reader, params).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
model.optimizer = optimizer

epo = 0
while epo < params['epoch']:
    print(f"epoch {epo} is training")
    epo += 1

    model.train()
    reader.set_phase("train")
    train_loader = DataLoader(reader, params['batch_size'], shuffle=True, pin_memory=True,
                              num_workers=params['n_worker'])

    t1 = time()
    pbar = tqdm(total=len(train_loader.dataset))
    step_loss = []
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        wrapped_batch = wrap_batch(batch_data, device)

        out_dict = model.do_forward_and_loss(wrapped_batch)
        loss = out_dict['loss']
        loss.backward()
        step_loss.append(loss.item())
        optimizer.step()
        pbar.update(params['batch_size'])
        # print(model.loss)
        # if (i + 1) % 10 == 0:
        # print(f"Iteration {i + 1}, loss {np.mean(step_loss[-100:])}")
    pbar.close()
    # print("Epoch {}; time {:.4f}".format(epo, time() - t1))
    print(f"epoch {epo} training loss: {np.mean(step_loss)}")

    # validation
    t2 = time()
    reader.set_phase("val")
    val_loader = DataLoader(reader, params['batch_size'], shuffle=False, pin_memory=False,
                            num_workers=params['n_worker'])
    valid_probs, valid_true = [], []
    pbar = tqdm(total=len(val_loader.dataset))
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            wrapped_batch = wrap_batch(batch_data, device)
            out_dict = model.forward(wrapped_batch)
            pbar.update(params['batch_size'])
            valid_probs.append(out_dict['probs'].cpu().numpy())
            valid_true.append(batch_data['feedback'].cpu().numpy())

            # print(f"valid_probs: {valid_probs}. valid_true: {valid_true}")
    pbar.close()
    probs = np.concatenate(valid_probs, axis=0).reshape(-1)  # 1D
    labels = np.concatenate(valid_true, axis=0).reshape(-1).astype(int)
    auc = roc_auc_score(labels, probs)
    print(probs)
    print(f"epoch {epo} validating" + "; auc: {:.4f}".format(np.mean(auc)))
    model.save_checkpoint()