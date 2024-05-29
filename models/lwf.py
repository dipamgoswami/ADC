import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from utils.data_manager import DummyDataset
from utils.inc_net import IncrementalNet, CosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
from utils.attack import Attack

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 100
lrate = 0.05
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 10

# Tiny-ImageNet200
# epochs = 100
# lrate = 0.001
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

# refer to supplementary materials for other dataset training settings

EPSILON = 1e-8

class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["cosine"]:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = CosineIncrementalNet(args, True)
            else:
                self._network = CosineIncrementalNet(args, False)
        else:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = IncrementalNet(args, True)
            else:
                self._network = IncrementalNet(args, False)
        self._protos = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args['resume']:
            self.save_checkpoint("{}{}_{}_{}_{}".format(self.args["model_dir"],self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"]))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100" or self.args['dataset'] == "cub200" or self.args['dataset'] == "cars":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.shot = None
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            shot=self.shot
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args['resume']  # set resume=True to use saved checkpoints
        if self._cur_task == 0:
            if resume:
                self._network.load_state_dict(torch.load("{}{}_{}_{}_{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._build_protos()
        else:
            resume = self.args['resume']
            if resume:
                self._network.load_state_dict(torch.load("{}{}_{}_{}_{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if self._old_network is not None:
                self._old_network.to(self._device)
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._build_protos()                

            # compensate the semantic drift in the prototypes with ADC
            if self.args["ADC"]:
                epoch = self.args["adv_epoch"]
                print('alpha: ',self.args["alpha"])

                for k, (_, data, label) in enumerate(train_loader):
                    if k == 0:
                        x_min = data.min()
                        x_max = data.max()
                    else:
                        if data.min() < x_min:
                            x_min = data.min()
                        if data.max() > x_max:
                            x_max = data.max()

                xx, yy, feats = [], [], []
                for _, data, label in train_loader:
                    xx.append(data)
                    yy.append(label)
                    feats.append(self._old_network(data.to(self._device))["features"])

                xx = torch.cat(xx, dim=0)
                yy = torch.cat(yy, dim=0)
                feats = torch.cat(feats, dim=0)

                for class_idx in range(0, self._known_classes):
                    d = torch.cdist(feats, self._protos[class_idx].unsqueeze(0)).squeeze()
                    closest = torch.argsort(d)[:self.args["sample_limit"]].cpu()
                    x_top = xx[[closest]]
                    y_top = yy[[closest]]
                    
                    idx_dataset = TensorDataset(x_top, y_top)
                    loader = DataLoader(idx_dataset, batch_size=int(self.args["sample_limit"]), shuffle=False)

                    attack = Attack(self._old_network, self._network, self.args["alpha"], loader, 
                                    self._protos[:self._known_classes], self._device, epoch, x_min, x_max, class_idx)
                    
                    x_, y_ = attack.run()
                    if len(x_) > 0:
                        idx_dataset = TensorDataset(x_, y_)
                    idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False)

                    if idx_loader is not None:
                        vectors_old = self._extract_vectors_adv(idx_loader, old=True)[0]
                        vectors = self._extract_vectors_adv(idx_loader)[0]

                    MU = np.asarray(self._protos[class_idx].unsqueeze(0).cpu())
                    gap = np.mean(vectors - vectors_old, axis=0)
                    MU += gap
                    self._protos[class_idx] = torch.tensor(MU).squeeze(0).to(self._device)
                    
            # compensate the semantic drift in the prototypes with SDC
            if self.args["SDC"]:
                emb_old = self._extract_vectors(train_loader, old=True)[0]
                emb = self._extract_vectors(train_loader)[0]
                MU = np.asarray(torch.stack(self._protos[:self._known_classes]).cpu())
                gap = self.displacement(emb_old, emb, MU, self.args["sigma"])

                MU += gap
                self._protos[:self._known_classes] = torch.tensor(MU).to(self._device)

    # SDC for the prototypes
    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = (Y2 - Y1)
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
        W = np.exp(-distance/(2*sigma ** 2)) +1e-5
        #print(W) # 1e-5
        W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement

    def _build_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                mode='test', shot=self.shot, ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0) # vectors.mean(0)
            self._protos.append(torch.tensor(class_mean).to(self._device))

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = lamda * loss_kd + loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

