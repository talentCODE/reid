#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function, absolute_import
import argparse
import copy
import os
import os.path as osp
import random
import numpy as np
import sys
import torch.nn.functional as F
import collections
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
# from scipy.special import softmax
from abmt import datasets
from abmt import models
from abmt.models import CosineLinear,SplitCosineLinear
from abmt.trainers import ABMTTrainer, UsicTrainer
from abmt.evaluators import Evaluator, extract_features
from abmt.utils.data import IterLoader
from abmt.utils.data import transforms as T
from abmt.utils.data.sampler import RandomMultipleGallerySampler
from abmt.utils.data.preprocessor import Preprocessor
from abmt.utils.logging import Logger
from abmt.utils.select_proto import select_proto, eval_nmi
from abmt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, unpickle, savepickle
from abmt.utils.rerank import compute_jaccard_dist
import math

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader



def create_model(args, classes, phase):
    '''
    创建模型，新建模型与保存的模型结构相同，以便可以进行载入权重
    :param args:
    :param classes:
    :param phase:
    :return:
    '''
    model_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=classes)
    initial_weights = load_checkpoint(osp.join(args.init, 'phase{}_model_best.pth.tar'.format(phase)))
    if 'module.classifier.fc1.weight' in initial_weights['state_dict'].keys():
        in_features = initial_weights['state_dict']['module.classifier.fc1.weight'].data.size(1)
        out_features1 = initial_weights['state_dict']['module.classifier.fc1.weight'].data.size(0)
        out_features2 = initial_weights['state_dict']['module.classifier.fc2.weight'].data.size(0)
        new_fc = SplitCosineLinear(in_features, out_features1, out_features2, sigma=True)
        model_ema.classifier = new_fc
        new_fc_max = copy.deepcopy(new_fc)
        model_ema.classifier_max = new_fc_max
    else:
        in_features = initial_weights['state_dict']['module.classifier.weight'].data.size(1)
        out_features = initial_weights['state_dict']['module.classifier.weight'].data.size(0)
        new_fc = CosineLinear(in_features=in_features, out_features=out_features, sigma=True)
        model_ema.classifier = new_fc
        new_fc_max = copy.deepcopy(new_fc)
        model_ema.classifier_max = new_fc_max
    copy_state_dict(initial_weights['state_dict'], model_ema, strip='module.')
    model_cur = copy.deepcopy(model_ema)
    model_ref = copy.deepcopy(model_ema)
    model_ema.cuda()
    model_ema = nn.DataParallel(model_ema)
    model_cur.cuda()
    model_cur = nn.DataParallel(model_cur)
    model_ref.cuda()
    model_ref = nn.DataParallel(model_ref)
    for param in model_ema.parameters():
        param.detach_()
    return model_cur, model_ema, model_ref


def get_old_class_num(proto_set):
    if len(proto_set) == 0:
        return 0
    max_pid = 0
    for (fname, pid, cam) in proto_set:
        if pid > max_pid:
            max_pid = pid
    return max_pid + 1


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    ########################################
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    ########处理dataset###########
    dataset_target = get_data(args.dataset_target, args.data_dir)

    # 创建测试使用的dataloader
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 128, args.workers)#args.batch_size * 4

    # 创建数据ID流
    order_name = "./checkpoint/seed_{}_{}_order.pkl".format(args.seed, args.dataset_target)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(dataset_target.num_train_pids)
        np.random.shuffle(order)
        savepickle(order, order_name)
    order_list = list(order)
    print(order_list)


    init_class_num = dataset_target.num_train_pids - (args.num_phase - 1) * args.nb_cl

    start_phase = 1
    for phase in range(start_phase, args.num_phase):
        global best_mAP
        best_mAP = 0
        if phase>0:
            proto_dataset = unpickle(osp.join(args.logs_dir, 'phase{}_proto.pkl'.format(phase - 1)))
        else:
            proto_dataset = []
        old_class_num = get_old_class_num(proto_dataset)
        print('\n\n phase {} have {} old classes'.format(phase, old_class_num))
        # 每个周期都需要载入最佳模型模型。
        model_cur, model_ema, model_ref = create_model(args, 100, phase)
        evaluator_ema = Evaluator(model_ema)
        # 更新 input dataset
        if phase == 0:
            iters = 200
            model_ref = None
            input_id = order_list[0:init_class_num]
            input_dataset = [(fname, pid, cam) for (fname, pid, cam) in dataset_target.train if pid in input_id]
            print('phase:{} input id:{},input image:{}'.format(phase, init_class_num, len(input_dataset)))
        else:
            iters = args.iters
            input_id = order_list[init_class_num + (phase - 1) * args.nb_cl:init_class_num + phase * args.nb_cl]
            input_dataset = [(fname, pid, cam) for (fname, pid, cam) in dataset_target.train if pid in input_id]
            print('phase:{} input id:{},input image:{}'.format(phase, args.nb_cl, len(input_dataset)))

        tar_cluster_loader = get_test_loader(dataset_target, args.height, args.width, 128,
                                             workers=args.workers,
                                             testset=sorted(input_dataset))


        for epoch in range(args.epochs):
            dict_f, _, dic_logit = extract_features(model_ema, tar_cluster_loader, print_freq=40)
            cf = torch.stack(list(dict_f.values())) #已经经过normalize
            agent_sim = torch.stack(list(dic_logit.values()))

            agent_sim = agent_sim[:, :old_class_num]  # probs dim=1,是否与old_num相同。
            agent_sim = F.softmax(agent_sim, dim=1)
            agent_sim_dist = torch.cdist(agent_sim, agent_sim, p=1) / 2
            agent_sim_dist = agent_sim_dist.numpy()
            rerank_dist = compute_jaccard_dist(cf, use_gpu=True).numpy()  # 经过rerank距离在0到1之间 args.rr_gpu
            lambda_a = 0.3 if phase==0 else 0
            total_dist = (1 - lambda_a) * rerank_dist + lambda_a * agent_sim_dist
            if (epoch == 0):
            #DBSCAN cluster
                tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
                tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
                tri_mat = np.sort(tri_mat, axis=None)
                rho = 2e-3
                top_num = np.round(rho * tri_mat.size).astype(int)
                eps = tri_mat[:top_num].mean()
                print('eps for cluster: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            print('Clustering and labeling...')
            labels = cluster.fit_predict(total_dist)
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            args.num_clusters = num_ids
            print('\n Clustered into {} classes \n'.format(args.num_clusters))

            # generate new dataset and calculate cluster centers
            new_dataset = []
            cluster_centers = collections.defaultdict(list)
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(input_dataset), labels)):
                if label == -1:
                    continue
                new_dataset.append((fname, label + old_class_num, cid))
                if label == -1:
                    print('error')
                cluster_centers[label].append(cf[i])

            cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
            cluster_centers = torch.stack(cluster_centers)
            cluster_centers_avg = F.normalize(cluster_centers[:, :2048], dim=1).float().cuda()
            cluster_centers_max = F.normalize(cluster_centers[:, 2048:], dim=1).float().cuda()
            # 新的类别为num_ids，开始更新模型分类器系数 consineLinear
            # 在每个epoch中不断更新 cur 和 ema的 new_fc层
            in_features = model_ema.module.classifier.in_features
            if phase == 0:
                out_features = args.num_clusters
                #创建new_fc_avg 为 model_cur和model_ema的avg赋值。
                new_fc_avg = CosineLinear(in_features=in_features, out_features=out_features, sigma=True).cuda()
                print("in_features:", in_features, "out_features:", out_features)
                # Evaluator
                new_fc_avg.weight.data.copy_(cluster_centers_avg)
                #new_fc.sigma.data = (model_ema.module.classifier.sigma.data+model_ema.module.classifier_max.sigma.data)/2  后面sigma 按20来处理
                model_ema.module.classifier = new_fc_avg
                model_cur.module.classifier = copy.deepcopy(new_fc_avg)

                # 创建new_fc_max 为 model_cur和model_ema的classifier_max赋值。
                new_fc_max= CosineLinear(in_features=in_features, out_features=out_features, sigma=True).cuda()
                new_fc_max.weight.data.copy_(cluster_centers_max)
                model_ema.module.classifier_max = new_fc_max
                model_cur.module.classifier_max = copy.deepcopy(new_fc_max)
                cur_lamda = 0

            elif phase == 1:
                ############################################################
                # increment classe
                if epoch == 0:
                    out_features = model_ema.module.classifier.out_features
                    print("in phase 1:epoch 0 : in_features:", in_features, "out_features:", out_features)

                    new_fc_cur = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_cur.fc1.weight.data = model_cur.module.classifier.weight.data  #shiyong id()lai panduan shifou tongyidizhi
                    new_fc_cur.fc2.weight.data.copy_(cluster_centers_avg)
                    model_cur.module.classifier = new_fc_cur

                    new_fc_cur_max = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_cur_max.fc1.weight.data = model_cur.module.classifier_max.weight.data
                    new_fc_cur_max.fc2.weight.data.copy_(cluster_centers_max)
                    model_cur.module.classifier_max = new_fc_cur_max

                    new_fc_ema = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_ema.fc1.weight.data = model_ema.module.classifier.weight.data
                    new_fc_ema.fc2.weight.data.copy_(cluster_centers_avg)
                    model_ema.module.classifier = new_fc_ema

                    new_fc_ema_max = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_ema_max.fc1.weight.data = model_ema.module.classifier_max.weight.data
                    new_fc_ema_max.fc2.weight.data.copy_(cluster_centers_max)
                    model_ema.module.classifier_max = new_fc_ema_max


                else:
                    out_features = model_ema.module.classifier.fc1.out_features
                    new_ema_fc = CosineLinear(in_features,args.num_clusters,sigma=False).cuda()
                    new_ema_fc.weight.data.copy_(cluster_centers_avg)
                    model_ema.module.classifier.fc2 = new_ema_fc

                    new_fc_cur = copy.deepcopy(new_ema_fc)
                    model_cur.module.classifier.fc2 = new_fc_cur

                    new_ema_fc_max = CosineLinear(in_features, args.num_clusters, sigma=False).cuda()
                    new_ema_fc_max.weight.data.copy_(cluster_centers_max)
                    model_ema.module.classifier_max.fc2 = new_ema_fc_max

                    new_fc_cur_max = copy.deepcopy(new_ema_fc_max)
                    model_cur.module.classifier_max.fc2 = new_fc_cur_max
                lamda_mult = out_features * 1.0 / args.num_clusters  # class_old / class_new
                cur_lamda = args.lamda * math.sqrt(lamda_mult)
                print("###############################")
                print("Lamda for less forget is set to ", cur_lamda)
                print("###############################")
                assert model_ema.module.classifier.fc1.weight.data.size(0) == old_class_num
                assert model_ema.module.classifier.fc2.weight.data.size(0) == args.num_clusters
            else:
                if epoch == 0:
                    out_features1 = model_ema.module.classifier.fc1.out_features
                    out_features2 = model_ema.module.classifier.fc2.out_features
                    out_features = out_features1 + out_features2
                    print("in_features:", in_features, "out_features1:", \
                          out_features1, "out_features2:", out_features2)

                    new_fc_cur = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_cur.fc1.weight.data[:out_features1].copy_(model_cur.module.classifier.fc1.weight.data)
                    new_fc_cur.fc1.weight.data[out_features1:].copy_(model_cur.module.classifier.fc2.weight.data)
                    new_fc_cur.fc2.weight.data.copy_(cluster_centers_avg)
                    #new_fc_cur.sigma.data = model_cur.module.classifier.sigma.data
                    model_cur.module.classifier = new_fc_cur

                    new_fc_cur_max = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_cur_max.fc1.weight.data[:out_features1].copy_(model_cur.module.classifier_max.fc1.weight.data)
                    new_fc_cur_max.fc1.weight.data[out_features1:].copy_(model_cur.module.classifier_max.fc2.weight.data)
                    new_fc_cur_max.fc2.weight.data.copy_(cluster_centers_max)
                    #new_fc_cur_max.sigma.data = model_cur.module.classifier_max.sigma.data
                    model_cur.module.classifier_max = new_fc_cur_max

                    new_fc_ema = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_ema.fc1.weight.data[:out_features1].copy_(model_ema.module.classifier.fc1.weight.data)
                    new_fc_ema.fc1.weight.data[out_features1:].copy_(model_ema.module.classifier.fc2.weight.data)
                    new_fc_ema.fc2.weight.data.copy_(cluster_centers_avg)
                    #new_fc_ema.sigma.data = model_ema.module.classifier.sigma.data
                    model_ema.module.classifier = new_fc_ema

                    new_fc_ema_max = SplitCosineLinear(in_features, out_features, args.num_clusters).cuda()
                    new_fc_ema_max.fc1.weight.data[:out_features1].copy_(model_ema.module.classifier_max.fc1.weight.data)
                    new_fc_ema_max.fc1.weight.data[out_features1:].copy_(model_ema.module.classifier_max.fc2.weight.data)
                    new_fc_ema_max.fc2.weight.data.copy_(cluster_centers_max)
                    #new_fc_ema_max.sigma.data  = model_ema.module.classifier_max.sigma.data
                    model_ema.module.classifier_max = new_fc_ema_max

                else:
                    out_features = model_ema.module.classifier.fc1.out_features

                    new_ema_fc = CosineLinear(in_features, args.num_clusters, sigma=False).cuda()
                    new_ema_fc.weight.data.copy_(cluster_centers_avg)
                    model_ema.module.classifier.fc2 = new_ema_fc

                    new_fc_cur = copy.deepcopy(new_ema_fc)
                    model_cur.module.classifier.fc2 = new_fc_cur

                    new_ema_fc_max = CosineLinear(in_features, args.num_clusters, sigma=False).cuda()
                    new_ema_fc_max.weight.data.copy_(cluster_centers_max)
                    model_ema.module.classifier_max.fc2 = new_ema_fc_max
                    new_fc_cur_max = copy.deepcopy(new_ema_fc_max)
                    model_cur.module.classifier_max.fc2 = new_fc_cur_max
                lamda_mult = (out_features) * 1.0 / (args.nb_cl)
                cur_lamda = args.lamda * math.sqrt(lamda_mult)
                print("###############################")
                print("Lamda for less forget is set to ", cur_lamda)
                print("###############################")
                assert model_ema.module.classifier.fc1.weight.data.size(0) == old_class_num
                assert model_ema.module.classifier.fc2.weight.data.size(0) == args.num_clusters

            # 生成相应的trainloader & optimizer
            # 设置optimizer.
            params = []
            for key, value in model_cur.named_parameters():
                if not value.requires_grad:
                    continue
                if key == 'module.classifier.sigma' or key == 'module.classifier_max.sigma':  # 需要更改温度系数的weight_decay, cosine loss中使用weight decay=0.1
                    params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay * 100}]
                    print('key: {} ,weight decay : {}, value : {}'.format(key, args.weight_decay * 100, value))
                elif key == 'module.classifier.fc1.weight' or key == 'module.classifier_max.fc1.weight':
                    params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}] #从零改变3.5e-4
                    print('lr of {} is 0'.format(key))
                else:
                    params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
            optimizer = torch.optim.Adam(params)

            # 设置dataloader与dataset
            if phase > 0:
                new_dataset.extend(proto_dataset)
            train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                                   args.batch_size, args.workers, args.num_instances, iters,
                                                   trainset=new_dataset)
            # Trainer
            trainer = UsicTrainer(model_cur=model_cur, model_ema=model_ema, model_ref=model_ref,
                                  old_class_num=old_class_num,new_class_num=args.num_clusters, alpha=args.alpha)

            train_loader_target.new_epoch()
            trainer.train(phase, epoch, train_loader_target, optimizer, cur_lamda,
                          ce_soft_weight=args.soft_ce_weight, print_freq=args.print_freq, train_iters=iters)

            def save_model(model_ema, is_best, best_mAP, phase):
                save_checkpoint({
                    'state_dict': model_ema.state_dict(),
                    'phase': phase,
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'phase{}_model_checkpoint.pth.tar'.format(phase)))

            if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
                mAP_1 = evaluator_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                               cmc_flag=False)
                is_best = mAP_1 > best_mAP
                best_mAP = max(mAP_1, best_mAP)

                save_model(model_ema, is_best, best_mAP, phase + 1)
                print('\n * Finished phase {:3d} epoch {:3d}  model no.1 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(phase, epoch, mAP_1, best_mAP, ' *' if is_best else ''))

        # 更新examplar
        # 载入最佳模型到model_ema
        print('update proto_dataset')
        best_weights = load_checkpoint(osp.join(args.init, 'phase{}_model_best.pth.tar'.format(phase + 1)))
        copy_state_dict(best_weights['state_dict'], model_ema)
        # 提取特征，并进行聚类
        dict_f, _, dic_logit = extract_features(model_ema, tar_cluster_loader, print_freq=40)
        cf = torch.stack(list(dict_f.values()))  # 已经经过normalize
        agent_sim = torch.stack(list(dic_logit.values()))

        agent_sim = agent_sim[:, :old_class_num]  # probs dim=1,是否与old_num相同。
        agent_sim = F.softmax(agent_sim, dim=1)
        agent_sim_dist = torch.cdist(agent_sim, agent_sim, p=1) / 2
        agent_sim_dist = agent_sim_dist.numpy()
        rerank_dist = compute_jaccard_dist(cf, use_gpu=True).numpy()  # 经过rerank距离在0到1之间 args.rr_gpu
        lambda_a = 0.3 if phase == 0 else 0
        rerank_dist = (1 - lambda_a) * rerank_dist + lambda_a * agent_sim_dist
        # dict_f, _,_= extract_features(model_ema, tar_cluster_loader, print_freq=40)
        # cf = torch.stack(list(dict_f.values()))
        # rerank_dist = compute_jaccard_dist(cf, use_gpu=args.rr_gpu).numpy()

        #generate DBSCAN
        if epoch==0:
            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            rho = 2e-3
            top_num = np.round(rho * tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        args.num_clusters = num_ids
        print('phase {}， Clustered into {} example classes '.format(phase, args.num_clusters))
        # 计算每个类的所有特征
        class_features = collections.defaultdict(list)
        image_index = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            class_features[label].append(cf[i])
            image_index[label].append(i)
        class_features = [torch.stack(class_features[idx]) for idx in
                          sorted(class_features.keys())]  # list，其中元素为每个类特征组成的张量
        image_index = [image_index[idx] for idx in sorted(image_index.keys())]
        tmp_dataset = sorted(input_dataset)
        #example_dataset, class_remain = select_proto(class_features, image_index, tmp_dataset, old_class_num=old_class_num)
        example_dataset, class_remain = select_proto(class_features, image_index, tmp_dataset, old_class_num=old_class_num,
                                                     mode='herd',delete_ratio=0.05)
        nmi = eval_nmi(example_dataset)
        print('NMI of phase{} is {} '.format(phase, nmi))

        # 保存此时的proto_dataset
        proto_dataset.extend(example_dataset)
        proto_dataset_save = osp.join(args.logs_dir, 'phase{}_proto.pkl'.format(phase))
        savepickle(proto_dataset, proto_dataset_save)

        # 更新最新模型FC参数
        class_features = torch.stack([torch.mean(features, dim=0) for features in class_remain])
        class_features_avg = F.normalize(class_features[:, :2048], dim=1).float().cuda()
        class_features_max = F.normalize(class_features[:, 2048:], dim=1).float().cuda()
        in_features = model_ema.module.classifier.in_features
        if isinstance(model_ema.module.classifier, CosineLinear):
            new_fc_avg = CosineLinear(in_features, len(class_features), sigma=True)
            new_fc_avg.weight.data.copy_(class_features_avg)
            # new_fc.sigma.data = model_ema.module.classifier.sigma.data
            model_ema.module.classifier = new_fc_avg

            new_fc_max = CosineLinear(in_features, len(class_features), sigma=True)
            new_fc_max.weight.data.copy_(class_features_max)
            # new_fc_max.sigma.data = model_ema.module.classifier_max.sigma.data
            model_ema.module.classifier_max = new_fc_max

        else:
            new_fc_avg = CosineLinear(in_features, len(class_features), sigma=False)
            new_fc_avg.weight.data.copy_(class_features_avg)
            model_ema.module.classifier.fc2 = new_fc_avg

            new_fc_max = CosineLinear(in_features, len(class_features), sigma=False)
            new_fc_max.weight.data.copy_(class_features_max)
            model_ema.module.classifier_max.fc2 = new_fc_max

        state = {
            'state_dict': model_ema.state_dict(),
            'phase': best_weights['phase'],
            'epoch': best_weights['epoch'],
            'best_mAP': best_weights['best_mAP'],
        }

        torch.save(state, osp.join(args.logs_dir, 'phase{}_model_best.pth.tar'.format(phase + 1)))

        # # 更新examplar
        # # 载入最佳模型到model_ema
        # print('update proto_dataset')
        # best_weights = load_checkpoint(osp.join(args.init, 'phase{}_model_best.pth.tar'.format(phase + 1)))
        # copy_state_dict(best_weights['state_dict'], model_ema)
        # # 提取特征，并进行聚类
        # dict_f, _,_= extract_features(model_ema, tar_cluster_loader, print_freq=40)
        # cf = torch.stack(list(dict_f.values()))
        # rerank_dist = compute_jaccard_dist(cf, use_gpu=args.rr_gpu).numpy()
        # print('Clustering and labeling...')
        # labels = cluster.fit_predict(rerank_dist)
        # num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        # args.num_clusters = num_ids
        # print('phase {}， Clustered into {} example classes '.format(phase, args.num_clusters))
        # # 计算每个类的所有特征
        # class_features = collections.defaultdict(list)
        # image_index = collections.defaultdict(list)
        # for i, label in enumerate(labels):
        #     if label == -1:
        #         continue
        #     class_features[label].append(cf[i])
        #     image_index[label].append(i)
        # class_features = [torch.stack(class_features[idx]) for idx in
        #                   sorted(class_features.keys())]  # list，其中元素为每个类特征组成的张量
        # image_index = [image_index[idx] for idx in sorted(image_index.keys())]
        # tmp_dataset = sorted(input_dataset)
        # for i in range(len(class_features)):
        #     class_mean = class_features[i].mean(dim=0, keepdim=True)
        #     class_mean = F.normalize(class_mean, p=2, dim=1)
        #     examplar_features = []
        #     m = 4
        #     for k in range(m):
        #         if len(examplar_features) == 0:
        #             S = torch.sum(torch.tensor(examplar_features))
        #         else:
        #             S = torch.sum(torch.stack(examplar_features, dim=0), dim=0)
        #         phi = class_features[i]
        #         mu = class_mean
        #         mu_p = 1.0 / (k + 1) * (phi + S)
        #         mu_p = F.normalize(mu_p, dim=1)
        #         idx = torch.argmin(torch.sum((mu - mu_p) ** 2, axis=1))
        #         img_idx = image_index[i][idx]
        #         proto_dataset.append((tmp_dataset[img_idx][0], i + old_class_num, tmp_dataset[img_idx][2]))
        #         examplar_features.append(class_features[i][idx])
        #
        # # 保存此时的proto_dataset
        # proto_dataset_save = osp.join(args.logs_dir, 'phase{}_proto.pkl'.format(phase))
        # savepickle(proto_dataset, proto_dataset_save)
        #
        # # 更新最新模型FC参数
        # class_features = torch.stack([torch.mean(features, dim=0) for features in class_features])
        # class_features_avg = F.normalize(class_features[:, :2048], dim=1).float().cuda()
        # class_features_max = F.normalize(class_features[:, 2048:], dim=1).float().cuda()
        #
        # if isinstance(model_ema.module.classifier, CosineLinear):
        #     new_fc_avg = CosineLinear(in_features, len(class_features), sigma=True)
        #     new_fc_avg.weight.data.copy_(class_features_avg)
        #     #new_fc.sigma.data = model_ema.module.classifier.sigma.data
        #     model_ema.module.classifier = new_fc_avg
        #
        #     new_fc_max = CosineLinear(in_features, len(class_features), sigma=True)
        #     new_fc_max.weight.data.copy_(class_features_max)
        #    # new_fc_max.sigma.data = model_ema.module.classifier_max.sigma.data
        #     model_ema.module.classifier_max = new_fc_max
        #
        # else:
        #     new_fc_avg = CosineLinear(in_features, len(class_features), sigma=False)
        #     new_fc_avg.weight.data.copy_(class_features_avg)
        #     model_ema.module.classifier.fc2 = new_fc_avg
        #
        #     new_fc_max = CosineLinear(in_features, len(class_features), sigma=True)
        #     new_fc_max.weight.data.copy_(class_features_max)
        #     model_ema.module.classifier_max.fc2 = new_fc_max
        #
        #
        # state = {
        #     'state_dict': model_ema.state_dict(),
        #     'phase': best_weights['phase'],
        #     'epoch': best_weights['epoch'],
        #     'best_mAP': best_weights['best_mAP'],
        # }
        #
        # torch.save(state, osp.join(args.logs_dir, 'phase{}_model_best.pth.tar'.format(phase + 1)))


if __name__ == "__main__":
    ######### Modifiable Settings ##########

    parser = argparse.ArgumentParser(description='unsupervise increment re-identification')
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnetCosine50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=100)
    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    ##############
    parser.add_argument('--num_phase', default=5, type=int, \
                        help='the number of classes in first group')
    parser.add_argument('--nb_cl', default=100, type=int, \
                        help='Classes per group')
    parser.add_argument('--nb_protos', default=4, type=int, \
                        help='Number of prototypes per class at the end')
    parser.add_argument('--nb_runs', default=1, type=int, \
                        help='Number of runs (random ordering of classes at each run)')
    parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, \
                        help='Checkpoint prefix')
    # parser.add_argument('--resume', action='store_true', \
    #     help='resume from checkpoint')
    # parser.add_argument('--less_forget', action='store_true', \
    #                     help='Less forgetful')
    parser.add_argument('--lamda', default=2, type=float, \
                        help='Lamda for LF')

    main()
