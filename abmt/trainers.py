from __future__ import print_function, absolute_import
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy, SoftWeightedTriplet
from .utils.meters import AverageMeter
from torch.nn import MSELoss
from sklearn.mixture import GaussianMixture


class ABMTPreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(ABMTPreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_ce_no_ls = nn.CrossEntropyLoss().cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()
        #scales = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_features_m, s_cls_out, s_cls_out_m = self.model(s_inputs)
            # target samples: only forward
            _ = self.model(t_inputs)

            # backward
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_features_m, s_cls_out, s_cls_out_m, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scales = self.model.module.classifier.sigma.item()
            scales_max = self.model.module.classifier_max.sigma.item()
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                        'scales and scales_max {:.3f} {:.3f}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg,
                              scales, scales_max
                              ))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_features_m, s_cls_out, s_cls_out_m, targets):
        loss_ce = (self.criterion_ce_no_ls(s_cls_out, targets) + self.criterion_ce_no_ls(s_cls_out_m, targets))/2 #调整为没有soft label smooth 的 交叉熵。
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = (self.criterion_triple(s_features, s_features, targets) + self.criterion_triple(s_features_m, s_features_m, targets))/2
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = (self.criterion_triple(s_features, targets) + self.criterion_triple(s_features_m, targets))/2
        prec, = accuracy(s_cls_out.data, targets.data)
        prec_p, = accuracy(s_cls_out_m.data, targets.data)
        prec = (prec[0] + prec_p[0])/2

        return loss_ce, loss_tr, prec


class ABMTTrainer(object):
    def __init__(self, model_1, model_1_ema, num_cluster=500, alpha=0.999):
        super(ABMTTrainer, self).__init__()
        self.model_1 = model_1
        self.num_cluster = num_cluster
        self.model_1_ema = model_1_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        # self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_1_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)

            # forward
            f_out_t1, f_out_t1_m, p_out_t1, p_out_t1_m = self.model_1(inputs_1)
            f_out_t1_ema, f_out_t1_ema_m, p_out_t1_ema, p_out_t1_ema_m = self.model_1_ema(inputs_1)

            loss_ce = (self.criterion_ce(p_out_t1, targets) + self.criterion_ce(p_out_t1_m, targets))/2

            # loss_tri_1 = (self.criterion_tri(f_out_t1, f_out_t1, targets) + self.criterion_tri(f_out_t1_p, f_out_t1_p, targets))/2

            loss_ce_soft = (self.criterion_ce_soft(p_out_t1, p_out_t1_ema_m) + self.criterion_ce_soft(p_out_t1_m, p_out_t1_ema))/2
            loss_tri_soft = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m, targets) + self.criterion_tri_soft(f_out_t1_m, f_out_t1_ema, targets))/2

            loss = (loss_ce + loss_ce_soft)/2 + loss_tri_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ce[0].update(loss_ce.item())
            # losses_tri[0].update(loss_tri_1.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg,
                              ))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets


class UsicTrainer2(object):
    def __init__(self, model_cur, model_ema,
                 model_ref, old_class_num, new_class_num, alpha=0.999):
        super(UsicTrainer2, self).__init__()
        self.model_cur = model_cur
        self.model_ema = model_ema
        self.model_ref = model_ref
        self.old_class_num = old_class_num
        self.new_class_num = new_class_num
        self.total_class_num = old_class_num + new_class_num
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(self.total_class_num).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        # self.criterion_tri_soft = SoftWeightedTriplet().cuda()

    def train(self, phase, epoch, train_loader_examplar, data_loader_target,
              optimizer, lamda, ce_soft_weight=0.5, print_freq=1, train_iters=200):

        self.model_cur.train()
        self.model_ema.train()
        if phase > 0:
            self.model_ref.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(), AverageMeter()]
        losses_tri = [AverageMeter(), AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        losses_lf = AverageMeter()
        losses_lf_tri = AverageMeter()
        losses_embedding = AverageMeter()
        precisions = [AverageMeter(), AverageMeter()]

        end = time.time()
        for i in range(train_iters):

            # source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            # data_time.update(time.time() - end)
            #
            # # process inputs
            # s_inputs, s_targets, _ = self._parse_data(source_inputs)
            # t_inputs, _, t_indexes = self._parse_data(target_inputs)
            #
            # # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)
            # # 上面操作可以使得每一个GPU拥有相同的源域和目标域数据比例。
            #
            # # forward
            # f_out = self._forward(inputs)
            #
            # # de-arrange batch
            # f_out = f_out.view(device_num, -1, f_out.size(-1))
            # f_out_s, f_out_t = f_out.split(f_out.size(1) // 2, dim=1)
            # f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1,
            #                                                                                             f_out.size(-1))
            #
            # # compute loss with the hybrid memory
            # loss_s = self.memory(f_out_s, s_targets)
            # loss_t = self.memory(f_out_t, t_indexes + self.source_classes)
            #
            # loss = loss_s + loss_t
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # losses_s.update(loss_s.item())
            # losses_t.update(loss_t.item())

            examplar_inputs=train_loader_examplar.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_exp1, inputs_exp2, targets_exp = self._parse_data(examplar_inputs)
            inputs_tar1, inputs_tar2, targets_tar = self._parse_data(target_inputs)

            inputs_1, inputs_2, targets =torch.cat([inputs_exp1,inputs_tar1]),torch.cat([inputs_exp2,inputs_tar2]),torch.cat([targets_exp,targets_tar])
            # forward
            f_out_t1, f_out_t1_m, p_out_t1, p_out_t1_m = self.model_cur(inputs_1)
            f_out_t1_ema, f_out_t1_ema_m, p_out_t1_ema, p_out_t1_ema_m = self.model_ema(inputs_2)

            loss_ce = (self.criterion_ce(p_out_t1, targets) + self.criterion_ce(p_out_t1_m, targets)) / 2

            # loss_tri_1 = (self.criterion_tri(f_out_t1, f_out_t1, targets) + self.criterion_tri(f_out_t1_p, f_out_t1_p, targets))/2

            loss_ce_soft = (self.criterion_ce_soft(p_out_t1, p_out_t1_ema_m) + self.criterion_ce_soft(p_out_t1_m,
                                                                                                      p_out_t1_ema)) / 2
            loss_tri_soft = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m, targets) + self.criterion_tri_soft(
                f_out_t1_m, f_out_t1_ema, targets)) / 2
            # loss_tri_soft = torch.tensor([0]).cuda()

            if phase > 0:
                # f_ref_out, f_ref_out_m,_,_ = self.model_ref(inputs_1)
                # bs= f_out_t1.shape[0]
                # loss_lf = (nn.CosineEmbeddingLoss()(f_out_t1, f_ref_out.detach(),torch.ones(bs).cuda())+ \
                #           nn.CosineEmbeddingLoss()(f_out_t1_m, f_ref_out_m.detach(), torch.ones(bs).cuda()))/2 * lamda
                logit_t1_old = p_out_t1[:, :self.old_class_num]
                logit_t1_m_old = p_out_t1_m[:, :self.old_class_num]
                with torch.no_grad():
                    # ref_feature, ref_logit = self.model_ref(inputs_1, eval_logit=True)
                    # loss_lf = nn.KLDivLoss()(F.log_softmax(logit_t1_old , dim=1), F.softmax(ref_logit , dim=1))
                    # # loss_tri_soft_ref = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m,
                    # #                                          targets) + self.criterion_tri_soft(f_out_t1_m,
                    # #                                                                             f_out_t1_ema,
                    # #                                                                             targets)) / 2
                    f_out_ref, f_out_ref_m, p_out_ref, p_out_ref_m = self.model_ref(inputs_1)
                # ref_logit = (p_out_ref+p_out_ref_m)/2

                loss_lf_logit = (self.criterion_ce_soft(logit_t1_old, p_out_ref) + self.criterion_ce_soft(
                    logit_t1_m_old, p_out_ref_m)) / 2
                # loss_lf_logit=torch.tensor([0]).cuda()
                # loss_lf_logit = nn.KLDivLoss()(F.log_softmax(logit_t1_old, dim=1), F.softmax(p_out_ref, dim=1))+\
                #                 nn.KLDivLoss()(F.log_softmax(logit_t1_m_old, dim=1), F.softmax(p_out_ref_m, dim=1))

                loss_lf_tri = (self.criterion_tri_soft(f_out_t1, f_out_ref, targets) +
                               self.criterion_tri_soft(f_out_t1_m, f_out_ref_m, targets)) / 2
                # loss_lf_tri=torch.tensor([0]).cuda()
                # mean_old_embedding = torch.mean(F.normalize(self.model_cur.module.classifier.fc1.weight,p=2,dim=1),dim=0)
                # mean_new_embedding = torch.mean(F.normalize(self.model_cur.module.classifier.fc2.weight,p=2,dim=1),dim=0)
                # mean_old_embedding_max = torch.mean(
                #     F.normalize(self.model_cur.module.classifier_max.fc1.weight, p=2, dim=1), dim=0)
                # mean_new_embedding_max = torch.mean(
                #     F.normalize(self.model_cur.module.classifier_max.fc2.weight, p=2, dim=1), dim=0)
                # loss_embedding = torch.sum(torch.pow(mean_old_embedding - mean_new_embedding, 2)) + \
                #                    torch.sum(torch.pow(mean_old_embedding_max - mean_new_embedding_max, 2))

                # loss_lf_tri=0

                loss = loss_ce * (
                            1 - ce_soft_weight) + loss_ce_soft * ce_soft_weight + loss_lf_tri + loss_lf_logit + loss_tri_soft  # lamda *lamda
            else:
                loss = loss_ce * (1 - ce_soft_weight) + loss_ce_soft * ce_soft_weight + loss_tri_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_cur, self.model_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ce[0].update(loss_ce.item())
            # losses_tri[0].update(loss_tri_1.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            losses_lf.update(loss_lf_logit.item() if phase > 0 else 0)
            losses_lf_tri.update(loss_lf_tri.item() if phase > 0 else 0)
            # losses_embedding.update(0 if phase > 0 else 0)
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_lf {:.3f}\t'
                      'Loss_lf_tri {:.3f}\t'

                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              losses_lf.avg, losses_lf_tri.avg,
                              precisions[0].avg, precisions[1].avg,
                              ))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets





class UsicTrainer(object):
    def __init__(self, model_cur, model_ema,
                       model_ref, old_class_num,new_class_num, alpha=0.999):
        super(UsicTrainer, self).__init__()
        self.model_cur = model_cur
        self.model_ema = model_ema
        self.model_ref = model_ref
        self.old_class_num = old_class_num
        self.new_class_num = new_class_num
        self.total_class_num = old_class_num + new_class_num
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(self.total_class_num).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        # self.criterion_tri_soft = SoftWeightedTriplet().cuda()



    def train(self, phase, epoch, data_loader_target,
              optimizer, lamda, ce_soft_weight=0.5, print_freq=1, train_iters=200):

        self.model_cur.train()
        self.model_ema.train()
        if phase > 0:
            self.model_ref.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        losses_lf = AverageMeter()
        losses_lf_tri = AverageMeter()
        losses_embedding = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)

            # forward
            f_out_t1, f_out_t1_m, p_out_t1, p_out_t1_m = self.model_cur(inputs_1)
            f_out_t1_ema, f_out_t1_ema_m, p_out_t1_ema, p_out_t1_ema_m = self.model_ema(inputs_2)

            loss_ce = (self.criterion_ce(p_out_t1, targets) + self.criterion_ce(p_out_t1_m, targets))/2

            # loss_tri_1 = (self.criterion_tri(f_out_t1, f_out_t1, targets) + self.criterion_tri(f_out_t1_p, f_out_t1_p, targets))/2

            loss_ce_soft = (self.criterion_ce_soft(p_out_t1, p_out_t1_ema_m) + self.criterion_ce_soft(p_out_t1_m, p_out_t1_ema))/2
            loss_tri_soft = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m, targets) + self.criterion_tri_soft(f_out_t1_m, f_out_t1_ema, targets))/2
            # loss_tri_soft = torch.tensor([0]).cuda()

            if phase>0:
                # f_ref_out, f_ref_out_m,_,_ = self.model_ref(inputs_1)
                # bs= f_out_t1.shape[0]
                # loss_lf = (nn.CosineEmbeddingLoss()(f_out_t1, f_ref_out.detach(),torch.ones(bs).cuda())+ \
                #           nn.CosineEmbeddingLoss()(f_out_t1_m, f_ref_out_m.detach(), torch.ones(bs).cuda()))/2 * lamda
                logit_t1_old = p_out_t1[:,:self.old_class_num]
                logit_t1_m_old = p_out_t1_m[:,:self.old_class_num]
                with torch.no_grad():
                    # ref_feature, ref_logit = self.model_ref(inputs_1, eval_logit=True)
                    # loss_lf = nn.KLDivLoss()(F.log_softmax(logit_t1_old , dim=1), F.softmax(ref_logit , dim=1))
                    # # loss_tri_soft_ref = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m,
                    # #                                          targets) + self.criterion_tri_soft(f_out_t1_m,
                    # #                                                                             f_out_t1_ema,
                    # #                                                                             targets)) / 2
                    f_out_ref, f_out_ref_m, p_out_ref, p_out_ref_m=self.model_ref(inputs_1)
                # ref_logit = (p_out_ref+p_out_ref_m)/2

                loss_lf_logit = (self.criterion_ce_soft(logit_t1_old, p_out_ref) + self.criterion_ce_soft(logit_t1_m_old, p_out_ref_m))/2
                # loss_lf_logit=torch.tensor([0]).cuda()
                # loss_lf_logit = nn.KLDivLoss()(F.log_softmax(logit_t1_old, dim=1), F.softmax(p_out_ref, dim=1))+\
                #                 nn.KLDivLoss()(F.log_softmax(logit_t1_m_old, dim=1), F.softmax(p_out_ref_m, dim=1))

                loss_lf_tri = (self.criterion_tri_soft(f_out_t1, f_out_ref, targets) +
                               self.criterion_tri_soft(f_out_t1_m, f_out_ref_m, targets)) / 2
                # loss_lf_tri=torch.tensor([0]).cuda()
                # mean_old_embedding = torch.mean(F.normalize(self.model_cur.module.classifier.fc1.weight,p=2,dim=1),dim=0)
                # mean_new_embedding = torch.mean(F.normalize(self.model_cur.module.classifier.fc2.weight,p=2,dim=1),dim=0)
                # mean_old_embedding_max = torch.mean(
                #     F.normalize(self.model_cur.module.classifier_max.fc1.weight, p=2, dim=1), dim=0)
                # mean_new_embedding_max = torch.mean(
                #     F.normalize(self.model_cur.module.classifier_max.fc2.weight, p=2, dim=1), dim=0)
                # loss_embedding = torch.sum(torch.pow(mean_old_embedding - mean_new_embedding, 2)) + \
                #                    torch.sum(torch.pow(mean_old_embedding_max - mean_new_embedding_max, 2))

                    # loss_lf_tri=0

                loss = loss_ce * (1 - ce_soft_weight) + loss_ce_soft * ce_soft_weight + loss_lf_tri + loss_lf_logit+ loss_tri_soft#lamda *lamda
            else:
                loss = loss_ce * (1 - ce_soft_weight) + loss_ce_soft * ce_soft_weight + loss_tri_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_cur, self.model_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ce[0].update(loss_ce.item())
            # losses_tri[0].update(loss_tri_1.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            losses_lf.update(loss_lf_logit.item() if phase > 0 else 0)
            losses_lf_tri.update(loss_lf_tri.item() if phase > 0 else 0)
            # losses_embedding.update(0 if phase > 0 else 0)
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_lf {:.3f}\t'
                      'Loss_lf_tri {:.3f}\t'
            
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              losses_lf.avg,losses_lf_tri.avg,
                              precisions[0].avg, precisions[1].avg,
                              ))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets


class UsicTrainer_E(object):
    '''
    这里使用了essential 中的方法，将分类器分离来减少类间不平衡问题。
    使用new_dataset 来训练新数据的分类问题
    使用添加了new_examplar 的proto_dataset 来进行整个fc层训练，保证新类旧类的分离。
    '''
    def __init__(self, model_cur, model_ema,
                       model_ref, old_class_num, new_class_num, alpha=0.999):

        super(UsicTrainer_E, self).__init__()
        self.model_cur = model_cur
        self.model_ema = model_ema
        self.model_ref = model_ref
        self.old_class_num = old_class_num
        self.new_class_num = new_class_num
        self.total_class_num = old_class_num + new_class_num
        self.alpha = alpha

        self.criterion_ce1 = CrossEntropyLabelSmooth(new_class_num).cuda()
        self.criterion_ce2 = CrossEntropyLabelSmooth(old_class_num + new_class_num).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()


    def train(self, phase, epoch, data_loader_target,data_loader_examplar,
              optimizer, lamda, ce_soft_weight=0.5, print_freq=1, train_iters=200):

        self.model_cur.train()
        self.model_ema.train()
        if phase > 0:
            self.model_ref.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = [AverageMeter(),AverageMeter()]
        losses_tri_soft = [AverageMeter(),AverageMeter()]
        losses_lf = [AverageMeter(), AverageMeter()]
        losses_lf_tri = [AverageMeter(), AverageMeter()]
        precisions = [AverageMeter(),AverageMeter()]

        # def calculate_loss(self,xx_inputs,begin,end, adaptive_lamda, cal_dist=False,T=2):
        #     inputs_1, inputs_2, targets = self._parse_data(xx_inputs)
        #     # input forward
        #     f_out_t1, f_out_t1_m, logit_t1, logit_t1_m = self.model_cur(inputs_1)
        #     p_out_t1 = logit_t1[:, begin:end]
        #     p_out_t1_m = logit_t1_m[:, begin:end]
        #     f_out_t1_ema, f_out_t1_ema_m, p_out_t1_ema, p_out_t1_ema_m = self.model_ema(inputs_1)
        #     p_out_t1_ema = p_out_t1_ema[:, begin:end]
        #     p_out_t1_ema_m = p_out_t1_ema_m[:, begin:end]
        #     loss_ce = (self.criterion_ce1(p_out_t1, targets - begin) +
        #                self.criterion_ce1(p_out_t1_m,targets - begin)) / 2
        #     loss_ce_soft = (self.criterion_ce_soft(p_out_t1, p_out_t1_ema_m) +
        #                     self.criterion_ce_soft(p_out_t1_m, p_out_t1_ema)) / 2
        #     loss_tri_soft = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m, targets) + self.criterion_tri_soft(
        #         f_out_t1_m, f_out_t1_ema, targets)) / 2
        #     if cal_dist:
        #         with torch.no_grad():
        #             ref_logit = self.model_ref(inputs_1, eval_logit=True)
        #         logit_t1_old = logit_t1[:, :begin]
        #
        #         loss_dist = nn.KLDivLoss()(F.log_softmax(logit_t1_old/ T, dim=1),
        #                                           F.softmax(ref_logit / T, dim=1))
        #     else:
        #         loss_dist=0
        #     loss_target_input = loss_ce * (1 - 0.5) + loss_ce_soft * 0.5 + loss_tri_soft + loss_dist * adaptive_lamda
        #     # ce_soft_weight = 0.5
        #     return loss_target_input, loss_ce, loss_ce_soft, loss_tri_soft, loss_dist


        end = time.time()
        use_dist = True if phase > 0 else False
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            loss_target_total, loss_ce_target, loss_ce_soft_target, loss_tri_soft_target, loss_dist_target,loss_dist_tri_target,pre_target=\
                self.calculate_loss(target_inputs,self.old_class_num,self.total_class_num,lamda,self.criterion_ce1,cal_dist= use_dist,T=2)#lamda
            if phase>0:
                examplar_inputs = data_loader_examplar.next()
                loss_exa_total, loss_ce_exa, loss_ce_soft_exa, loss_tri_soft_exa, loss_dist_exa,loss_dist_tri_exa,pre_exa = \
                self.calculate_loss(examplar_inputs, 0, self.total_class_num, lamda,self.criterion_ce2, cal_dist=use_dist,T=2)#lamda

            else:
                loss_exa_total=0
            loss = loss_target_total*0.2 + loss_exa_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_cur, self.model_ema, self.alpha, epoch * len(data_loader_target) + i)



            losses_ce[0].update(loss_ce_target.item())
            losses_ce[1].update(loss_ce_exa.item() if phase>0 else 0)
            # losses_tri[0].update(loss_tri_1.item())
            losses_ce_soft[0].update(loss_ce_soft_target.item())
            losses_ce_soft[1].update(loss_ce_soft_exa.item() if phase>0 else 0)
            losses_tri_soft[0].update(loss_tri_soft_target.item())
            losses_tri_soft[1].update(loss_tri_soft_exa.item() if phase>0 else 0)
            losses_lf[0].update(loss_dist_target.item() if phase>0 else 0)
            losses_lf[1].update(loss_dist_exa.item() if phase>0 else 0)
            losses_lf_tri[0].update(loss_dist_tri_target.item() if phase>0 else 0)
            losses_lf_tri[1].update(loss_dist_tri_exa.item() if phase>0 else 0)
            precisions[0].update(pre_target[0].item())
            precisions[1].update(pre_exa[0].item() if phase>0 else 0)

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f} / {:.3f}\t'
                      'Loss_tri_soft {:.3f} / {:.3f}\t'
                      'Loss_lf {:.3f} / {:.3f}\t'
                      'Loss_lf_tri {:.3f} / {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_ce_soft[0].avg, losses_ce_soft[1].avg,
                              losses_tri_soft[0].avg, losses_tri_soft[1].avg,
                              losses_lf[0].avg, losses_lf[1].avg,
                              losses_lf_tri[0].avg,losses_lf_tri[1].avg,
                              precisions[0].avg, precisions[1].avg,
                              ))


    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets

    def calculate_loss(self, xx_inputs, begin, end, adaptive_lamda,activation, cal_dist=False, T=2):
        inputs_1, inputs_2, targets = self._parse_data(xx_inputs)
        # input forward
        f_out_t1, f_out_t1_m, logit_t1, logit_t1_m = self.model_cur(inputs_1)
        p_out_t1 = logit_t1[:, begin:end]
        p_out_t1_m = logit_t1_m[:, begin:end]
        f_out_t1_ema, f_out_t1_ema_m, p_out_t1_ema, p_out_t1_ema_m = self.model_ema(inputs_1)
        p_out_t1_ema = p_out_t1_ema[:, begin:end]
        p_out_t1_ema_m = p_out_t1_ema_m[:, begin:end]
        loss_ce = (activation(p_out_t1, targets - begin) +
                   activation(p_out_t1_m, targets - begin)) / 2
        loss_ce_soft = (self.criterion_ce_soft(p_out_t1, p_out_t1_ema_m) +
                        self.criterion_ce_soft(p_out_t1_m, p_out_t1_ema)) / 2
        loss_tri_soft = (self.criterion_tri_soft(f_out_t1, f_out_t1_ema_m, targets) + self.criterion_tri_soft(
            f_out_t1_m, f_out_t1_ema, targets)) / 2



        if cal_dist:
            logit_t1_old = logit_t1[:, :self.old_class_num]
            logit_t1_m_old = logit_t1_m[:, :self.old_class_num]
            with torch.no_grad():
                f_out_ref, f_out_ref_m, p_out_ref, p_out_ref_m = self.model_ref(inputs_1)
            # ref_logit = (p_out_ref + p_out_ref_m) / 2
            # logit_t1_old = (logit_t1[:, :self.old_class_num] + logit_t1_m[:, :self.old_class_num]) / 2
            # loss_dist = nn.KLDivLoss()(F.log_softmax(logit_t1_old / T, dim=1), F.softmax(ref_logit.detach() / T, dim=1))
            # loss_lf_tri = (self.criterion_tri_soft(f_out_t1, f_out_ref_m.detach(), targets) +
            #                self.criterion_tri_soft(f_out_t1_m, f_out_ref.detach(), targets)) / 2
            loss_dist = (self.criterion_ce_soft(logit_t1_old, p_out_ref.detach()) + self.criterion_ce_soft(logit_t1_m_old,p_out_ref_m.detach())) / 2
            loss_lf_tri = (self.criterion_tri_soft(f_out_t1, f_out_ref.detach(), targets) +
                           self.criterion_tri_soft(f_out_t1_m, f_out_ref_m.detach(), targets)) / 2
        # if phase > 0:
        #     logit_t1_old = p_out_t1[:, :self.old_class_num]
        #     logit_t1_m_old = p_out_t1_m[:, :self.old_class_num]
        #     with torch.no_grad():
        #         f_out_ref, f_out_ref_m, p_out_ref, p_out_ref_m = self.model_ref(inputs_1)
        #
        #     loss_lf_logit = (self.criterion_ce_soft(logit_t1_old, p_out_ref) + self.criterion_ce_soft(logit_t1_m_old,
        #                                                                                               p_out_ref_m)) / 2
        #     loss_lf_tri = (self.criterion_tri_soft(f_out_t1, f_out_ref_m, targets) +
        #                    self.criterion_tri_soft(f_out_t1_m, f_out_ref, targets)) / 2
        #


        else:
            loss_dist = 0
            loss_lf_tri=0
        loss_target_input = loss_ce * (1 - 0.5) + loss_ce_soft * 0.5 + loss_tri_soft + (loss_dist + loss_lf_tri)*adaptive_lamda*0.1  # * adaptive_lamda未使用自适应系数
        # ce_soft_weight = 0.5
        precision = accuracy(p_out_t1.data, (targets - begin).data)
        return loss_target_input, loss_ce, loss_ce_soft, loss_tri_soft, loss_dist, loss_lf_tri,precision

