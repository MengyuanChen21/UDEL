import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import torch.nn.init as torch_init
from uncertainty import obtain_aleatoric_uct
from edl_loss import EvidenceLoss
from edl_loss import relu_evidence, exp_evidence, softplus_evidence

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat


# fusion split modal single+ bit_wise_atten dropout+ contrastive + mutual learning +fusion feat(cat)
# ------TOP!!!!!!!!!!
class CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio

        self.origin_ratio = args['opt'].origin_ratio

        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        # intra-video temporal relation modeling layers
        intermediate_channel = 512
        self.intra_temp_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=(1,), stride=(1,), padding=0)
        self.intra_temp_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=(1,), stride=(1,), padding=0)
        self.intra_temp_conv2_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=(3,), stride=(1,), padding=1)
        self.intra_temp_conv2_v = nn.Conv1d(n_feature, n_feature, kernel_size=(3,), stride=(1,), padding=1)

        # relu layer
        self.opt = nn.ReLU()
        # intra classifier layer
        self.intra_temp_classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class+1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class + 1, 1)
        )

        self.kl_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)

        self.apply(weights_init)

    def intra_weight(self, temp_feat):
        # temp_feat: [n, d, t]
        temp_feat = temp_feat.transpose(-1, -2)                             # [n, t, d]
        cos = lambda m: F.normalize(m) @ F.normalize(m).t()
        cos_similarity = torch.stack([cos(m) for m in temp_feat])           # [n, t, t]
        cos_similarity = F.softmax(cos_similarity, dim=1)                   # [n, t, t]  在1维上进行softmax
        return cos_similarity

    def weighted_sum(self, temp_feat, cos_sim):
        # temp_feat: (n, d, t)      cos_sim: (n, t, t)
        weighted_feat = torch.bmm(temp_feat, cos_sim)                       # (n, d, t)
        weighted_feat = weighted_feat + temp_feat                           # (n, d, t)
        return weighted_feat

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        # Tensor:(10, 2048, 320)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)

        # intra-video temporal relation modeling
        layer1_key = self.intra_temp_conv1_k(nfeat)
        layer1_value = self.intra_temp_conv1_v(nfeat)
        layer1_sim_weight = self.intra_weight(layer1_key)
        layer1_feat = self.weighted_sum(layer1_value, layer1_sim_weight)
        layer1_feat = self.opt(layer1_feat)

        layer2_key = self.intra_temp_conv2_k(layer1_feat)
        layer2_value = self.intra_temp_conv2_v(layer1_feat)
        layer2_sim_weight = self.intra_weight(layer2_key)
        layer2_feat = self.weighted_sum(layer2_value, layer2_sim_weight)
        layer2_feat = self.opt(layer2_feat)

        intra_cls = self.intra_temp_classifier(layer2_feat)

        x_cls = self.classifier(nfeat)

        x_cls_fusion = self.origin_ratio * x_cls + (1 - self.origin_ratio) * intra_cls

        outputs = {'feat': nfeat.transpose(-1, -2),  # (10, 320, 2048)
                   'cas': x_cls_fusion.transpose(-1, -2),
                   'cas_origin': x_cls.transpose(-1, -2),  # (10, 320, 21)
                   'intra_cas': intra_cls.transpose(-1, -2),
                   'attn': x_atn.transpose(-1, -2),  # (10, 320, 1)
                   'v_atn': v_atn.transpose(-1, -2),  # (10, 320, 1)
                   'f_atn': f_atn.transpose(-1, -2),  # (10, 320, 1)
                   }

        return outputs

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        intra_cas = outputs['intra_cas']  # (B, 500, 21)
        origin_cas = outputs['cas_origin']  # (B, 500, 21)
        loss_intra_cls, _ = self.topkloss(intra_cas, labels, is_back=True, rat=args['opt'].k)
        loss_origin_cls, _ = self.topkloss(origin_cas, labels, is_back=True, rat=args['opt'].k)

        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())

        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             labels,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             mutual_weight=args['opt'].mutual_weight
                                             )

        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k)

        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        total_loss = (
                    args['opt'].alpha_edl * edl_loss +
                    args['opt'].alpha_uct_guide * uct_guide_loss +
                    loss_mil_orig.mean() + loss_mil_supp.mean() + loss_intra_cls.mean() + loss_origin_cls.mean() +
                    args['opt'].alpha3 * loss_3_supp_Contrastive +
                    args['opt'].alpha4 * mutual_loss +
                    args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                    args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_intra_cls': loss_intra_cls.mean(),
            'loss_origin_cls': loss_origin_cls.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       labels,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude,
                       mutual_weight):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        alpha = alpha[..., :-1]

        epistemtic_snippet_uct = n_class / torch.sum(alpha, dim=-1)

        b, t, c = alpha.shape
        coarse_t = 8
        scale = int(t / coarse_t)
        coarse_alpha = alpha.reshape((b, scale, coarse_t, c)).mean(dim=1)
        coarse_snippet_uct = obtain_aleatoric_uct(
            alpha=coarse_alpha.reshape((b * coarse_t, c)),
            target=torch.repeat_interleave(labels, repeats=coarse_t, dim=0),
            blur=0.1,
            scaling=0.4,
            n_sample=10)\
            .reshape((b, coarse_t))

        curve = self.course_function(epoch, total_epoch, coarse_t, amplitude)

        loss_guide = self.mutual_loss(1 - element_atn, epistemtic_snippet_uct.unsqueeze(-1), mutual_weight)
        v_loss_guide = self.mutual_loss(1 - v_atn, epistemtic_snippet_uct.unsqueeze(-1), mutual_weight)
        f_loss_guide = self.mutual_loss(1 - f_atn, epistemtic_snippet_uct.unsqueeze(-1), mutual_weight)

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(coarse_snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)
        fine_sorted_curve = torch.repeat_interleave(sorted_curve, repeats=scale, dim=1)

        uct_guide_loss = torch.mul(fine_sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def mutual_loss(self, p, q, weight=0.5):
        return weight * F.mse_loss(p, q.detach()) + (1 - weight) * F.mse_loss(p.detach(), q)

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )

        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]

        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)

        # From -1 to 1
        theta = 2 * (idx + 0.5) / total_snippet_num - 1

        # From 1 to -1
        delta = - 2 * epoch / total_epoch + 1

        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 rat=8):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(topk_val, dim=-2)

        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = - (labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)

        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn
