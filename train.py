import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, seed_worker
from datasets import get_dataset, HyperX
import os
import time
import numpy as np
import pandas as pd
import argparse
from con_losses import SupConLoss
from network import Generator
from network import Discriminator
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch D3net')
parser.add_argument('--save_path', type=str, default='./Results')

parser.add_argument('--data_path', type=str, default='./datasets/Pavia/')
parser.add_argument('--source_name', type=str, default='paviaU')
parser.add_argument('--target_name', type=str, default='paviaC')

# parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
# parser.add_argument('--source_name', type=str, default='Houston13')
# parser.add_argument('--target_name', type=str, default='Houston18')

# parser.add_argument('--data_path', type=str, default='./datasets/Indiana/')
# parser.add_argument('--source_name', type=str, default='IndianaS')
# parser.add_argument('--target_name', type=str, default='IndianaT')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0, help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4, help='the L2  weight decay')
parser.add_argument('--training_sample_ratio', type=float, default=0.5, )
parser.add_argument('--re_ratio', type=int, default=5)  # 数据增强倍数
parser.add_argument('--lambda_1', type=float, default=1.0)  # 用于计算损失的权重参数，默认为1.0。
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lr_scheduler', type=str, default='none')  # --lr_scheduler: 学习率调度器的类型，默认为'none'。
parser.add_argument('--log_interval', type=int, default=40)

parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--d_se', type=int, default=64)  # 生成器网络中的通道数，默认为64。
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--n_blocks', type=int, default=4)
parser.add_argument('--Omega', type=int, default=0.9)  # 分类贡献加权参数

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13)
group_train.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256)
group_train.add_argument('--pro_dim', type=int, default=128)  # 判别器的输出通道数，默认为128。
group_train.add_argument('--test_stride', type=int, default=1)  # 推断时的滑动窗口步幅，默认为1

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,  # 是否进行随机翻转数据增强，默认为True。
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,  # 是否进行随机辐射噪声（光照）数据增强
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,  # 是否进行光谱混合数据增强
                      help="Random mixes between spectra")
args = parser.parse_args()


def evaluate(net, val_loader, gpu, tgt=False):
    """
        评估神经网络在验证集上的性能。

        参数:
        - net: 待评估的神经网络模型。
        - val_loader: 提供验证数据批次的验证 DataLoader。
        - gpu: 指定计算应在哪个 GPU 上执行。
        - tgt: 一个布尔标志，表示是否打印额外的分类指标（默认为 False）。

        返回:
        - acc: 模型在验证集上的准确率（百分比）。
    """

    # 初始化用于存储预测和真实标签的列表
    ps = []  # 存储预测标签
    ys = []  # 存储真实标签

    # 遍历验证数据加载器的每个批次
    for i, (x1, y1) in enumerate(val_loader):
        # 将标签减去1，可能是为了处理从1开始的标签
        y1 = y1 - 1

        # 禁用梯度计算，因为在评估阶段不需要进行梯度更新
        with torch.no_grad():
            # 将输入数据移到GPU上
            x1 = x1.to(gpu)

            # 使用神经网络进行推理，获取预测的类别
            p1 = net(x1)
            p1 = p1.argmax(dim=1)

            # 将预测标签和真实标签附加到列表中
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())

    # 将列表中的预测标签和真实标签连接成一个数组
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)

    # 计算准确率
    acc = np.mean(ys == ps) * 100

    # 如果tgt为True，打印额外的分类指标
    if tgt:
        # 调用metrics函数计算额外的分类指标
        results = metrics(ps, ys.astype(int), n_classes=int(ys.max() + 1))

        print(results['Confusion_matrix'], '\n', 'TPR:\n', np.round(results['TPR'] * 100, 2), '\n OA:',
              results['Accuracy'], 'AA:', sum(np.round(results['TPR'] * 100, 2)) / len(results['TPR']), 'Kappa:',
              results['Kappa'])
    # 返回准确率
    return acc
    # , results

def evaluate_tgt(cls_net, gpu, loader, modelpath):
    """
        评估预训练的分类器模型在目标集上的性能。

        参数:
        - cls_net: 待评估的预训练分类器模型。
        - gpu: 指定计算应在哪个 GPU 上执行。
        - loader: 提供验证数据批次的 DataLoader。
        - modelpath: 预训练模型的权重路径。

    返回:
        - teacc: 模型在验证集上的准确率（百分比）。
    """
    # 从指定路径加载预训练的模型权重
    saved_weight = torch.load(modelpath)

    # 将预训练的权重加载到分类器模型cls_net中
    cls_net.load_state_dict(saved_weight['Discriminator'])

    # 将模型设置为评估模式（不进行梯度更新）
    cls_net.eval()

    # 调用 evaluate 函数计算在验证集上的性能指标，设置 tgt 参数为 True 表示打印额外的分类指标
    teacc = evaluate(cls_net, loader, gpu, tgt=True)

    # 返回在验证集上的准确率
    return teacc


def experiment():
    # 打印超参数
    hyperparams = vars(args)
    print(hyperparams)

    # 获取当前时间并构建日志目录
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)

    # 创建保存目录
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 将超参数保存到params.txt文件中
    df = pd.DataFrame([args], columns=['*params*'])
    df.to_csv(os.path.join(log_dir, 'params.txt'), index=False)

    # 设置随机数生成器的种子
    seed_worker(args.seed)

    # 加载源域和目标域的数据集
    img_src, gt_src, LABEL_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name, args.data_path)
    img_tar, gt_tar, LABEL_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name, args.data_path)

    # 计算样本数量和扩增倍数
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar  #

    # 获取图像维度和类别数
    num_classes = int(gt_src.max())
    N_BANDS = img_src.shape[-1]

    # 更新超参数字典
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    # 对图像进行零填充，以适应模型的感受野
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    # 随机划分源域的训练和验证集
    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')

    # 随机划分目标域的测试集
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')

    # 构建源域和目标域的训练、验证和测试数据集
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src

    # 如果样本不足，通过复制源域数据增强
    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    # 构建训练、验证和测试数据加载器
    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,  # 如果设置为 True，数据加载器会将加载的数据存储在 CUDA 固定内存上，以加速 GPU 数据传输。
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True, )

    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'])

    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'])

    # 设置图像尺寸
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    def vit_base_patch1_13(num_classes: int = 7):
        model = Discriminator.VisionTransformer(img_size=args.patch_size,
                                                patch_size=1,
                                                in_c=args.n_bands,
                                                embed_dim=args.embed_dim,
                                                depth=args.depth,
                                                num_heads=args.num_heads,
                                                representation_size=None,
                                                num_classes=num_classes,
                                                pro_dim=args.pro_dim,
                                                ou=args.Omega, )
        return model

    # 构建判别器和生成器模型
    D_net = vit_base_patch1_13().to('cuda')
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr)

    G_net = Generator.Generator(d_se=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, n_blocks=args.n_blocks,
                                device=args.gpu).to(args.gpu)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)

    # 定义损失函数
    cls_criterion = nn.CrossEntropyLoss()  # 分类损失
    con_criterion = SupConLoss(device=args.gpu)  # 对比损失

    # 初始化最佳准确率
    best_acc = 0
    taracc, taracc_list = 0, []
    jiange = 0

    # 主训练循环
    for epoch in range(1, args.max_epoch + 1):
        t1 = time.time()
        loss_list = []
        D_net.train()
        jiange += 1

        # 遍历训练集
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1

            # 生成对抗样本
            with torch.no_grad():
                x_ED = G_net(x)

            x_tgt = G_net(x)

            # 获取判别器的输出和中间特征，p_SD为分类头，z_SD为投影头
            p_SD, z_SD = D_net(x, mode='train')
            p_ED, z_ED = D_net(x_ED, mode='train')
            z_SDcatED = torch.cat([z_SD.unsqueeze(1), z_ED.unsqueeze(1)], dim=1)

            # 计算源域分类损失
            src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(p_ED, y.long())
            # 计算目标域分类损失
            p_tgt, z_tgt = D_net(x_tgt, mode='train')
            tgt_cls_loss = cls_criterion(p_tgt, y.long())

            # 计算对比损失
            zall = torch.cat([z_tgt.unsqueeze(1), z_SDcatED], dim=1)
            con_loss = con_criterion(zall, y, adv=False)

            # 计算总体损失
            loss = src_cls_loss + args.lambda_1 * con_loss + tgt_cls_loss

            # 判别器参数梯度清零
            D_opt.zero_grad()

            # 反向传播
            loss.backward(retain_graph=True)

            # 计算对抗损失
            zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1).detach()], dim=1)
            con_loss_adv = 0
            idx_1 = np.random.randint(0, z_SDcatED.size(1))

            for i, id in enumerate(y.unique()):
                mask = y == y.unique()[i]
                z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
                y_i = torch.cat([torch.zeros(z_SD_i.shape[0]), torch.ones(z_SD_i.shape[0])])
                zall = torch.cat([z_SD_i.unsqueeze(1).detach(), zsrc_i[:, idx_1:idx_1 + 1]], dim=0)

                # 计算对抗损失
                if y_i.size()[0] > 2:
                    con_loss_adv += con_criterion(zall, y_i, adv=True)
            con_loss_adv = con_loss_adv / y.unique().shape[0]

            # 计算生成器的损失
            loss = tgt_cls_loss + args.lambda_2 * con_loss_adv
            G_opt.zero_grad()
            loss.backward()

            # 更新判别器和生成器的参数
            D_opt.step()
            G_opt.step()

            # 将当前批次的损失记录下来
            # 源领域分类损失（src_cls_loss）、目标领域分类损失（tgt_cls_loss）、对比损失（con_loss）和对抗性对比损失（con_loss_adv）
            loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item()])

        # 计算平均损失
        src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv = np.mean(loss_list, 0)

        # 在测试集上评估判别器性能
        D_net.eval()
        teacc = evaluate(D_net, val_loader, args.gpu)

        # 如果在测试集上获得了更好的准确率，则保存模型权重
        if best_acc <= teacc:
            best_acc = teacc
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pkl'))

            # t2 = time.time()
            # print(
            #     f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f} ||| val {len(val_loader.dataset)}, teacc {teacc:2.2f}')

            # pklpath = f'{log_dir}/best.pkl'
            # taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath)
            # taracc_list.append(round(taracc, 2))
            # print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')
            # continue

        t2 = time.time()
        # 打印当前训练和验证的一些指标
        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f} ||| val {len(val_loader.dataset)}, teacc {teacc:2.2f}')
        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath)
            taracc_list.append(round(taracc, 2))

            existing_df = pd.read_csv(os.path.join(log_dir, 'params.txt'))
            existing_df['*max OA*'] = max(taracc_list)
            existing_df.to_csv(os.path.join(log_dir, 'params.txt'), index=False)

            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')


if __name__ == '__main__':
    experiment()
