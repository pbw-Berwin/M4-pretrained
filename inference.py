# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from ops.dataset import VideoDataSetOnTheFly
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

best_prec1 = 0
best_mAP = 0
bceCriterion = torch.nn.BCEWithLogitsLoss(reduce=False).cuda()

warp_w = {}
sum = 0
for i in range(313):
  sum += 1./(i+1)
  warp_w[i] = sum

def warp_r(ranks):
  for i in range(ranks.size(0)):
    ranks[i] = warp_w[ranks[i].item()]
  return ranks

def warp(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1))).add(1)
  if weights is not None:
    return (diffs.clamp(0,1e10).mul(mask).sum(1).mul(weights).masked_select(labels.bool())
                 .mul(warp_r(scores.sort(descending=True)[1].masked_select(labels.bool()).float()))
                 .mean())
  else:
    return (diffs.clamp(0,1e10).mul(mask).sum(1).masked_select(labels.bool())
                 .mul(warp_r(scores.sort(descending=True)[1].masked_select(labels.bool()).float()))
                 .mean())

def bce(output, target, weights=None):
  if weights is not None:
    return (((1.-weights)*target + weights*(1.-target))*
             bceCriterion(output, torch.autograd.Variable(target))).sum(1).mean()
  else:
    return bceCriterion(output, torch.autograd.Variable(target)).sum(1).mean()

def bp_mll(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
  if weights is not None:
    return diffs.exp().mul(mask).sum(1).mul(weights).sum(1).mean()
  else:
    return diffs.exp().mul(mask).sum(1).sum(1).mean()

def lsep(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
  if weights is not None:
    return diffs.exp().mul(mask).sum(1).add_(1).log().mul(weights).masked_select(labels.bool()).mean()
  else:
    return diffs.exp().mul(mask).sum(1).add_(1).log().masked_select(labels.bool()).mean()

def lsep_orig(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
  return diffs.exp().mul(mask).sum().add(1).log().mean()

def multi_softmax(scores, labels, weights=None):
  if weights is not None:
    return -torch.nn.functional.log_softmax(scores,1).mul(labels).mul(weights).mean()
  else:
    return -torch.nn.functional.log_softmax(scores,1).mul(labels).mean()

def main():
    global args, best_mAP
    args = parser.parse_args()
    
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.loss_type:
        args.store_name += '_{}'.format(args.loss_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()
    
    val_loader = torch.utils.data.DataLoader(
        VideoDataSetOnTheFly(args.root_path, args.val_list, num_groups=args.num_segments,
                   is_train=False,
                   metadir=args.metadir,
                   num_classes=num_class,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sampling=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss(weight=None).cuda()
    elif args.loss_type == 'lsep':
        criterion = lsep
    elif args.loss_type == 'bce':
        criterion = bce
    elif args.loss_type == 'bp_mll':
        criterion = bp_mll
    elif args.loss_type == 'multi_softmax':
        criterion = multi_softmax
    elif args.loss_type == 'warp':
        criterion = warp
    elif args.loss_type == 'lsep_orig':
        criterion = lsep_orig
    else:
        raise ValueError("Unknown loss type")

    validate(val_loader, model, criterion, 0)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    meter = performanceMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            meter.add(output.data, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                scores = meter.value()
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1:.3f}\t'
                          'Prec@5 {top5:.3f}\t'
                          'mAP {mAP:.3f}'.format(
                              i,
                              len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              top1=scores[0],
                              top5=scores[1],
                              mAP=scores[2]))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
    scores = meter.value()
    output = (
        'Testing Results: Prec@1 {top1:.3f} Prec@5 {top5:.3f} mAP {mAP:.3f} Loss {loss.avg:.5f}'
        .format(top1=scores[0], top5=scores[1], mAP=scores[2], loss=losses))
    print(output)
    output_best = '\nBest mAP: %.3f' % (best_mAP)
    print(output_best)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', scores[0], epoch)
        tf_writer.add_scalar('acc/test_top5', scores[1], epoch)
        tf_writer.add_scalar('acc/test_mAP', scores[2], epoch)

    return scores[2]


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (np.sum((epoch >= np.array(lr_steps)).astype('int8')))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

class performanceMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.top1 = 0
        self.top5 = 0
        self.ap = 0
        self.count = 0

    def add(self, output, target):
      assert output.size(0) == target.size(0)
      assert output.size(1) == target.size(1)
      _,order = output.sort(descending=True)
      for i in range(order.size(0)):
        a = 0
        tp = 0
        fp = 0
        pos = target[i].nonzero()
        if pos.size(0) > 0:
          pos = pos.reshape(pos.size(0))
          sum = pos.size(0)
          indPrev = 0
          indices=torch.FloatTensor(pos.size(0))
          for j in range(pos.size(0)):
            indices[j] = (order[i] == pos[j]).nonzero().squeeze().item()
          indices,_ = indices.sort()
          if (indices < 1).any():
            self.top1 += 1
            self.top5 += 1
          elif (indices < 5).any():
            self.top5 += 1
          for j in range(pos.size(0)):
            ind = indices[j]
            fp += (ind - indPrev)
            tp += 1
            a += tp/(tp+fp)
            indPrev = ind+1
          self.count += 1
          self.ap += a.item()/sum

    def value(self):
        return (self.top1/max(1, self.count), self.top5/max(1, self.count), self.ap/max(1, self.count))


if __name__ == '__main__':
    main()
