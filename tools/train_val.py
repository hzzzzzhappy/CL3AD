import argparse
import logging
import os
import pprint
import shutil
import time
import warnings
import torch
import torch.distributed as dist
import torch.optim
import yaml
import copy
import torch.nn.functional as F
from datasets.data_builder import build_dataloader
from easydict import EasyDict
import torch.multiprocessing as mp
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion,build_svd_loss
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from functools import partial
from utils.ConstrainedSGD import ConstrainedSGD
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")

def update_ema_variables(model, ema_model, alpha, global_step, active_layers):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    alpha = alpha
    with torch.no_grad():
        for layer in active_layers:
            module = getattr(model.module, layer)
            model_state_dict = module.state_dict()
            ema_module = getattr(ema_model.module,layer)
            ema_model_state_dict = ema_module.state_dict()

            try:
                for entry in model_state_dict.keys():
                    ema_param = ema_model_state_dict[entry].clone().detach()
                    param = model_state_dict[entry].clone().detach()

                    # Interpolation
                    new_param = (ema_param * alpha) + (param * (1. - alpha))

                    model_state_dict[entry] = new_param
            except:
                continue

            module.load_state_dict(model_state_dict)

def normalize_perturbation(p_list, max_norm):
    """
    一个辅助函数 用于归一化扰动列表 正确处理1D和ND张量。
    """
    for p in p_list:
        if p.dim() >= 2:
            p.data.renorm_(p=2, dim=0, maxnorm=max_norm)
        else:
            # 对于1D张量（如偏置），将整个张量视为一个向量并归一化
            norm = torch.norm(p.data)
            if norm > max_norm:
                p.data.mul_(max_norm / (norm + 1e-6))

def get_adversarial_kl_loss(model, input_data, original_outputs, config):
    """
    Calculates the adversarial KL divergence loss by finding the worst-case perturbation.
    This implements the inner maximization loop of the min-max problem.
    """
    # 1. Detach original outputs from the graph
    original_logits = original_outputs.get('pred')
    if original_logits is None:
        print("Warning: 'pred' key not found in model output for KL loss calculation. Returning 0.")
        return torch.tensor(0.0, device=model.device)
    original_logits = original_logits.detach()

    # 2. Extract hyperparameters
    epsilon = config.trainer.adversarial_epsilon
    xi = config.trainer.adversarial_xi
    num_steps = config.trainer.adversarial_steps

    # 3. 创建扰动 `delta` 并初始化
    trainable_params_orig = [p for p in model.parameters() if p.requires_grad]
    
    # 初始化 delta 为 L2 球内的一个小随机向量
    delta = [torch.randn_like(p) for p in trainable_params_orig]
    normalize_perturbation(delta, xi)
    
    perturbed_model = copy.deepcopy(model)
    perturbed_model.eval() # 始终在评估模式下计算扰动
    trainable_params_pert = [p for p in perturbed_model.parameters() if p.requires_grad]

    # 4. 执行 K 步投影梯度上升（PGD）来寻找最差情况的 delta
    for _ in range(num_steps):
        for p_pert, p_orig, d in zip(trainable_params_pert, trainable_params_orig, delta):
            p_pert.data = p_orig.detach().data + d.data
        
        # 梯度清零，为本次迭代做准备
        perturbed_model.zero_grad()
        
        # 使用扰动模型进行前向传播
        perturbed_outputs = perturbed_model(input_data)
        perturbed_logits = perturbed_outputs.get('pred')
        
        # 我们要最大化 MSE，等价于最小化它的相反数
        l_mse = F.mse_loss(perturbed_logits, original_logits)
        loss_for_ascent = -l_mse # 或者直接 kl_div.backward() 然后在更新时用减法，但这样更直观

        loss_for_ascent.backward()
        
        # 【关键修改 3】: 使用 p_pert.grad 来更新 delta
        for i, p_pert in enumerate(trainable_params_pert):
            g = p_pert.grad # 获取 KL 散度关于 p_pert 的梯度
            d = delta[i]
            if g is not None:
                d.data.add_(epsilon * g / (torch.norm(g) + 1e-6))
        
        # 将 delta 投影回 L2 球内
        normalize_perturbation(delta, epsilon)

    # 用最终找到的 delta 计算最终的 MSE 损失
    with torch.no_grad():
        for p_pert, p_orig, d in zip(trainable_params_pert, trainable_params_orig, delta):
            p_pert.data = p_orig.data + d.data
        
        # perturbed_outputs_final = perturbed_model(input_data)
        # perturbed_logits_final = perturbed_outputs_final.get('pred')
        
        # log_p_original_final = F.log_softmax(original_logits, dim=1)
        # p_perturbed_final = F.softmax(perturbed_logits_final, dim=1)
        
        # final_kl_loss = F.kl_div(log_p_original_final, p_perturbed_final, reduction='batchmean')
        perturbed_outputs_final = perturbed_model(input_data)
        perturbed_pred_final = perturbed_outputs_final.get('pred')
        
        final_adv_loss = F.mse_loss(perturbed_pred_final, original_logits)

    del perturbed_model # 释放内存
    
    return final_adv_loss


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)
    # config = update_config(config)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    model_old = ModelHelper(config.net)
    model_old.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model_old = DDP(
        model_old,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    cls_parameters = [
        {"params": getattr(model.module, layer).get_finetune_params()} for layer in active_layers
    ]
    parameters = [
        {"params": getattr(model.module, layer).get_base_params()} for layer in active_layers
    ]
    layer = active_layers[0]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    
    constrainedSGD = ConstrainedSGD(getattr(model.module, layer).base_named_parameters(), lr=1)    ###############################
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0
    skip = True

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    if(config.AD_training):
        train_loaders, val_loaders = build_dataloader(config.dataset, distributed=False)
        SVDLoss = build_svd_loss()

        if args.evaluate:
            validate(train_loaders[-1],val_loaders[-1], model,0,ori_feature=torch.load(load_path)["feature_metrix"])
            return
        total_epoch = 0
        criterion = build_criterion(config.criterion)
        for i,(train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
            logger = logging.getLogger("global_logger")
            logger.info(
                "Training on Task {} with {} samples".format(
                    i + 1, len(train_loader)
                )
            )
            best_metric = 0
            if(i!=0):
                load_state(load_path,model_old)
                skip = False
                optimizer = constrainedSGD
                lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
            for epoch in range(last_epoch, config.trainer.max_epoch):
                total_epoch += 1
                last_iter = epoch * len(train_loader)
                train_one_epoch(
                    train_loader,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    last_iter,
                    tb_logger,
                    criterion,
                    frozen_layers,
                    model_old,
                    SVDLoss,
                    active_layers,
                    skip
                )
                lr_scheduler.step(epoch)

                if (epoch + 1) % config.trainer.val_freq_epoch == 0:
                    logger.info(
                        "Testing on Task {} with {} samples".format(
                            i + 1, len(val_loader)
                        )
                    )
                    if(i==0):
                        ret_metrics,outputs_dict = validate(train_loader,val_loader, model,total_epoch,ori_feature={})
                    else:
                        ret_metrics,outputs_dict = validate(train_loader,val_loader, model,total_epoch,ori_feature=torch.load(load_path)["feature_metrix"])
                    # only ret_metrics on rank0 is not empty
                    if rank == 0:
                        ret_key_metric = ret_metrics[key_metric]
                        is_best = ret_key_metric >= best_metric
                        best_metric = max(ret_key_metric, best_metric)
                        save_checkpoint(
                            {
                                "epoch": epoch + 1,
                                "arch": config.net,
                                "state_dict": model.state_dict(),
                                "best_metric": best_metric,
                                "optimizer": optimizer.state_dict(),
                                "feature_metrix": outputs_dict,
                            },
                            is_best,
                            config,
                        )
    # =================================== cls =================================== 
    if(config.cls_training):
        logger.info(
                "Training classfication task!"
            )
        train_loader,val_loader = build_dataloader(config.cls_dataset, distributed=False,is_cls=True)
        criterion = build_criterion(config.cls_criterion)
        cls_optimizer = get_optimizer(cls_parameters, config.trainer.optimizer)
        optimizer = cls_optimizer
        lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
        for epoch in range(last_epoch, config.trainer.max_epoch*2):
            total_epoch = 0
            last_iter = epoch * len(train_loader)
            train_cls_one_epoch(
                train_loader,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                last_iter,
                tb_logger,
                criterion,
                frozen_layers,
            )
            lr_scheduler.step(epoch)
            if (epoch + 1) % config.trainer.val_freq_epoch == 0:
                ret_metrics,outputs_dict = validate(train_loader,val_loader, model,total_epoch,ori_feature={})
                # only ret_metrics on rank0 is not empty
                if rank == 0:
                    ret_key_metric = ret_metrics[key_metric]
                    is_best = ret_key_metric >= best_metric
                    best_metric = max(ret_key_metric, best_metric)
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "arch": config.net,
                            "state_dict": model.state_dict(),
                            "best_metric": best_metric,
                            "optimizer": optimizer.state_dict(),
                            "feature_metrix": outputs_dict,
                        },
                        is_best,
                        config,
                    )

def train_cls_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
):
    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)
        # if skip == False:
        #     update_ema_variables(model, model_old, 0.1, curr_step,active_layers)

        # forward
        outputs = model(input)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        reduced_loss = loss.clone()

        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch*2,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
    model_old,
    SVDLoss,
    active_layers,
    skip
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)
    losses_svd = AverageMeter(config.trainer.print_freq_step)
    losses_kl = AverageMeter(config.trainer.print_freq_step)

    model.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()
    # concatenated_tensor_0 = torch.empty(0).cuda()
    # concatenated_tensor_1 = torch.empty(0).cuda()
    # concatenated_tensor_2 = torch.empty(0).cuda()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)
        if skip == False:
            update_ema_variables(model, model_old, 0.1, curr_step, active_layers)

        # forward
        outputs = model(input)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        
        loss_kl = torch.tensor(0.0, device=loss.device) # 初始化为0
        if config.trainer.get('use_kl_loss', False): # 检查是否启用KL损失
            # 调用辅助函数计算KL损失
            loss_kl = get_adversarial_kl_loss(model, input, outputs, config)
            # 将KL损失按权重添加到总损失中
            loss += config.trainer.kl_loss_weight * loss_kl
        
        # concatenated_tensor_0 = torch.cat([concatenated_tensor_0, outputs["middle_decoder_feature_0"].clone().detach()], dim=0)
        # concatenated_tensor_1 = torch.cat([concatenated_tensor_1, outputs["middle_decoder_feature_1"].clone().detach()], dim=0)
        # concatenated_tensor_2 = torch.cat([concatenated_tensor_2, outputs["middle_decoder_feature_2"].clone().detach()], dim=0)

        # if concatenated_tensor_0.shape[0] > 768:
        #     concatenated_tensor_0 = concatenated_tensor_0[48:]
        
        # if concatenated_tensor_1.shape[0] > 768:
        #     concatenated_tensor_1 = concatenated_tensor_1[48:]
        
        # if concatenated_tensor_2.shape[0] > 768:
        #     concatenated_tensor_2 = concatenated_tensor_2[48:]

        loss_svd = 0
        # loss_svd = SVDLoss(concatenated_tensor_0,concatenated_tensor_1,concatenated_tensor_2)
        # if skip == True:
        #     loss += 10*loss_svd

        reduced_loss = loss.clone()
        # reduced_loss_svd = loss_svd.clone().detach()
        reduced_loss_kl = loss_kl.clone().detach()
        loss += reduced_loss_kl

        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())
        # losses_svd.update(reduced_loss_svd.item())
        losses_kl.update(reduced_loss_kl.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            # tb_logger.add_scalar("svd_loss", losses_svd.avg, curr_step + 1)
            tb_logger.add_scalar("kl_loss", losses_kl.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                # "Loss_SVD {loss_svd.val:.5f} ({loss_svd.avg:.5f})\t"
                "Loss_KL {loss_kl.val:.5f} ({loss_kl.avg:.5f})\t" # 新增
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    # loss_svd=losses_svd,
                    loss_kl=losses_kl, # 新增
                    lr=current_lr,
                )
            )

        end = time.time()



def hook_fn(module, input, output, outputs_dict, name):
    # 保存每一层的输出
    module_name = str(name)
    if torch.is_tensor(output):
        device = torch.device("cuda:0")
        output = output.mean(dim=0).to(device) 
        if module_name in outputs_dict:
            outputs_dict[module_name] = torch.cat(
                [outputs_dict[module_name], output.detach()], dim=0)
        else:
            outputs_dict[module_name] = output.detach()

def feature_metrix_cal(outputs_dict):
    feature_metrix_dict={}
    for k,v in outputs_dict.items():
        U, S, Vh = torch.linalg.svd(
            v, full_matrices=False)
        feature_metrix_dict[k] = Vh
    return feature_metrix_dict

def validate(train_val_loader,val_loader, model,total_epoch,ori_feature):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()
    outputs_dict = {}
    merged_dict = ori_feature

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    dist.barrier()
    if total_epoch % 2000 == 0:    
        for name, layer in model.named_modules():
            if ("transformer" in name and "decode" in name and "linear2" in name) or ("transformer" in name and "encoder" in name and "linear2" in name):
                # print("LAYER:", name)
                handle = layer.register_forward_hook(
                    partial(hook_fn, outputs_dict=outputs_dict, name=name))
        with torch.no_grad():
            for i, input in enumerate(train_val_loader):
                # forward
                logger.info(i)
                outputs = model(input)

            if ori_feature:
                merged_dict = {k: torch.cat([ori_feature[k], outputs_dict[k]], dim=0) for k in outputs_dict.copy()}
            else:
                merged_dict = outputs_dict.copy()
            
            handle.remove()
    
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, labels,image_max_score,masks,pred,cls_label,points,cls_pred = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(fileinfos, labels,image_max_score,masks,pred,cls_label,cls_pred)
        log_metrics(ret_metrics, config.evaluator.metrics)
        if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                pred,
                masks,
                labels,
                points,
                config.evaluator.vis_compound,
            )
        # if args.evaluate and config.evaluator.get("vis_single", None):
        #     visualize_single(
        #         fileinfos,
        #         preds,
        #         config.evaluator.vis_single,
        #         config.dataset.image_reader,
        #     )
    model.train()
    return ret_metrics,merged_dict

if __name__ == "__main__":
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    main()
