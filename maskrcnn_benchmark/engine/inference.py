# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
    Tensor = 张量 = PyTorch 里的 多维数组（可放 GPU）。
    例如 1 张 224×224 的 RGB 图像 → 形状 [3, 224, 224] 的 Tensor。
"""
import logging
import time
import os

import json
import torch
from tqdm import tqdm # 进度条库

from maskrcnn_benchmark.config import cfg # 导入全局配置对象 cfg（存储所有超参数）
from maskrcnn_benchmark.data.datasets.evaluation import evaluate # 导入 evaluate：给定预测结果与数据集，计算 AP 等指标
from ..utils.comm import is_main_process, get_world_size # 多卡通信工具：判断当前进程是否主进程、获取总 GPU 数
from ..utils.comm import all_gather # 多卡通信：把各进程的字典收集到主进程
from ..utils.comm import synchronize # 多卡同步：阻塞直到所有进程跑到这里
from ..utils.timer import Timer, get_time_str # 计时器：统计总耗时与打印可读字符串
from .bbox_aug import im_detect_bbox_aug # 导入测试时增强（TTA）函数，用于 bbox 多尺度/翻转增强

# 逐张图片前向推断，返回结果字典
def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None): # synchronize_gather：是否多卡同步收集结果
    model.eval() # 把模型设为评估模式（关闭 dropout、BN 用均值方差）
    results_dict = {} # 存储每张图片的推断结果 {image_id: BoxList}
    cpu_device = torch.device("cpu") # 结果先搬回 CPU，避免显存爆炸
    torch.cuda.empty_cache() # 清空显存碎片
    # 遍历 DataLoader，tqdm 显示进度条
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad(): # 不计算梯度，节省显存与时间
            images, targets, image_ids = batch # images：预处理后的图片 Tensor，targets：标注（推断阶段可忽略），image_ids：每张图片的唯一编号
            targets = [target.to(device) for target in targets] # 把标注移到 GPU（某些关系检测需要）
            if timer:
                timer.tic() # 开始计时
            # 如果开启了测试时增强 (TTA)
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device) #把同一张图片做翻转、多尺度缩放 等变换 → 得到多张变体。每张变体都跑一次前向 → 得到各自的 bbox 预测。把所有变体的结果 融合（如投票/加权平均），提高精度。
            else:
                # relation detection needs the targets
                output = model(images.to(device), targets)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        # 多卡模式下，收集各进程结果
        if synchronize_gather:
            synchronize() # 先同步
            # 构造 {image_id: result} 字典
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            # 主进程把各子进程的字典合并
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            # 单卡或无需同步，直接更新
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
    torch.cuda.empty_cache()
    return results_dict # 返回所有图片的推断结果

# 把多卡的预测字典合并成一个有序列表
def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True): # predictions_per_gpu：每张卡的 {image_id: BoxList}
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    # 非主进程直接退出，不评估
    if not is_main_process():
        return
    # 已经同步好，直接用
    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    
    # convert a dict where the key is the index in a list
    # 按 image_id 排序，确保顺序
    image_ids = list(sorted(predictions.keys()))
    # 如果 id 不连续，说明可能丢图
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    # 把 dict 转成 list，按 image_id 顺序
    predictions = [predictions[i] for i in image_ids]
    return predictions

# 主推断函数：负责整体流程
def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",), # 主推断函数：负责整体流程
        box_only=False, # 只用 RPN 时设为 True
        device="cuda",
        expected_results=(), # 期望的 AP 结果，用于报警
        expected_results_sigma_tol=4, # 容忍区间
        output_folder=None, # 结果保存目录
        logger=None,
):
    # 是否允许从缓存加载预测结果
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    device = torch.device(device) # 转成 torch.device
    num_devices = get_world_size() # GPU 总数
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
    # 打印评估数据集及图片数量
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()  # 开始计时
    # 若缓存存在，直接加载
    if load_prediction_from_cache:
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
    else:
        # 否则真正跑推断
        predictions = compute_on_dataset(model, data_loader, device, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize() # 等待所有进程完成
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    # 如果不是从缓存加载，则合并多卡结果
    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)
    # 非主进程结束
    if not is_main_process():
        return -1.0

    #if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
    # 传给 evaluate 的额外参数
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    # 自定义场景图评估分支
    if cfg.TEST.CUSTUM_EVAL:
        detected_sgg = custom_sgg_post_precessing(predictions)
        with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json'), 'w') as outfile:  
            json.dump(detected_sgg, outfile)
        print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json')) + ' SAVED !')
        return -1.0
    # 正式评估：计算 AP 并返回
    return evaluate(cfg=cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)


# 自定义场景图后处理：把 BoxList 转成 JSON 所需格式
def custom_sgg_post_precessing(predictions):
    output_dict = {}
    for idx, boxlist in enumerate(predictions):
        xyxy_bbox = boxlist.convert('xyxy').bbox
        # current sgg info
        current_dict = {}
        # sort bbox based on confidence
        sortedid, id2sorted = get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())
        # sorted bbox label and score
        bbox = []
        bbox_labels = []
        bbox_scores = []
        for i in sortedid:
            bbox.append(xyxy_bbox[i].tolist())
            bbox_labels.append(boxlist.get_field('pred_labels')[i].item())
            bbox_scores.append(boxlist.get_field('pred_scores')[i].item())
        current_dict['bbox'] = bbox
        current_dict['bbox_labels'] = bbox_labels
        current_dict['bbox_scores'] = bbox_scores
        # sorted relationships
        rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())
        # sorted rel
        rel_pairs = []
        rel_labels = []
        rel_scores = []
        rel_all_scores = []
        for i in rel_sortedid:
            rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
            rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
            rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
            old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
            rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['rel_pairs'] = rel_pairs
        current_dict['rel_labels'] = rel_labels
        current_dict['rel_scores'] = rel_scores
        current_dict['rel_all_scores'] = rel_all_scores
        output_dict[idx] = current_dict
    return output_dict
# 工具函数：根据得分列表返回排序映射
def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted