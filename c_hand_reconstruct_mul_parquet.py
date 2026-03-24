from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional, List, Tuple
import threading
import queue
from collections import defaultdict
import time
import logging

# 关闭 HaMeR 的调试输出
logging.getLogger("hamer").setLevel(logging.WARNING)

# HaMeR 相关导入
from hamer.hamer.configs import CACHE_DIR_HAMER
from hamer.hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.hamer.utils import recursive_to
from hamer.hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.hamer.utils.renderer import Renderer, cam_crop_to_full

# 渲染颜色
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# 导入 ViTPose 模型
from hamer.vitpose_model import ViTPoseModel

# Parquet 相关导入
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("警告: pyarrow 未安装，将使用 JSON 格式保存结果。安装命令: pip install pyarrow")


def build_output_path(episode_dir: str) -> str:
    """
    根据 episode_dir（视频文件路径）构建输出目录
    例如: /path/to/video.mp4 -> output/hand_reconstruction/path/to/video
    """
    video_path = Path(episode_dir)
    video_name = video_path.stem
    video_parent = video_path.parent
    
    if str(video_parent) == '.':
        output_dir = os.path.join("output", "hand_reconstruction", video_name)
    else:
        path_parts = list(video_parent.parts)
        output_dir = os.path.join("output", "hand_reconstruction", *path_parts, video_name)
    
    return output_dir


def detect_hands_in_frame(img_cv2: np.ndarray, detector, cpm, args) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    检测单帧图像中的手部边界框
    返回: (boxes, is_right) 或 (None, None)
    """
    img_rgb = img_cv2[:, :, ::-1]  # BGR -> RGB
    
    # 1. 人体检测
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    if not any(valid_idx):
        return None, None
    
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    
    # 选择置信度最高的人
    best_idx = np.argmax(pred_scores)
    best_bbox = pred_bboxes[best_idx:best_idx+1]
    best_score = pred_scores[best_idx:best_idx+1]
    
    # 2. 手部关键点检测
    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([best_bbox, best_score[:, None]], axis=1)],
    )
    
    if len(vitposes_out) == 0:
        return None, None
    vitposes = vitposes_out[0]
    
    # 提取左右手关键点
    left_keyp = vitposes['keypoints'][-42:-21]
    right_keyp = vitposes['keypoints'][-21:]
    
    boxes = []
    is_right = []
    
    # 左手
    valid = left_keyp[:, 2] > 0.5
    if sum(valid) > 3:
        x1 = left_keyp[valid, 0].min()
        y1 = left_keyp[valid, 1].min()
        x2 = left_keyp[valid, 0].max()
        y2 = left_keyp[valid, 1].max()
        boxes.append([x1, y1, x2, y2])
        is_right.append(0)
    
    # 右手
    valid = right_keyp[:, 2] > 0.5
    if sum(valid) > 3:
        x1 = right_keyp[valid, 0].min()
        y1 = right_keyp[valid, 1].min()
        x2 = right_keyp[valid, 0].max()
        y2 = right_keyp[valid, 1].max()
        boxes.append([x1, y1, x2, y2])
        is_right.append(1)
    
    if len(boxes) == 0:
        return None, None
    
    return np.stack(boxes), np.stack(is_right)


class BatchHandDataset(torch.utils.data.Dataset):
    """批量手部数据集，支持多帧多手"""
    
    def __init__(self, images: List[np.ndarray], boxes: List[np.ndarray], 
                 is_right: List[int], model_cfg, rescale_factor: float = 2.0):
        self.images = images
        self.boxes = boxes
        self.is_right = is_right
        self.model_cfg = model_cfg
        self.rescale_factor = rescale_factor
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        box = self.boxes[idx]
        right = self.is_right[idx]
        
        # 使用 ViTDetDataset 的逻辑处理单个样本
        dataset = ViTDetDataset(self.model_cfg, img, box[None], np.array([right]), 
                                rescale_factor=self.rescale_factor)
        return dataset[0]


def process_batch(batch_items: List[Tuple], model, model_cfg, device, args) -> List[Dict]:
    """
    处理一批手部数据，返回每只手的结果字典
    batch_items: [(frame_idx, img, box, is_right), ...]
    """
    if not batch_items:
        return []
    
    # 准备批量数据
    images = [item[1] for item in batch_items]
    boxes = [item[2] for item in batch_items]
    is_right = [item[3] for item in batch_items]
    
    # 创建数据集和数据加载器
    dataset = BatchHandDataset(images, boxes, is_right, model_cfg, args.rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(batch_items), 
                                             shuffle=False, num_workers=0)
    
    all_results = []
    
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        
        # 处理左右手镜像
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        
        # 修复：处理标量张量
        if img_size.dim() == 0:
            img_size = img_size.unsqueeze(0)
        
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, 
                                          scaled_focal_length).detach().cpu().numpy()
        
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            is_right_flag = batch['right'][n].cpu().numpy()
            # 调整X轴方向
            verts[:, 0] = (2 * is_right_flag - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[n]
            
            # 修复：处理标量张量
            if isinstance(scaled_focal_length, torch.Tensor):
                if scaled_focal_length.dim() == 0:
                    f_val = float(scaled_focal_length.item())
                elif len(scaled_focal_length) > n:
                    f_val = float(scaled_focal_length[n].item())
                else:
                    f_val = float(scaled_focal_length[0].item())
            else:
                f_val = float(scaled_focal_length)
            
            # 保存手部参数
            hand_dict = {
                'pred_cam': pred_cam[n].detach().cpu().numpy().tolist(),
                'pred_cam_t': cam_t.tolist(),
                'focal_length': [f_val, f_val],
                'pred_mano_params': {
                    'global_orient': out['pred_mano_params']['global_orient'][n].detach().cpu().numpy().tolist(),
                    'hand_pose': out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy().tolist(),
                    'betas': out['pred_mano_params']['betas'][n].detach().cpu().numpy().tolist(),
                },
                'pred_vertices': verts.tolist(),
                'pred_keypoints_3d': out['pred_keypoints_3d'][n].detach().cpu().numpy().tolist(),
                'pred_keypoints_2d': out['pred_keypoints_2d'][n].detach().cpu().numpy().tolist(),
                'hand_type': 'left' if is_right_flag == 0 else 'right',
                'bbox': boxes[n].tolist(),
            }
            all_results.append(hand_dict)
    
    return all_results


def render_frame(img_cv2: np.ndarray, hands_results: List[Dict], renderer, model_cfg) -> Optional[np.ndarray]:
    """渲染单帧图像"""
    if not hands_results:
        return None
    
    all_verts = []
    all_cam_t = []
    all_right_flags = []
    
    for hand_dict in hands_results:
        all_verts.append(np.array(hand_dict['pred_vertices']))
        all_cam_t.append(np.array(hand_dict['pred_cam_t']))
        all_right_flags.append(1 if hand_dict['hand_type'] == 'right' else 0)
    
    # 获取图像尺寸
    img_h, img_w = img_cv2.shape[:2]
    all_img_size = torch.tensor([img_w, img_h])
    
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)
    
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    
    cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t,
                                             render_res=all_img_size,
                                             is_right=all_right_flags, **misc_args)
    
    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    rendered_img = (255 * overlay[:, :, ::-1]).astype(np.uint8)
    
    return rendered_img


def save_to_parquet(all_frame_data: List[Dict], output_path: str):
    """保存数据为 Parquet 格式"""
    if not PARQUET_AVAILABLE:
        print("pyarrow 未安装，跳过 Parquet 保存")
        return False
    
    # 简化版本：将复杂数据转换为扁平结构
    flat_data = []
    
    for fd in all_frame_data:
        frame_idx = fd['frame_index']
        timestamp = fd['timestamp_sec']
        
        for hand in fd['hands']:
            row = {
                'frame_index': frame_idx,
                'timestamp_sec': timestamp,
                'hand_type': hand['hand_type'],
                'bbox': str(hand['bbox']),  # 转为字符串
                'confidence': hand.get('confidence', None),
                'pred_cam': str(hand['pred_cam']),
                'pred_cam_t': str(hand['pred_cam_t']),
                'focal_length': str(hand['focal_length']),
                # 保存为 JSON 字符串避免嵌套复杂度
                'pred_mano_params': json.dumps(hand['pred_mano_params']),
                'pred_vertices': json.dumps(hand['pred_vertices']),
                'pred_keypoints_3d': json.dumps(hand['pred_keypoints_3d']),
                'pred_keypoints_2d': json.dumps(hand['pred_keypoints_2d']),
            }
            flat_data.append(row)
    
    if not flat_data:
        return False
    
    try:
        # 使用 pandas 作为中间格式（更简单）
        import pandas as pd
        df = pd.DataFrame(flat_data)
        df.to_parquet(output_path, index=False, compression='snappy')
        return True
    except ImportError:
        # 如果没有 pandas，使用 pyarrow 直接转换
        try:
            table = pa.Table.from_pylist(flat_data)
            pq.write_table(table, output_path, compression='snappy')
            return True
        except Exception as e:
            print(f"保存 Parquet 失败: {e}")
            return False
    except Exception as e:
        print(f"保存 Parquet 失败: {e}")
        return False


def producer(video_path: str, task_queue: queue.Queue, detector, cpm, args, stop_event):
    """生产者线程：读取视频并检测手部边界框"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        stop_event.set()
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    print(f"生产者线程启动，总帧数: {total_frames}")
    
    while not stop_event.is_set():
        ret, img = cap.read()
        if not ret:
            break
        
        try:
            # 检测手部
            boxes, is_right = detect_hands_in_frame(img, detector, cpm, args)
            
            if boxes is not None:
                for i in range(len(boxes)):
                    # 限制队列大小，避免内存爆炸
                    while task_queue.qsize() > 200 and not stop_event.is_set():
                        time.sleep(0.01)
                    task_queue.put((frame_idx, img.copy(), boxes[i], is_right[i]))
        except Exception as e:
            print(f"生产者处理帧 {frame_idx} 时出错: {e}")
        
        frame_idx += 1
        
        # 进度显示
        if frame_idx % 100 == 0:
            print(f"生产者进度: {frame_idx}/{total_frames}")
    
    cap.release()
    print(f"生产者完成，共处理 {frame_idx} 帧")
    
    # 发送结束标志给所有消费者
    for _ in range(args.num_consumers):
        task_queue.put(None)


def consumer(task_queue: queue.Queue, result_queue: queue.Queue, model, model_cfg, 
             device, args, stop_event, consumer_id):
    """消费者线程：批量推理手部数据"""
    batch = []
    batch_count = 0
    
    print(f"消费者 {consumer_id} 启动")
    
    while not stop_event.is_set():
        try:
            item = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        if item is None:  # 结束标志
            if batch:
                # 处理剩余批次
                try:
                    results = process_batch(batch, model, model_cfg, device, args)
                    result_queue.put((batch, results))
                    batch_count += 1
                except Exception as e:
                    print(f"消费者 {consumer_id} 处理批次时出错: {e}")
            break
        
        batch.append(item)
        
        if len(batch) >= args.batch_size:
            try:
                results = process_batch(batch, model, model_cfg, device, args)
                result_queue.put((batch, results))
                batch_count += 1
                batch = []
            except Exception as e:
                print(f"消费者 {consumer_id} 处理批次时出错: {e}")
                batch = []
    
    print(f"消费者 {consumer_id} 完成，处理了 {batch_count} 个批次")


def result_collector(result_queue: queue.Queue, output_dir: str, args, 
                    renderer, model_cfg, total_frames: int, fps: float):
    """结果收集线程：整理结果并保存"""
    frame_results = defaultdict(list)  # frame_idx -> list of hand dicts
    processed_batches = 0
    
    print("结果收集线程启动")
    
    while True:
        try:
            item = result_queue.get(timeout=5)
        except queue.Empty:
            # 超时，继续等待
            continue
        
        if item is None:
            break
        
        batch, results = item
        
        # 将结果分配到对应帧
        for (frame_idx, img, box, is_right), hand_result in zip(batch, results):
            frame_results[frame_idx].append(hand_result)
        
        processed_batches += 1
        if processed_batches % 10 == 0:
            print(f"结果收集进度: 已处理 {processed_batches} 个批次，涉及 {len(frame_results)} 帧")
    
    print("开始整理和保存结果...")
    
    # 重新读取视频用于渲染
    cap = cv2.VideoCapture(args.episode_dir)
    if not cap.isOpened():
        print(f"无法打开视频文件用于渲染: {args.episode_dir}")
        cap = None
    
    # 视频写入器
    video_writer = None
    if args.render_output and args.full_frame and cap:
        out_video_path = os.path.join(output_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        print(f"输出视频将保存到: {out_video_path}")
    
    # 收集所有帧数据
    all_frame_data = []
    frame_idx = 0
    rendered_count = 0
    
    while cap and cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        if frame_idx in frame_results:
            # 渲染该帧
            rendered = None
            if args.render_output and args.full_frame and video_writer:
                try:
                    rendered = render_frame(img, frame_results[frame_idx], renderer, model_cfg)
                    if rendered is not None:
                        video_writer.write(rendered)
                        rendered_count += 1
                except Exception as e:
                    print(f"渲染帧 {frame_idx} 时出错: {e}")
            
            # 构建帧数据
            frame_data = {
                'frame_index': frame_idx,
                'timestamp_sec': frame_idx / fps if fps > 0 else frame_idx,
                'hands': frame_results[frame_idx],
            }
            all_frame_data.append(frame_data)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"渲染进度: {frame_idx}/{total_frames} 帧，已渲染 {rendered_count} 帧")
    
    if cap:
        cap.release()
    if video_writer:
        video_writer.release()
    
    # 保存结果
    if args.save_json:
        json_path = os.path.join(output_dir, "hand_data.json")
        print(f"正在保存 JSON 到 {json_path}...")
        # 转换为 JSON 兼容格式
        json_data = []
        for fd in all_frame_data:
            fd_copy = {
                'frame_index': fd['frame_index'],
                'timestamp_sec': fd['timestamp_sec'],
                'hands': []
            }
            for hand in fd['hands']:
                hand_copy = {}
                for key, value in hand.items():
                    if isinstance(value, (list, dict)):
                        hand_copy[key] = value
                    else:
                        hand_copy[key] = value
                fd_copy['hands'].append(hand_copy)
            json_data.append(fd_copy)
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"已保存 JSON 到 {json_path}")
    
    # 保存 Parquet
    if args.save_parquet and PARQUET_AVAILABLE:
        parquet_path = os.path.join(output_dir, "hand_data.parquet")
        print(f"正在保存 Parquet 到 {parquet_path}...")
        if save_to_parquet(all_frame_data, parquet_path):
            print(f"已保存 Parquet 到 {parquet_path}")
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"总帧数: {frame_idx}")
    print(f"有效帧数（检测到手部）: {len(all_frame_data)}")
    if frame_idx > 0:
        print(f"成功率: {len(all_frame_data)/frame_idx*100:.2f}%")
    if args.render_output and args.full_frame:
        print(f"渲染帧数: {rendered_count}")
    
    print("结果收集完成")


def main():
    parser = argparse.ArgumentParser(description='HaMeR video processing with batch inference and multi-threading')
    # 输入参数
    parser.add_argument('--episode_dir', type=str, required=True,
                        help='视频文件路径，例如 /path/to/video.mp4')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='HaMeR预训练模型检查点路径')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='输出结果文件夹（如果指定，将覆盖自动生成的路径）')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True,
                        help='是否将渲染的手部叠加到原图上')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='推理批大小（跨帧累积的手部数量）')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='边界框填充因子')
    parser.add_argument('--body_detector', type=str, default='vitdet',
                        choices=['vitdet', 'regnety'],
                        help='人体检测器类型')
    parser.add_argument('--save_json', action='store_true', default=False,
                        help='是否将检测结果保存为JSON文件')
    parser.add_argument('--save_parquet', action='store_true', default=True,
                        help='是否将检测结果保存为Parquet文件（推荐）')
    parser.add_argument('--render_output', action='store_true', default=True,
                        help='是否保存渲染图像（否则仅保存JSON/Parquet）')
    parser.add_argument('--num_consumers', type=int, default=2,
                        help='消费者线程数（GPU推理线程）')
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.episode_dir):
        raise FileNotFoundError(f"视频文件不存在: {args.episode_dir}")
    
    # 构建输出目录
    if args.out_folder is not None:
        output_dir = args.out_folder
    else:
        output_dir = build_output_path(args.episode_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入视频: {args.episode_dir}")
    print(f"输出目录: {output_dir}")
    
    # 加载模型
    print("正在加载模型...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    print(f"使用设备: {device}")
    
    # 加载人体检测器
    print("正在加载人体检测器...")
    from hamer.hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    # 加载手部关键点检测器
    print("正在加载手部关键点检测器...")
    cpm = ViTPoseModel(device)
    
    # 初始化渲染器
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    # 获取视频信息
    cap = cv2.VideoCapture(args.episode_dir)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    
    # 创建队列
    task_queue = queue.Queue(maxsize=100)  # 限制队列大小避免内存爆炸
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    # 创建并启动线程
    threads = []
    
    # 生产者线程
    producer_thread = threading.Thread(
        target=producer,
        args=(args.episode_dir, task_queue, detector, cpm, args, stop_event)
    )
    producer_thread.start()
    threads.append(producer_thread)
    
    # 消费者线程
    consumer_threads = []
    for i in range(args.num_consumers):
        consumer_thread = threading.Thread(
            target=consumer,
            args=(task_queue, result_queue, model, model_cfg, device, args, stop_event, i)
        )
        consumer_thread.start()
        consumer_threads.append(consumer_thread)
        threads.append(consumer_thread)
    
    # 结果收集线程
    collector_thread = threading.Thread(
        target=result_collector,
        args=(result_queue, output_dir, args, renderer, model_cfg, total_frames, fps)
    )
    collector_thread.start()
    threads.append(collector_thread)
    
    # 等待生产者完成
    producer_thread.join()
    
    # 等待所有消费者完成
    for ct in consumer_threads:
        ct.join()
    
    # 所有消费者已结束，发送结束信号给结果收集线程
    result_queue.put(None)
    
    # 等待结果收集线程完成
    collector_thread.join()
    
    print("所有处理完成！")


if __name__ == '__main__':
    main()