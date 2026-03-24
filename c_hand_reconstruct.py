from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional

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


def build_output_path(episode_dir: str) -> str:
    """
    根据 episode_dir（视频文件路径）构建输出目录
    例如: /path/to/video.mp4 -> output/hand_reconstruction/path/to/video
    """
    # 获取视频文件路径（不含扩展名）
    video_path = Path(episode_dir)
    video_name = video_path.stem  # 不带扩展名的文件名
    video_parent = video_path.parent  # 父目录
    
    # 构建输出路径: output/hand_reconstruction/父目录路径/视频文件名
    if str(video_parent) == '.':
        output_dir = os.path.join("output", "hand_reconstruction", video_name)
    else:
        # 将父目录路径拆分为多个部分
        path_parts = list(video_parent.parts)
        output_dir = os.path.join("output", "hand_reconstruction", *path_parts, video_name)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo for video/hand reconstruction')
    # 输入参数
    parser.add_argument('--episode_dir', type=str, required=True,
                        help='视频文件路径，例如 /path/to/video.mp4')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='HaMeR预训练模型检查点路径')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='输出结果文件夹（如果指定，将覆盖自动生成的路径）')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False,
                        help='是否同时渲染侧视图')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True,
                        help='是否将渲染的手部叠加到原图上')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False,
                        help='是否保存3D网格为.obj文件')
    parser.add_argument('--batch_size', type=int, default=128, help='推理批大小')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='边界框填充因子')
    parser.add_argument('--body_detector', type=str, default='vitdet',
                        choices=['vitdet', 'regnety'],
                        help='人体检测器类型')
    parser.add_argument('--save_json', action='store_true', default=True,
                        help='是否将检测结果保存为JSON文件')
    parser.add_argument('--render_output', action='store_true', default=False,
                        help='是否保存渲染图像（否则仅保存JSON）')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'],
                        help='图像文件扩展名（保留兼容性）')
    args = parser.parse_args()

    # --------------------------
    # 检查输入视频文件是否存在
    # --------------------------
    if not os.path.isfile(args.episode_dir):
        raise FileNotFoundError(f"视频文件不存在: {args.episode_dir}")
    
    video_path = args.episode_dir
    
    # --------------------------
    # 构建输出目录
    # --------------------------
    if args.out_folder is not None:
        output_dir = args.out_folder
    else:
        output_dir = build_output_path(video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入视频: {video_path}")
    print(f"输出目录: {output_dir}")

    # --------------------------
    # 加载模型
    # --------------------------
    print("正在加载模型...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    print(f"使用设备: {device}")

    # 加载人体检测器
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
    else:  # regnety
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # 加载手部关键点检测器
    cpm = ViTPoseModel(device)

    # 初始化渲染器
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # --------------------------
    # 处理视频
    # --------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS, 分辨率 {width}x{height}")

    # 用于存储每一帧的数据
    all_frame_data = []
    
    # 可选：创建输出视频写入器
    video_writer = None
    if args.render_output:
        out_video_path = os.path.join(output_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        print(f"输出视频将保存到: {out_video_path}")

    frame_idx = 0
    processed_frames = 0
    
    try:
        while True:
            ret, img_cv2 = cap.read()
            if not ret:
                break
            
            # 处理当前帧
            frame_data, rendered_img = process_frame(
                img_cv2, model, model_cfg, detector, cpm, renderer, device, args
            )
            
            if frame_data is not None:
                # 添加时间戳（帧索引和时间）
                frame_data['frame_index'] = frame_idx
                frame_data['timestamp_sec'] = frame_idx / fps if fps > 0 else frame_idx
                
                if args.save_json:
                    all_frame_data.append(frame_data)
                
                processed_frames += 1
                
                # 保存渲染图像/视频
                if args.render_output and rendered_img is not None:
                    if video_writer is not None:
                        video_writer.write(rendered_img)
                    
                    # 同时保存每帧图像（可选）
                    if args.save_mesh:  # 使用save_mesh标志控制是否保存单帧图像
                        frame_img_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_img_path, rendered_img)
            
            frame_idx += 1
            
            # 进度显示
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"进度: {frame_idx}/{total_frames} 帧, 有效帧: {processed_frames}")
                
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    # --------------------------
    # 保存JSON结果
    # --------------------------
    if args.save_json and all_frame_data:
        # 将numpy数组转换为列表以便JSON序列化
        json_data = []
        for fd in all_frame_data:
            fd_copy = fd.copy()
            if 'hands' in fd_copy:
                for hand in fd_copy['hands']:
                    for key, value in hand.items():
                        if isinstance(value, np.ndarray):
                            hand[key] = value.tolist()
            json_data.append(fd_copy)
        
        json_path = os.path.join(output_dir, "hand_data.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"已保存JSON到 {json_path}")
        print(f"总共保存了 {len(all_frame_data)} 帧的有效数据")
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"总帧数: {frame_idx}")
    print(f"有效帧数（检测到手部）: {processed_frames}")
    print(f"成功率: {processed_frames/frame_idx*100:.2f}%" if frame_idx > 0 else "N/A")


def process_frame(img_cv2: np.ndarray, model, model_cfg, detector, cpm, renderer, device, args) -> tuple:
    """
    处理单帧图像，返回该帧的手部检测数据及渲染图像。
    返回: (frame_data_dict, rendered_image) 或 (None, None)
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

    # 选择置信度最高的人（第一人称假设）
    best_idx = np.argmax(pred_scores)
    best_bbox = pred_bboxes[best_idx:best_idx+1]  # 保持形状 (1,4)
    best_score = pred_scores[best_idx:best_idx+1]

    # 2. 手部关键点检测（仅对选中的人）
    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([best_bbox, best_score[:, None]], axis=1)],
    )
    
    if len(vitposes_out) == 0:
        return None, None
    vitposes = vitposes_out[0]

    # 提取左右手关键点
    left_keyp = vitposes['keypoints'][-42:-21]   # 左手21点
    right_keyp = vitposes['keypoints'][-21:]     # 右手21点

    bboxes = []      # 存储 (x1,y1,x2,y2)
    is_right = []    # 0左 1右

    # 左手
    valid = left_keyp[:, 2] > 0.5
    if sum(valid) > 3:
        x1 = left_keyp[valid, 0].min()
        y1 = left_keyp[valid, 1].min()
        x2 = left_keyp[valid, 0].max()
        y2 = left_keyp[valid, 1].max()
        bboxes.append([x1, y1, x2, y2])
        is_right.append(0)

    # 右手
    valid = right_keyp[:, 2] > 0.5
    if sum(valid) > 3:
        x1 = right_keyp[valid, 0].min()
        y1 = right_keyp[valid, 1].min()
        x2 = right_keyp[valid, 0].max()
        y2 = right_keyp[valid, 1].max()
        bboxes.append([x1, y1, x2, y2])
        is_right.append(1)

    if len(bboxes) == 0:
        return None, None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # 3. 准备HaMeR数据集
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 收集当前帧所有手的结果
    hands_data = []      # 每个手一个字典
    all_verts = []       # 用于全图渲染
    all_cam_t = []
    all_right_flags = []
    all_img_size = None

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
            # print("out:", out)

        # 处理左右手镜像
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            is_right_flag = batch['right'][n].cpu().numpy()
            # 调整X轴方向
            verts[:, 0] = (2 * is_right_flag - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[n]

            if isinstance(scaled_focal_length, torch.Tensor):
                f_val = float(scaled_focal_length.item())
            else:
                f_val = float(scaled_focal_length)

            # 保存手部参数
            hand_dict = {
                # 1. 相机参数
                'pred_cam': pred_cam[n].detach().cpu().numpy().tolist(),  # (3,) [s, tx, ty]
                'pred_cam_t': cam_t.tolist(),  # (3,) [tx, ty, tz]
                'focal_length': [f_val, f_val],
                
                # 2. MANO 参数（核心）
                'pred_mano_params': {
                    'global_orient': out['pred_mano_params']['global_orient'][n].detach().cpu().numpy().tolist(),  # (1, 3, 3)
                    'hand_pose': out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy().tolist(),  # (15, 3, 3)
                    'betas': out['pred_mano_params']['betas'][n].detach().cpu().numpy().tolist(),  # (10,)
                },
                
                # 3. 3D 输出
                'pred_vertices': verts.tolist(),  # (778, 3) - 已处理左右手镜像
                'pred_keypoints_3d': out['pred_keypoints_3d'][n].detach().cpu().numpy().tolist(),  # (21, 3)
                
                # 4. 2D 输出
                'pred_keypoints_2d': out['pred_keypoints_2d'][n].detach().cpu().numpy().tolist(),  # (21, 2)
                
                # 元数据（便于后续解析）
                'hand_type': 'left' if is_right_flag == 0 else 'right',
                'bbox': boxes[n].tolist() if n < len(boxes) else None,
                'confidence': float(best_score[0]) if best_score.size > 0 else None,
            }
            hands_data.append(hand_dict)

            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right_flags.append(is_right_flag)
            all_img_size = img_size[n]

    # 4. 可选：全图渲染
    rendered_img = None
    if args.full_frame and len(all_verts) > 0 and args.render_output:
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

    # 返回该帧的数据
    frame_data = {
        'hands': hands_data,
    }
    return frame_data, rendered_img


if __name__ == '__main__':
    main()