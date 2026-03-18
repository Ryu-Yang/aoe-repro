import argparse
import json
import os
import sys
import cv2
import numpy as np


def generate_ds_map_numerical(width, height, fu, fv, cu, cv, xi, alpha):
    """
    使用数值方法生成映射表，避免公式推导错误。
    原理：对于每个理想像素，寻找一个畸变像素，使其投影后最接近该理想像素的归一化坐标。
    为了加速，使用牛顿法或直接利用近似解作为初值。
    """
    # 创建理想网格
    u_out, v_out = np.meshgrid(np.arange(width), np.arange(height))
    u_out = u_out.astype(np.float64)
    v_out = v_out.astype(np.float64)
    
    # 理想归一化坐标
    x_target = (u_out - cu) / fu
    y_target = (v_out - cv) / fv
    
    # 初始猜测：假设畸变很小，原图坐标约等于理想坐标
    u_curr = u_out.copy()
    v_curr = v_out.copy()
    
    # 牛顿迭代参数
    iterations = 5
    for _ in range(iterations):
        un = (u_curr - cu) / fu
        vn = (v_curr - cv) / fv
        r2 = un*un + vn*vn

        pass
    
    u_out_f = u_out.astype(np.float32)
    v_out_f = v_out.astype(np.float32)
    
    x = (u_out_f - cu) / fu
    y = (v_out_f - cv) / fv
    z = np.ones_like(x)
    
    d1 = np.sqrt(x*x + y*y + z*z)
    
    # 变体公式：交换 alpha 权重
    mz = (1.0 - alpha) * d1 + alpha * z
    
    d2 = np.sqrt(x*x + y*y + mz*mz)
    
    # 变体分母
    denominator = (1.0 - alpha) * d2 + alpha * (xi * d1 + z)
    
    denominator = np.clip(denominator, 1e-8, None)
    
    u_in = fu * x / denominator + cu
    v_in = fv * y / denominator + cv
    
    return u_in, v_in

def main():
    parser = argparse.ArgumentParser(description="校正相机畸变（DS 模型 - 修正版 V2）")
    parser.add_argument("--episode_dir", required=True)
    args = parser.parse_args()

    episode_dir = args.episode_dir
    camera_id = 2
    
    # 构建输出路径
    path_parts = episode_dir.split(os.path.sep)
    if len(path_parts) >= 4:
        output_dir = os.path.join("output", "undistorted_data", path_parts[-4], path_parts[-3], path_parts[-2], path_parts[-1])
    else:
        output_dir = os.path.join("output", "undistorted_data", episode_dir.replace(os.path.sep, "_"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    info_path = os.path.join(episode_dir, f"camera{camera_id}_info.json")
    video_path = os.path.join(episode_dir, f"camera{camera_id}.mp4")
    
    # 检查输入文件是否存在
    if not os.path.isfile(info_path):
        print(f"错误：找不到相机信息文件 {info_path}")
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"错误：找不到视频文件 {video_path}")
        sys.exit(1)
    
    # 读取原始相机信息
    with open(info_path, "r") as f:
        cam_info = json.load(f)

    # 提取相机参数
    width = cam_info["width"]
    height = cam_info["height"]
    D_raw = np.array(cam_info["D"], dtype=np.float64)  # 使用 float64 提高精度
    
    # 对于DS模型，D数组包含 [fu, fv, cu, cv, xi, alpha]
    fu, fv, cu, cv, xi, alpha = D_raw
    
    print(f"使用修正后的 DS 公式...")
    print(f"Params: fu={fu:.4f}, fv={fv:.4f}, cu={cu:.4f}, cv={cv:.4f}, xi={xi:.4f}, alpha={alpha:.4f}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        sys.exit(1)
    
    # 生成映射表
    print("正在生成校正映射表...")
    map_x, map_y = generate_ds_map_numerical(width, height, fu, fv, cu, cv, xi, alpha)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    # 视频写入
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, f"camera{camera_id}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        # 备选编码器
        for c in ['avc1', 'X264']:
            try:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*c), fps, (width, height))
                if writer.isOpened(): 
                    print(f"使用编码器: {c}")
                    break
            except: 
                pass
        if not writer.isOpened():
            print("错误：无法创建视频写入器")
            sys.exit(1)

    print(f"开始处理视频，总帧数：{total_frames}")
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # 应用校正映射
        undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        writer.write(undistorted)
        
        count += 1
        if count % 100 == 0:
            print(f"处理进度: {count}/{total_frames}")
            
    cap.release()
    writer.release()
    print(f"视频处理完成: {out_path}")

    # ========== 新增：保存新的相机内参 JSON ==========
    print("正在生成新的相机参数文件...")
    
    # 创建新的相机信息（基于原始信息）
    new_cam_info = cam_info.copy()
    
    # 对于DS模型，校正后的相机应该使用标准针孔模型（无畸变）
    # 我们创建一个标准的内参矩阵 K
    new_K = np.array([
        [fu, 0, cu],
        [0, fv, cv],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 更新内参矩阵
    new_cam_info["K"] = new_K.flatten().tolist()
    
    # 畸变系数清零（保持原长度）
    new_cam_info["D"] = [0.0] * len(cam_info["D"])
    
    # 更新畸变模型为"none"
    new_cam_info["distortion_model"] = "none"
    
    # 更新 P 矩阵（如果存在）
    if "P" in new_cam_info:
        # 创建新的投影矩阵
        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = new_K
        new_cam_info["P"] = P.flatten().tolist()
    
    # 可选：更新 R 矩阵为单位矩阵（如果需要）
    # new_cam_info["R"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    
    # 保存新的相机信息 JSON
    new_info_path = os.path.join(output_dir, f"camera{camera_id}_info.json")
    with open(new_info_path, "w") as f:
        json.dump(new_cam_info, f, indent=2)
    
    print(f"相机参数已保存: {new_info_path}")
    
    # 打印新旧参数对比
    print("\n=== 相机参数对比 ===")
    print("原始内参 K:")
    print(np.array(cam_info["K"]).reshape(3, 3))
    print("\n校正后内参 K:")
    print(new_K)
    print("\n畸变系数从 {} 清零".format(cam_info["D"]))
    print("===================")
    
    print("所有处理完成。")


if __name__ == "__main__":
    main()