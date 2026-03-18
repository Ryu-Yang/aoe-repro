import argparse
import json
import os
import sys
import time
import base64
import tempfile
import shutil
from PIL import Image
import cv2
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

# ==================== 配置区域 ====================
DEFAULT_GRAIN = 1.0               # 细粒度（秒）
DEFAULT_ADVANCE = 8               # 增大超前量，减少调用次数
CHANGE_THRESHOLD = 1.0            # 增大变化点精度，减少二分查找次数
TARGET_IMAGE_SIZE = (320, 240)    # 减小图像尺寸，加快传输和处理
DEFAULT_CLIP_DURATION = 2.0       # 每个视频片段的时长（秒）
MAX_FRAMES_PER_CLIP = 15          # 减少最大帧数，加快响应
MIN_FRAMES_REQUIRED = 4           # 模型要求的最小帧数
MAX_FPS = 10.0                    # 模型允许的最大fps
MIN_FPS = 0.1                     # 模型允许的最小fps
CACHE_SIZE = 128                  # LRU缓存大小
MAX_WORKERS = 3                   # 并行API调用线程数
SKIP_THRESHOLD = 3                # 连续无变化跳过次数阈值
# =================================================

SYSTEM_PROMPT = """你是一个视频动作识别专家。给定一个短时视频片段（多张连续图像），你需要识别该片段中正在发生的动作，特别是人（手）与物体的交互。请输出一个JSON对象，包含以下字段：
- "verb": 动词原形，如 "hold", "cut", "pour"（英文）
- "object": 物体名称，如 "carrot", "egg", "pan"（英文）
- "description_en": 英文描述，如 "right hand holds carrot"
- "description_cn": 中文描述，如 "右手拿着胡萝卜"
- "bbox": 边界框，格式为 [x1, y1, x2, y2]，表示手或物体所在的区域（像素坐标）
- "confidence": 置信度，0到1之间的浮点数

请仔细观察这个视频片段中人物正在做什么，关注连续的动作过程。
如果片段中没有明显的动作或无法识别，请输出 null。
只输出JSON，不要有其他文字。"""

# 线程本地存储，用于每个线程独立的客户端
thread_local = threading.local()

def get_client(api_key, base_url):
    """获取线程本地的OpenAI客户端"""
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(api_key=api_key, base_url=base_url)
    return thread_local.client

def get_video_info(video_path):
    """获取视频的fps、总帧数和时长（秒）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return fps, frame_count, duration

def frame_to_base64(frame, target_size=TARGET_IMAGE_SIZE):
    """将OpenCV帧（BGR）转换为base64字符串（JPEG格式），并调整大小"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)  # 降低图片质量加快编码
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_clip_base64(video_path, center_time_sec, clip_duration, max_frames=MAX_FRAMES_PER_CLIP, target_size=TARGET_IMAGE_SIZE):
    """
    抽取以 center_time_sec 为中心的视频片段，时长为 clip_duration。
    自动调整采样率，使总帧数不超过 max_frames。
    返回 (base64列表, 实际对应的时间戳列表)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算片段的时间范围
    half_duration = clip_duration / 2
    start_time = max(0, center_time_sec - half_duration)
    end_time = min(total_frames / fps, center_time_sec + half_duration)
    actual_duration = end_time - start_time
    
    # 计算需要抽取的总帧数（原始帧数）
    total_original_frames = int(actual_duration * fps)
    
    # 确定采样间隔，使总帧数不超过 max_frames
    if total_original_frames <= max_frames:
        sample_interval = 1
        num_frames = total_original_frames
    else:
        sample_interval = total_original_frames // max_frames
        num_frames = max_frames
    
    # 计算起始帧索引
    start_idx = int(start_time * fps)
    
    base64_list = []
    timestamps = []
    
    for i in range(0, num_frames, 2):  # 隔一帧取一帧，进一步减少帧数
        idx = start_idx + i * sample_interval
        if idx >= total_frames:
            idx = total_frames - 1
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            base64_list.append(frame_to_base64(frame, target_size))
            timestamps.append(idx / fps)
    
    cap.release()
    
    # 确保至少有 MIN_FRAMES_REQUIRED 帧
    if len(base64_list) < MIN_FRAMES_REQUIRED and base64_list:
        while len(base64_list) < MIN_FRAMES_REQUIRED:
            base64_list.append(base64_list[-1])
            timestamps.append(timestamps[-1] + 0.01)
    
    return base64_list, timestamps

def call_vlm_video(frames_base64, timestamps, prompt, api_key, base_url):
    """
    调用多模态大模型，传入多帧（视频）和提示词，返回解析后的动作字典。
    """
    try:
        client = get_client(api_key, base_url)
        
        # 计算合理的 fps
        if len(frames_base64) > 1 and timestamps[-1] > timestamps[0]:
            raw_fps = len(frames_base64) / (timestamps[-1] - timestamps[0])
        else:
            raw_fps = 5.0
        fps = max(MIN_FPS, min(MAX_FPS, raw_fps))

        # 构造内容
        content = []
        video_content = {
            "type": "video",
            "video": [f"data:image/jpeg;base64,{b64}" for b64 in frames_base64],
            "fps": fps
        }
        content.append(video_content)
        content.append({"type": "text", "text": prompt})

        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            timeout=30  # 设置超时时间
        )
        
        result_str = completion.choices[0].message.content
        import re
        json_match = re.search(r'\{.*\}|null', result_str, re.DOTALL)
        if json_match:
            result = json_match.group()
            if result.strip() == "null":
                return None
            return json.loads(result)
        return None
    except Exception as e:
        print(f"模型调用失败: {e}", file=sys.stderr)
        return None

def compare_actions(act1, act2):
    """比较两个动作是否相同（基于verb和object）"""
    if act1 is None or act2 is None:
        return False
    return act1.get("verb") == act2.get("verb") and act1.get("object") == act2.get("object")

def build_action_commands(actions_en):
    """从英文动作列表构建最终输出格式。"""
    actions_output_en = []
    actions_output_cn = []
    for act in actions_en:
        en_item = {
            "start_time": act["start_time"],
            "end_time": act["end_time"],
            "verb": act["verb"],
            "object": act["object"],
            "description": act.get("description_en", ""),
            "bbox": act.get("bbox", []),
            "confidence": act.get("confidence", 0.0)
        }
        cn_item = {
            "start_time": act["start_time"],
            "end_time": act["end_time"],
            "verb": act["verb"],
            "object": act["object"],
            "description": act.get("description_cn", ""),
            "bbox": act.get("bbox", []),
            "confidence": act.get("confidence", 0.0)
        }
        actions_output_en.append(en_item)
        actions_output_cn.append(cn_item)
    return actions_output_en, actions_output_cn

class ActionDetector:
    def __init__(self, video_path, api_key, base_url, clip_duration, max_frames):
        self.video_path = video_path
        self.api_key = api_key
        self.base_url = base_url
        self.clip_duration = clip_duration
        self.max_frames = max_frames
        self.cache = {}  # 简单缓存，不使用lru_cache因为需要序列化
        
    def get_action_at(self, time_sec):
        """获取指定时间的动作（带缓存）"""
        # 四舍五入到0.1秒，增加缓存命中率
        cache_key = round(time_sec * 10) / 10
        if cache_key in self.cache:
            print(f"  缓存命中时间 {time_sec:.2f}s")
            return self.cache[cache_key]
        
        print(f"  处理时间 {time_sec:.2f}s 附近的片段（时长 {self.clip_duration:.1f}秒）...")
        frames_b64, timestamps = extract_clip_base64(
            self.video_path, time_sec, self.clip_duration, self.max_frames, TARGET_IMAGE_SIZE
        )
        
        if len(frames_b64) < MIN_FRAMES_REQUIRED:
            print(f"  无法抽取到至少 {MIN_FRAMES_REQUIRED} 帧，返回 None")
            self.cache[cache_key] = None
            return None
        
        print(f"  抽取了 {len(frames_b64)} 帧，时间范围 {timestamps[0]:.2f}-{timestamps[-1]:.2f}秒")
        act = call_vlm_video(frames_b64, timestamps, SYSTEM_PROMPT, self.api_key, self.base_url)
        
        self.cache[cache_key] = act
        return act
    
    def get_actions_batch(self, time_points):
        """批量获取多个时间点的动作（并行处理）"""
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_time = {
                executor.submit(self.get_action_at, t): t 
                for t in time_points if round(t * 10) / 10 not in self.cache
            }
            
            for future in as_completed(future_to_time):
                t = future_to_time[future]
                try:
                    results[t] = future.result()
                except Exception as e:
                    print(f"处理时间 {t:.2f}s 失败: {e}", file=sys.stderr)
                    results[t] = None
        
        # 合并缓存结果
        for t in time_points:
            cache_key = round(t * 10) / 10
            if cache_key in self.cache:
                results[t] = self.cache[cache_key]
        
        return results

def main():
    parser = argparse.ArgumentParser(description="基于多模态大模型的视频动作分割与标注（使用连续帧）")
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--camera_id", type=int, default=2)
    parser.add_argument("--grain", type=float, default=DEFAULT_GRAIN)
    parser.add_argument("--advance", type=int, default=DEFAULT_ADVANCE)
    parser.add_argument("--threshold", type=float, default=CHANGE_THRESHOLD)
    parser.add_argument("--clip_duration", type=float, default=DEFAULT_CLIP_DURATION)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_CLIP)
    parser.add_argument("--api_key", help="DashScope API Key")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--fast_mode", action="store_true", help="快速模式（使用更激进的优化）")
    args = parser.parse_args()

    # 快速模式配置
    if args.fast_mode:
        args.advance = 12
        args.threshold = 1.5
        args.max_frames = 10
        args.clip_duration = 1.5
        print("快速模式已启用")

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：未提供API Key", file=sys.stderr)
        sys.exit(1)

    episode_dir = args.episode_dir
    camera_id = args.camera_id
    grain = args.grain
    advance = args.advance
    threshold = args.threshold
    clip_duration = args.clip_duration
    max_frames = args.max_frames

    # 输出目录
    path_parts = episode_dir.split(os.path.sep)
    if len(path_parts) >= 4:
        output_dir = os.path.join("output", "action_commands", path_parts[-4], path_parts[-3], path_parts[-2], path_parts[-1])
    else:
        output_dir = os.path.join("output", "action_commands", episode_dir.replace(os.path.sep, "_"))
    os.makedirs(output_dir, exist_ok=True)
    out_path_en = os.path.join(output_dir, "action_commands.json")
    out_path_cn = os.path.join(output_dir, "action_commands_cn.json")

    info_path = os.path.join(episode_dir, f"camera{camera_id}_info.json")
    video_path = os.path.join(episode_dir, f"camera{camera_id}.mp4")

    if not os.path.exists(info_path) or not os.path.exists(video_path):
        print("错误：文件不存在", file=sys.stderr)
        sys.exit(1)

    # 获取视频信息
    with open(info_path, "r") as f:
        cam_info = json.load(f)
    fps = cam_info.get("fps") or get_video_info(video_path)[0]
    _, _, duration = get_video_info(video_path)
    
    print(f"视频信息: fps={fps:.2f}, 时长={duration:.2f}秒")
    print(f"配置: advance={advance}, threshold={threshold}s, clip_duration={clip_duration}s, max_frames={max_frames}")

    # 创建检测器
    detector = ActionDetector(video_path, api_key, args.base_url, clip_duration, max_frames)

    # 初始化
    print("处理第0秒...")
    first_action = detector.get_action_at(0.0)
    if first_action is None:
        print("错误：无法识别第一帧动作", file=sys.stderr)
        sys.exit(1)

    actions = []
    first_action["start_time"] = 0.0
    actions.append(first_action)
    
    prev_action = first_action
    prev_time = 0.0
    t = 0.0
    no_change_count = 0

    while t < duration:
        next_t = min(t + advance * grain, duration)
        print(f"\n当前时间 {t:.2f}s，跳至下一候选时间 {next_t:.2f}s")
        
        next_action = detector.get_action_at(next_t)
        
        if next_action is None:
            print("  无法识别，假设与上一动作相同")
            t = next_t
            prev_time = t
            no_change_count += 1
            continue

        if compare_actions(prev_action, next_action):
            print("  动作未变")
            no_change_count += 1
            # 如果连续多次无变化，加大步进
            if no_change_count >= SKIP_THRESHOLD:
                old_advance = advance
                advance = min(advance * 2, 20)  # 最大步进不超过20
                print(f"  连续{no_change_count}次无变化，步进从{old_advance}调整为{advance}")
            t = next_t
            prev_time = t
        else:
            print(f"  动作可能变化（{prev_action.get('verb')} -> {next_action.get('verb')}）")
            no_change_count = 0  # 重置计数
            
            # 二分查找变化点
            left = prev_time
            right = next_t
            mid_points = []
            
            # 预计算二分查找点
            while right - left > threshold:
                mid = (left + right) / 2
                mid_points.append(mid)
                # 根据比较结果更新边界
                # 这里我们先收集点，然后批量处理
                if (right - left) / 2 > threshold:
                    left = mid
                else:
                    right = mid
            
            if mid_points:
                # 批量获取中间点的动作
                mid_results = detector.get_actions_batch(mid_points)
                
                # 重新执行二分查找逻辑
                left = prev_time
                right = next_t
                for mid in mid_points:
                    mid_action = mid_results.get(mid)
                    if mid_action is None or compare_actions(prev_action, mid_action):
                        left = mid
                    else:
                        right = mid
                        break
            
            change_time = right
            print(f"  变化点确定为 {change_time:.2f}s")

            # 获取新动作
            new_action = detector.get_action_at(change_time)
            if new_action is None:
                new_action = next_action
            new_action["start_time"] = change_time

            actions[-1]["end_time"] = change_time
            actions.append(new_action)

            prev_action = new_action
            prev_time = change_time
            t = change_time

    # 设置最后一个动作的结束时间
    if actions:
        actions[-1]["end_time"] = duration

    # 输出结果
    actions_en, actions_cn = build_action_commands(actions)
    
    with open(out_path_en, "w", encoding="utf-8") as f:
        json.dump({"atomic_actions": actions_en}, f, indent=4, ensure_ascii=False)
    with open(out_path_cn, "w", encoding="utf-8") as f:
        json.dump({"atomic_actions": actions_cn}, f, indent=4, ensure_ascii=False)

    print(f"\n处理完成！")
    print(f"英文动作文件: {out_path_en}")
    print(f"中文动作文件: {out_path_cn}")
    print(f"总API调用次数: {len(detector.cache)}")

if __name__ == "__main__":
    main()