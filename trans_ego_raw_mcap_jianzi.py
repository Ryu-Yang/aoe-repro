import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from mcap.reader import make_reader

# 尝试导入 protobuf 相关模块
try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from google.protobuf.json_format import MessageToDict
    protobuf_available = True
except ImportError:
    protobuf_available = False

# 尝试导入 ROS2 解码（备用）
try:
    from mcap_ros2.decoder import decode_ros2_message
    ros2_available = True
except ImportError:
    ros2_available = False

# 缓存 protobuf 消息类
_proto_classes = {}

def decode_message(schema, data):
    """
    根据 schema 编码解码消息数据
    返回：对于 protobuf 返回消息对象；对于 JSON/ROS2 返回字典
    """
    encoding = schema.encoding.lower() if schema else 'unknown'
    if encoding == 'json':
        return json.loads(data.decode('utf-8'))
    elif encoding == 'ros2msg':
        if not ros2_available:
            raise ImportError("缺少 mcap_ros2 模块，无法解码 ROS2 消息")
        return decode_ros2_message(schema, data)
    elif encoding == 'protobuf':
        if not protobuf_available:
            raise ImportError("缺少 protobuf 模块，无法解码 protobuf 消息")
        return decode_protobuf_message(schema, data)
    else:
        raise ValueError(f"不支持的编码: '{schema.encoding if schema else 'unknown'}' (topic schema name: {schema.name if schema else 'None'})")

def decode_protobuf_message(schema, data):
    """
    动态解码 protobuf 消息
    """
    msg_type_name = schema.name
    pool = descriptor_pool.Default()
    key = (schema.id, msg_type_name)

    if key in _proto_classes:
        msg_class = _proto_classes[key]
    else:
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(schema.data)
        for f in fds.file:
            pool.Add(f)
        try:
            desc = pool.FindMessageTypeByName(msg_type_name)
        except KeyError:
            raise ValueError(f"无法在描述符池中找到消息类型: {msg_type_name}")
        msg_class = message_factory.GetMessageClass(desc)
        _proto_classes[key] = msg_class

    msg = msg_class()
    msg.ParseFromString(data)
    return msg

def extract_timestamp(msg_obj):
    """
    从解码后的消息对象中提取时间戳（用于排序）
    支持 protobuf 消息（header.timestamp）和字典
    """
    try:
        if hasattr(msg_obj, 'header') and hasattr(msg_obj.header, 'timestamp'):
            return msg_obj.header.timestamp
        elif hasattr(msg_obj, 'timestamp'):  # 某些消息直接包含 timestamp 字段
            return msg_obj.timestamp
        elif isinstance(msg_obj, dict):
            header = msg_obj.get('header', {})
            if isinstance(header, dict) and 'timestamp' in header:
                return header['timestamp']
            if 'timestamp' in msg_obj:
                return msg_obj['timestamp']
    except:
        pass
    return None  # 无法提取则使用消息的 publish_time（后面备选）

def ensure_mp4(h264_bytes, output_mp4_path):
    """
    将 H.264 裸流封装为 MP4 文件（调用系统 ffmpeg）。
    如果 ffmpeg 不可用或封装失败，返回 False 并保留原始裸流（文件名改为 .h264）。
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

    with tempfile.NamedTemporaryFile(suffix='.h264', delete=False) as tmp:
        tmp.write(h264_bytes)
        tmp_path = tmp.name

    try:
        cmd = ['ffmpeg', '-y', '-i', tmp_path, '-c', 'copy', '-f', 'mp4', output_mp4_path]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.SubprocessError:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def main():
    parser = argparse.ArgumentParser(description="解析 MCAP 文件，提取相机信息与压缩视频流")
    parser.add_argument("--file_path", required=True, help="输入的 MCAP 文件路径")
    args = parser.parse_args()

    file_path = args.file_path
    path_parts = file_path.split(os.path.sep)
    # 获取最后三级目录和文件名
    if len(path_parts) >= 4:
        # 提取目录部分
        third_level = path_parts[-4]  # 三级目录
        second_level = path_parts[-3] # 二级目录
        first_level = path_parts[-2]  # 一级目录
        
        # 获取文件名（不含扩展名）
        file_name_with_ext = path_parts[-1]
        file_name_without_ext = os.path.splitext(file_name_with_ext)[0]
        
        # 拼接输出路径
        output_dir = os.path.join("output", "raw_data", third_level, second_level, first_level, file_name_without_ext)
    else:
        # 处理路径层级不足的情况
        print("警告：文件路径层级不足，使用完整路径（去除扩展名）作为输出目录名")
        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join("output", "raw_data", dir_name.replace(os.path.sep, '_'), base_name)


    os.makedirs(output_dir, exist_ok=True)

    # 存储相机信息（每个相机只存第一个）
    camera_info = {}
    # 存储相机视频数据（每个相机存所有消息的 (timestamp, data) 列表）
    camera_videos = defaultdict(list)

    with open(args.file_path, "rb") as f:
        reader = make_reader(f)

        for schema, channel, message in reader.iter_messages():
            topic = channel.topic

            # --- 相机信息话题 (CameraInfo) ---
            info_match = re.match(r"^/robot0/sensor/camera(\d+)/camera_info$", topic)
            if info_match:
                cam_idx = info_match.group(1)
                if cam_idx in camera_info:
                    continue  # 只取第一条
                try:
                    msg_obj = decode_message(schema, message.data)
                    if hasattr(msg_obj, 'DESCRIPTOR'):
                        msg_dict = MessageToDict(msg_obj, preserving_proto_field_name=True)
                    else:
                        msg_dict = msg_obj
                    camera_info[cam_idx] = msg_dict
                except Exception as e:
                    print(f"处理相机信息话题 {topic} 时出错: {e}")
                continue

            # --- 压缩视频话题 (CompressedImage) ---
            video_match = re.match(r"^/robot0/sensor/camera(\d+)/compressed$", topic)
            if video_match:
                cam_idx = video_match.group(1)
                try:
                    msg_obj = decode_message(schema, message.data)

                    # 提取 data 字段
                    if hasattr(msg_obj, 'data'):
                        data_field = msg_obj.data
                    elif isinstance(msg_obj, dict):
                        data_field = msg_obj.get("data")
                    else:
                        raise TypeError("未知的消息类型，无法提取 data 字段")

                    if data_field is None:
                        print(f"警告: 话题 {topic} 中未找到 'data' 字段")
                        continue

                    # 转换为字节流
                    if isinstance(data_field, bytes):
                        video_bytes = data_field
                    elif isinstance(data_field, (list, tuple)):
                        video_bytes = bytes(data_field)
                    else:
                        raise TypeError(f"不支持的 data 字段类型: {type(data_field)}")

                    # 提取时间戳（用于排序）
                    ts = extract_timestamp(msg_obj)
                    if ts is None:
                        # 使用 MCAP 消息的 publish_time 作为备选（纳秒）
                        ts = message.publish_time

                    camera_videos[cam_idx].append((ts, video_bytes))

                except Exception as e:
                    print(f"处理视频话题 {topic} 时出错: {e}")
                continue

    # --- 保存相机信息 ---
    for cam_idx, msg_dict in camera_info.items():
        out_path = os.path.join(output_dir, f"camera{cam_idx}_info.json")
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(msg_dict, jf, indent=2, ensure_ascii=False)
        print(f"已保存相机信息: {out_path}")

    # --- 处理视频数据（按时间戳排序后拼接）---
    for cam_idx, entries in camera_videos.items():
        if not entries:
            continue

        # 按时间戳排序
        entries.sort(key=lambda x: x[0])

        # 拼接所有 data
        total_bytes = b''.join(byte_data for _, byte_data in entries)

        # 输出文件路径
        mp4_path = os.path.join(output_dir, f"camera{cam_idx}.mp4")
        h264_path = os.path.join(output_dir, f"camera{cam_idx}.h264")

        # 尝试封装为 MP4
        if ensure_mp4(total_bytes, mp4_path):
            print(f"已保存视频文件: {mp4_path} (共 {len(entries)} 条消息)")
        else:
            # 回退保存裸流
            with open(h264_path, "wb") as vf:
                vf.write(total_bytes)
            print(f"ffmpeg 不可用或封装失败，已保存裸流: {h264_path} (共 {len(entries)} 条消息)")
            print(f"  提示：可使用命令封装为 MP4：ffmpeg -i {h264_path} -c copy {mp4_path}")

    print("处理完成。")

if __name__ == "__main__":
    main()