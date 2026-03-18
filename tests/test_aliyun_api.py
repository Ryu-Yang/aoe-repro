import os
import time
import shutil
import tempfile
from PIL import Image
from openai import OpenAI
import base64

def resize_images(image_paths, target_size=(640, 480), output_dir=None):
    """
    将指定图像列表调整为目标尺寸，并保存到临时目录。
    返回新路径列表和临时目录路径。
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    new_paths = []
    for path in image_paths:
        with Image.open(path) as img:
            # 直接拉伸到目标尺寸（如需保持比例，可改用 thumbnail + 填充）
            img_resized = img.resize(target_size)
            # 生成新文件名
            base = os.path.basename(path)
            name, ext = os.path.splitext(base)
            new_path = os.path.join(output_dir, f"{name}_resized{ext}")
            img_resized.save(new_path)
            new_paths.append(new_path)
    return new_paths, output_dir

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 原始图像列表
original_images = [
    "images/football1.jpg",
    "images/football2.jpg",
    "images/football3.jpg",
    "images/football4.jpg"
]

# 1. 调整图像分辨率
resized_images, temp_dir = resize_images(original_images)
print("图像已调整为 640x480，临时目录：", temp_dir)

base64_image1 = encode_image(resized_images[0])
base64_image2 = encode_image(resized_images[1])
base64_image3 = encode_image(resized_images[2])
base64_image4 = encode_image(resized_images[3])

# 2. 初始化 OpenAI 客户端（阿里云 DashScope 兼容模式）
client = OpenAI(
    api_key="sk-sb250114514",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# images_path = [f"file://{img}" for img in original_images]

print("开始请求...")

# 3. 记录开始时间
start_time = time.time()

# 4. 调用 API
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [
                        f"data:image/jpeg;base64,{base64_image1}",
                        f"data:image/jpeg;base64,{base64_image2}",
                        f"data:image/jpeg;base64,{base64_image3}",
                        f"data:image/jpeg;base64,{base64_image4}",
                    ],
                    "fps": 2
                },
                {
                    "type": "text",
                    "text": "描述这个视频的具体过程"
                }
            ]
        }
    ]
)

# 5. 记录结束时间
end_time = time.time()
elapsed = end_time - start_time

# 6. 输出结果和统计信息
print(f"\n请求耗时：{elapsed:.2f} 秒")

# 检查响应中是否包含 token 使用信息（OpenAI 标准格式）
if hasattr(completion, 'usage') and completion.usage:
    usage = completion.usage
    print(f"提示 tokens：{usage.prompt_tokens}")
    print(f"生成 tokens：{usage.completion_tokens}")
    print(f"总计 tokens：{usage.total_tokens}")
else:
    print("响应中未包含 token 消耗信息")

# 打印模型返回的描述
print("\n模型描述：")
print(completion.choices[0].message.content)

# 7. 清理临时文件
shutil.rmtree(temp_dir)
print(f"\n临时目录 {temp_dir} 已清理。")