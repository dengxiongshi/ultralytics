import os
import cv2

# ====================== 【核心配置 - 仅此处需要手动修改】 ======================
# 1. 数据集根目录（你的数据集顶层路径）
ROOT_DIR = r"H:\datasets\Basic_Single_Fish"

# 2. 当前批次鱼种编码
# 单鱼示例：F02
# 混养示例：F03F08、F02F05F12
FISH_ID = "F29F44F45"

# 3. 抽帧配置（可自定义帧数，支持2/3/4/任意帧数）
EXTRACT_FRAME_NUM = 2  # 按需修改：2=双帧、3=三帧、4=四帧...
SAVE_SUFFIX = ".jpg"
SKIP_EXIST = True  # 跳过已生成图片，避免重复抽帧


# =============================================================================

def parse_env_from_path(full_video_path):
    """
    从视频绝对路径自动解析环境参数 T/D/S/A/W/L
    适配路径结构：.../Txx/Dx/Sx/Ax/Wx/Lx/日期/视频.mp4
    """
    path_parts = full_video_path.replace("\\", "/").split("/")
    T = D = S = A = W = L = None

    for part in path_parts:
        if part.startswith("T") and len(part) == 3:
            T = part
        elif part.startswith("D"):
            D = part
        elif part.startswith("S"):
            S = part
        elif part.startswith("A"):
            A = part
        elif part.startswith("W"):
            W = part
        elif part.startswith("L"):
            L = part

    return T, D, S, A, W, L


def save_frame(frame, save_path):
    """安全保存图片，高质量jpg压缩"""
    cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def extract_video_frame(video_path, save_dir, T, D, S, A, W, L):
    """
    单个视频均匀抽帧：支持自定义2/3/4/N帧均匀采样
    首帧必取，剩余帧数均匀分布在视频中段，保证采样均衡
    """
    # 获取视频纯文件名（不含后缀）
    video_name = os.path.basename(video_path)
    video_key = os.path.splitext(video_name)[0]

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 视频打开失败：{video_name}")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= EXTRACT_FRAME_NUM:
        print(f"⚠️ 视频帧数不足，跳过：{video_name}")
        cap.release()
        return

    # 生成均匀抽帧索引（首帧固定+均匀间隔采样）
    frame_index_list = []
    if EXTRACT_FRAME_NUM == 1:
        frame_index_list = [0]
    else:
        # 均匀划分区间，生成对应帧数索引
        interval = total_frames / (EXTRACT_FRAME_NUM - 1)
        for i in range(EXTRACT_FRAME_NUM):
            frame_idx = int(i * interval)
            frame_index_list.append(frame_idx)

    # 批量抽帧并保存
    success_count = 0
    for idx, frame_pos in enumerate(frame_index_list):
        # 三位补零序号 001/002/003...
        frame_serial = f"{idx + 1:03d}"
        # 拼接标准图片名称
        name_prefix = f"{T}_{D}_{L}_{S}_{A}_{W}_{FISH_ID}_{video_key}"
        save_path = os.path.join(save_dir, f"{name_prefix}_{frame_serial}{SAVE_SUFFIX}")

        # 跳过已存在文件
        if SKIP_EXIST and os.path.exists(save_path):
            success_count += 1
            continue

        # 读取并保存帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            save_frame(frame, save_path)
            print(f"📸 生成帧{frame_serial}：{os.path.basename(save_path)}")
            success_count += 1

    cap.release()
    print(f"✅ {video_name} 抽帧完成，成功生成{success_count}张图片")


def batch_extract(root_path):
    """全局批量遍历所有视频，自动抽帧"""
    print("=" * 60)
    print(f"🚀 开始批量抽帧，当前鱼种：{FISH_ID}，单视频抽帧数：{EXTRACT_FRAME_NUM}")
    print("=" * 60)

    for root, _, files in os.walk(root_path):
        # 只遍历mp4视频
        for file in files:
            if file.lower().endswith(".mp4"):
                video_abs_path = os.path.join(root, file)
                # 解析环境参数
                T, D, S, A, W, L = parse_env_from_path(video_abs_path)
                if not all([T, D, S, A, W, L]):
                    print(f"⚠️ 环境参数缺失，跳过：{video_abs_path}")
                    continue
                # 抽帧并保存到视频同级目录
                extract_video_frame(video_abs_path, root, T, D, S, A, W, L)

    print("\n" + "=" * 60)
    print("🎉 全部视频抽帧完成！")
    print("=" * 60)


if __name__ == "__main__":
    batch_extract(ROOT_DIR)
