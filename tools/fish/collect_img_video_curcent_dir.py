import os
import cv2

# ====================== 【核心配置 - 仅此处需要手动修改】 ======================
# 1. 固定需要处理的目标视频目录（末尾目录格式要求：日期_F编码，例：20260610_F3、20260610_F26F45）
TARGET_PATH = r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W0\L3\20260610_F50"

# 2. 图片输出子文件夹名称（自动在目标路径下生成）
OUTPUT_FOLDER = "images"

# 3. 抽帧配置（可自定义帧数，支持2/3/4/任意帧数）
EXTRACT_FRAME_NUM = 5  # 按需修改：2=双帧、3=三帧、4=四帧...
SAVE_SUFFIX = ".jpg"
SKIP_EXIST = True      # 跳过已生成图片，避免重复抽帧
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


def extract_video_frame(video_path, save_dir, T, D, S, A, W, L, fish_id):
    """
    单个视频均匀抽帧：支持自定义2/3/4/N帧均匀采样
    首帧必取，剩余帧数均匀分布在视频中段，保证采样均衡
    :param fish_id: 自动解析得到的鱼种编码
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
    # 有效区间：0 ~ 倒数第2帧（精准排除最后1帧，不裁剪任何有效画面）
    valid_max_frame = total_frames - 1

    if valid_max_frame <= EXTRACT_FRAME_NUM:
        print(f"⚠️ 视频有效帧数不足，跳过：{video_name}")
        cap.release()
        return

    # ========== 均匀采样逻辑 ==========
    frame_index_list = []
    if EXTRACT_FRAME_NUM == 1:
        frame_index_list = [0]
    elif EXTRACT_FRAME_NUM == 2:
        # 固定规则：首帧 + 严格中间帧
        frame_index_list = [0, valid_max_frame // 2]
    else:
        # 多帧均匀采样：覆盖从开头到倒数第二帧的全部有效画面
        interval = valid_max_frame / (EXTRACT_FRAME_NUM - 1)
        for i in range(EXTRACT_FRAME_NUM):
            frame_idx = int(i * interval)
            frame_index_list.append(frame_idx)

    # 批量抽帧并保存
    success_count = 0
    for idx, frame_pos in enumerate(frame_index_list):
        # 三位补零序号 001/002/003...
        frame_serial = f"{idx + 1:03d}"
        # 拼接标准图片名称（使用自动解析的fish_id）
        name_prefix = f"{T}_{D}_{S}_{A}_{W}_{L}_{fish_id}_{video_key}"
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


def parse_fish_id_from_target_path(root_path):
    """
    从目标路径最后一级目录解析鱼种ID
    规则：目录名格式 = 日期_F编码，例：20260610_F3、20260610_F26F45
    :return: 解析出的 fish_id，解析失败返回 None
    """
    # 获取路径最后一级目录名称（自动兼容 Windows/Linux 分隔符）
    last_dir = os.path.basename(os.path.normpath(root_path))
    if "_" not in last_dir:
        print(f"❌ 路径格式错误：[{last_dir}] 缺少下划线，无法解析鱼种ID")
        return None

    # 仅分割第一个下划线，后半段即为鱼种编码
    _, fish_candidate = last_dir.split("_", 1)
    # 校验必须以 F 开头（业务格式约束）
    if not fish_candidate.startswith("F"):
        print(f"❌ 解析失败：[{fish_candidate}] 不是合法鱼种编码（需以 F 开头）")
        return None

    return fish_candidate


def batch_extract():
    """仅处理指定目录视频，自动抽帧并保存至images文件夹"""
    # 1. 校验目标路径是否存在
    if not os.path.exists(TARGET_PATH):
        print(f"❌ 目标路径不存在：{TARGET_PATH}")
        return

    # 2. 自动解析鱼种ID（核心新增逻辑）
    fish_id = parse_fish_id_from_target_path(TARGET_PATH)
    if not fish_id:
        return  # 解析失败，直接退出

    # 3. 构建图片输出目录，不存在则自动创建
    output_images_path = os.path.join(TARGET_PATH, OUTPUT_FOLDER)
    os.makedirs(output_images_path, exist_ok=True)

    # 4. 打印运行信息
    print("=" * 60)
    print(f"🚀 开始批量抽帧")
    print(f"📂 处理目录：{TARGET_PATH}")
    print(f"🖼️  输出目录：{output_images_path}")
    print(f"🐟 自动解析鱼种ID：{fish_id}，单视频抽帧数：{EXTRACT_FRAME_NUM}")
    print("=" * 60)

    # 5. 遍历目标目录下所有mp4视频
    file_list = os.listdir(TARGET_PATH)
    for file in file_list:
        if file.lower().endswith(".mp4"):
            video_abs_path = os.path.join(TARGET_PATH, file)
            # 解析环境参数 T/D/S/A/W/L
            T, D, S, A, W, L = parse_env_from_path(video_abs_path)
            if not all([T, D, S, A, W, L]):
                print(f"⚠️ 环境参数缺失，跳过：{file}")
                continue
            # 抽帧（传入自动解析的鱼种ID）
            extract_video_frame(video_abs_path, output_images_path, T, D, S, A, W, L, fish_id)

    print("\n" + "=" * 60)
    print("🎉 指定目录所有视频抽帧完成！")
    print("=" * 60)


if __name__ == "__main__":
    batch_extract()