import os
import cv2

# ====================== 【核心配置 - 仅此处手动修改】 ======================
# 待处理视频路径列表，多个视频用逗号分隔
VIDEO_LIST = [
    r"H:\datasets\Basic_Single_Fish\T06\D1\S0\A1\W1\L8\20260526_F3\2026052614_1779776048.mp4"
]

# 抽帧模式切换【核心规则】
# 👉 EXTRACT_FRAME_NUM > 0  : 均匀抽取【指定数量】图片帧
# 👉 EXTRACT_FRAME_NUM <= 0 : 逐帧抽取【视频所有有效帧】（全量保存）
EXTRACT_FRAME_NUM = 300

# 图片输出子文件夹（自动在视频同级目录生成）
OUTPUT_FOLDER = "images"

SAVE_SUFFIX = ".jpg"
SKIP_EXIST = True       # 跳过已生成图片，避免重复抽帧
# =============================================================================


def parse_env_from_path(full_video_path):
    """
    从视频绝对路径自动解析环境参数 T/D/S/A/W/L
    适配路径结构：.../Txx/Dx/Sx/Ax/Wx/Lx/日期_F编码/视频.mp4
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


def parse_fish_id_from_video(video_path):
    """
    从视频所在【上一级目录名】解析鱼种ID
    目录格式要求：日期_F编码 例：20260610_F3、20260611_F26F45
    :return: 解析成功返回 Fxxx，失败返回 None
    """
    video_parent_dir = os.path.dirname(video_path)
    dir_name = os.path.basename(os.path.normpath(video_parent_dir))

    if "_" not in dir_name:
        print(f"⚠️ 目录[{dir_name}] 格式异常：缺少下划线，无法解析鱼种ID")
        return None

    _, fish_code = dir_name.split("_", 1)
    if not fish_code.startswith("F"):
        print(f"⚠️ 目录[{dir_name}] 解析失败：[{fish_code}] 不是合法F开头鱼种编码")
        return None

    return fish_code


def save_frame(frame, save_path):
    """安全保存图片，高质量jpg压缩"""
    cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def extract_video_frame(video_path, save_dir, T, D, S, A, W, L, fish_id):
    """
    视频抽帧核心逻辑
    1. EXTRACT_FRAME_NUM > 0 ：均匀抽取指定帧数
    2. EXTRACT_FRAME_NUM <= 0：逐帧抽取全部有效帧（全量保存）
    """
    video_name = os.path.basename(video_path)
    video_key = os.path.splitext(video_name)[0]

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 视频打开失败：{video_name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_max_frame = total_frames - 1  # 有效帧：排除最后一帧

    if valid_max_frame <= 0:
        print(f"⚠️ 视频无有效帧，跳过：{video_name}")
        cap.release()
        return

    success_count = 0

    # ===================== 模式1：全帧抽取（EXTRACT_FRAME_NUM <= 0） =====================
    if EXTRACT_FRAME_NUM <= 0:
        print(f"ℹ️ 开始全量抽取所有有效帧，总有效帧数：{valid_max_frame + 1}")
        # 遍历所有有效帧 0 ~ valid_max_frame
        for frame_pos in range(valid_max_frame + 1):
            frame_serial = f"{frame_pos + 1:03d}"
            name_prefix = f"{T}_{D}_{S}_{A}_{W}_{L}_{fish_id}_{video_key}"
            save_path = os.path.join(save_dir, f"{name_prefix}_{frame_serial}{SAVE_SUFFIX}")

            # 跳过已存在文件
            if SKIP_EXIST and os.path.exists(save_path):
                success_count += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                save_frame(frame, save_path)
                print(f"📸 生成帧{frame_serial}：{os.path.basename(save_path)}")
                success_count += 1

    # ===================== 模式2：均匀抽取指定帧数（EXTRACT_FRAME_NUM > 0） =====================
    else:
        if valid_max_frame <= EXTRACT_FRAME_NUM:
            print(f"⚠️ 视频有效帧数不足，跳过抽帧：{video_name}")
            cap.release()
            return

        frame_index_list = []
        if EXTRACT_FRAME_NUM == 1:
            frame_index_list = [0]
        elif EXTRACT_FRAME_NUM == 2:
            # 固定规则：首帧 + 严格中间帧
            frame_index_list = [0, valid_max_frame // 2]
        else:
            # 多帧均匀采样
            interval = valid_max_frame / (EXTRACT_FRAME_NUM - 1)
            for i in range(EXTRACT_FRAME_NUM):
                frame_idx = int(i * interval)
                frame_index_list.append(frame_idx)

        for idx, frame_pos in enumerate(frame_index_list):
            # frame_serial = f"{idx + 1:03d}"
            frame_serial = f"{frame_pos + 1:03d}"
            name_prefix = f"{T}_{D}_{L}_{S}_{A}_{W}_{fish_id}_{video_key}"
            save_path = os.path.join(save_dir, f"{name_prefix}_{frame_serial}{SAVE_SUFFIX}")

            if SKIP_EXIST and os.path.exists(save_path):
                success_count += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                save_frame(frame, save_path)
                print(f"📸 生成帧{frame_serial}：{os.path.basename(save_path)}")
                success_count += 1

    cap.release()
    print(f"✅ {video_name} 处理完成，成功生成{success_count}张图片\n")


def batch_extract_by_list():
    """批量处理入口"""
    if not VIDEO_LIST:
        print("❌ VIDEO_LIST为空，请填写需要处理的视频路径！")
        return

    # 区分运行模式
    if EXTRACT_FRAME_NUM > 0:
        run_mode = "【均匀抽帧模式】"
        mode_tip = f"单视频均匀抽取 {EXTRACT_FRAME_NUM} 帧"
    else:
        run_mode = "【全帧抽取模式】"
        mode_tip = "抽取视频全部有效帧（逐帧保存）"

    print("=" * 60)
    print(f"🚀 开始批量处理视频 {run_mode}")
    print(f"📋 待处理视频总数：{len(VIDEO_LIST)}")
    print(f"ℹ️ 运行说明：{mode_tip}")
    print("=" * 60 + "\n")

    for video_path in VIDEO_LIST:
        video_path = video_path.strip()
        # 基础文件校验
        if not os.path.isfile(video_path):
            print(f"⚠️ 文件不存在，跳过：{video_path}\n")
            continue
        if not video_path.lower().endswith(".mp4"):
            print(f"⚠️ 非mp4格式，跳过：{os.path.basename(video_path)}\n")
            continue

        # 解析鱼种ID
        fish_id = parse_fish_id_from_video(video_path)
        if not fish_id:
            print(f"❌ 鱼种ID解析失败，跳过当前视频\n")
            continue

        # 创建图片输出目录
        video_parent_dir = os.path.dirname(video_path)
        output_images_path = os.path.join(video_parent_dir, OUTPUT_FOLDER)
        os.makedirs(output_images_path, exist_ok=True)

        print(f"▶️ 正在处理：{os.path.basename(video_path)}")
        print(f"🐟 自动解析鱼种ID：{fish_id}")
        print(f"📂 图片输出目录：{output_images_path}")

        # 解析环境参数 T/D/S/A/W/L
        T, D, S, A, W, L = parse_env_from_path(video_path)
        if not all([T, D, S, A, W, L]):
            print(f"⚠️ 环境参数T/D/S/A/W/L缺失，跳过该视频\n")
            continue

        # 执行抽帧
        extract_video_frame(video_path, output_images_path, T, D, S, A, W, L, fish_id)

    print("=" * 60)
    print("🎉 所有视频处理任务全部结束！")
    print("=" * 60)


if __name__ == "__main__":
    batch_extract_by_list()