from pathlib import Path

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


def get_multi_labels(T, D, S, A, W, L):
    """
    多标签规则（可同时命中多个，返回标签列表）
    L in [1,2,4,6,8] → 灯光异常 → 0
    A in [1,2] → 有水藻 → 1
    W in [1,2] → 水体浑浊 → 2
    """
    labels = []
    L_num = int(L[1:]) if L is not None else None
    A_num = int(A[1:]) if A is not None else None
    W_num = int(W[1:]) if W is not None else None

    if L_num in {1, 2, 4, 6, 8}:
        labels.append(0)
    if A_num in {1, 2}:
        labels.append(1)
    if W_num in {1, 2}:
        labels.append(2)
    return labels


def process_dataset_dirs(root_dir_list):
    """
    root_dir_list: 每个目录下包含 images 文件夹
    标签txt输出到同级 labels 文件夹；
    无论有没有标签，每张图片都生成对应的txt文件（无标签则为空文件）
    """
    for dir_path_str in root_dir_list:
        root_dir = Path(dir_path_str)
        images_dir = root_dir / "images"
        labels_dir = root_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        if not images_dir.exists():
            print(f"跳过，不存在images文件夹: {images_dir}")
            continue

        img_suffix = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in img_suffix:
                continue

            T, D, S, A, W, L = parse_env_from_path(str(img_file))
            label_list = get_multi_labels(T, D, S, A, W, L)

            txt_path = labels_dir / (img_file.stem + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for lab in label_list:
                    f.write(f"{lab}\n")

            if len(label_list) > 0:
                print(f"生成标签{label_list} → {txt_path}")
            else:
                print(f"生成空标签文件 → {txt_path}")


if __name__ == "__main__":
    dataset_dirs = [
        r"H:\datasets\Basic_Single_Fish\T06\D1\S1\A0\W0\L1\20260716_F31F36F40F41",
        r"H:\datasets\Basic_Single_Fish\T06\D1\S1\A0\W0\L2\20260716_F31F36F40F41",
        # 在这里追加剩余目录
    ]
    process_dataset_dirs(dataset_dirs)
