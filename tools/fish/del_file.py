import os


def batch_delete_files(root_dir: str, delete_formats: list):
    """
    递归删除多层子目录中指定后缀的文件
    :param root_dir: 顶层根目录
    :param delete_formats: 要删除的文件后缀列表，例如 [".jpg", ".txt", ".png"]
    """
    # 路径标准化校验
    root_path = os.path.abspath(root_dir)
    if not os.path.isdir(root_path):
        print(f"❌ 错误：目录不存在 → {root_path}")
        return

    # 安全提示
    print("=" * 55)
    print(f"📂 扫描根目录：{root_path}")
    print(f"🗑️  待删除格式：{delete_formats}")
    print("⚠️  文件删除后不可恢复，请谨慎确认！")
    print("=" * 55)

    # 二次确认防误删
    if input("\n输入【y】确认执行删除，其他键退出：").strip().lower() != "y":
        print("\n🚫 已取消删除操作")
        return

    # 统计计数
    success_count = 0
    fail_count = 0

    # 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_path):
        if not filenames:  # 跳过空文件夹
            continue

        for filename in filenames:
            file_lower = filename.lower()
            # 匹配指定后缀（大小写兼容）
            if any(file_lower.endswith(ext.lower()) for ext in delete_formats):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"✅ 已删除：{file_path}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ 删除失败：{file_path} | 原因：{str(e)}")
                    fail_count += 1

    # 最终结果统计
    print("\n" + "=" * 55)
    print("🎉 批量删除任务执行完成！")
    print(f"✅ 成功删除：{success_count} 个文件")
    print(f"❌ 删除失败：{fail_count} 个文件")
    print("=" * 55)


# ====================== 【使用示例：自由传参调用】 ======================
if __name__ == "__main__":
    # 1. 设置你的根目录
    TARGET_ROOT = r"H:\datasets\5\20260612_F3"

    # 2. 【灵活传参】传入要删除的格式列表，想删什么就写什么
    # 示例1：删除 jpg、jpeg
    # batch_delete_files(TARGET_ROOT, [".jpg", ".jpeg"])

    # 示例2：删除 txt、png、gif
    # batch_delete_files(TARGET_ROOT, [".txt", ".png", ".gif"])

    # 示例3：删除所有图片+文本（默认执行这个）
    batch_delete_files(TARGET_ROOT, [".jpg", ".jpeg", ".png", ".txt", ".gif"])