import csv


def extract_pixel_events(csv_path, target_row, target_col, max_events=500):
    """
    读取事件相机CSV文件，提取指定像素点的前N个事件时间戳及间隔

    参数:
    csv_path (str): CSV文件路径
    target_row (int): 目标像素的行地址
    target_col (int): 目标像素的列地址
    max_events (int): 最大提取事件数 (默认500)

    返回:
    list: 包含元组(timestamp, interval)的列表，interval为与前一事件的时间差
    """
    timestamps = []

    # 逐行读取CSV，避免加载整个文件到内存
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # 跳过空行或不完整行
            if len(row) < 5:
                continue

            try:
                # 解析每列数据 (行, 列, 亮度, 极性, 时间戳)
                # 处理可能的浮点格式整数（如1.0e2）
                x = int(float(row[0]))
                y = int(float(row[1]))
                ts = float(row[4])

                # 匹配目标像素
                if x == target_row and y == target_col:
                    timestamps.append(ts)
                    # 提前终止：达到最大事件数
                    if len(timestamps) >= max_events:
                        break
            except (ValueError, IndexError):
                continue  # 跳过格式错误的行

    # 按时间戳排序（确保时序正确）
    timestamps.sort()

    # 计算时间间隔
    results = []
    for i, ts in enumerate(timestamps):
        if i == 0:
            interval = float('nan')  # 首个事件无间隔
        else:
            interval = ts - timestamps[i - 1]
        results.append((ts, interval))

    return results


def main():
    # ====== 在这里修改您的参数设置 ======
    CSV_FILE_PATH = "/data/zm/micro_dataset_shixiong/0.2778_1.csv"  # 替换为您的CSV文件路径
    TARGET_ROW = 188  # 目标像素的行地址
    TARGET_COL = 2  # 目标像素的列地址
    MAX_EVENTS = 500  # 最大提取事件数（不足时显示全部）
    # ==================================

    # 提取事件数据
    events = extract_pixel_events(CSV_FILE_PATH, TARGET_ROW, TARGET_COL, MAX_EVENTS)

    # 打印结果
    print(f"像素位置: ({TARGET_ROW}, {TARGET_COL}), 共找到 {len(events)} 个事件")
    print("-" * 60)
    print(f"{'序号':<6} | {'时间戳':<20} | {'间隔':<20}")
    print("-" * 60)

    for i, (ts, interval) in enumerate(events, 1):
        # 格式化时间戳：根据数值大小自动选择合适的小数位数
        ts_str = f"{ts:.3f}" if ts < 10000 else f"{ts:.0f}"
        interval_str = "N/A" if i == 1 else f"{interval:.3f}" if interval < 1000 else f"{interval:.1f}"
        print(f"{i:<6} | {ts_str:<20} | {interval_str:<20}")


if __name__ == "__main__":
    main()