'''<Intruction> Time: 2026/06/17
train --> merge --> eval --> compared_plot -->logPlot
多视频文件, 输出多任务学习caption;
继承自 Qwen3_vl_2B_multi_CoVAL_data_creater_V2.2.py;
增加车辆速度 velocities_calib 字段;
增加内外参矩阵 extrinsic_matrix, intrinsic_matrix 字段;
dataset_num15V3
|-->diy_dataset/my_multimodal_dataset_num15V3
|-->qwen3_lora_num15V320
|-->unsloth_grpo_outputs_num15V320
'''

import os
import json
import cv2
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

def load_jsonl_by_frame_id(path):
    """通用函数：加载 CoVLA 格式的 .jsonl 文件（每行 {"frame_id": data})"""
    data = {}
    if not os.path.exists(path):
        print(f"警告：文件不存在 {path}")
        return data
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry:
                    frame_id = next(iter(entry))  # 获取第一个 key (如 "0", "1")
                    data[frame_id] = entry[frame_id]
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 in {path}: {e}")
                continue
    return data

def process_single_video(video_id, video_filename, base_dir="./CoVLA-Dataset", base_dir_output="./dataset"):
    """处理单个视频文件并返回数据集条目列表"""
    # 构建路径
    video_path = os.path.join(base_dir, "videos", video_filename)
    jsonl_path = os.path.join(base_dir, "states", f"{video_id}.jsonl")
    captions_path = os.path.join(base_dir, "captions", f"{video_id}.jsonl")
    frame_dir = os.path.join(base_dir_output, "frames", video_id)  # 每个视频单独的帧目录
    
    # 创建输出目录
    os.makedirs(frame_dir, exist_ok=True)
    
    # 加载数据
    print(f"\n加载轨迹数据: {jsonl_path}")
    states_data = load_jsonl_by_frame_id(jsonl_path)
    for k in list(states_data.keys()):
        try:
            states_data[k]["extrinsic_matrix"] = np.array(states_data[k]["extrinsic_matrix"])
            states_data[k]["intrinsic_matrix"] = np.array(states_data[k]["intrinsic_matrix"])
        except Exception as e:
            print(f"矩阵转换错误 (frame {k}): {e}")
            continue
    print(f"加载了 {len(states_data)} 帧的轨迹数据")
    
    print(f"加载字幕数据: {captions_path}")
    captions_data = load_jsonl_by_frame_id(captions_path)
    print(f"加载了 {len(captions_data)} 帧的字幕数据")
    
    # 处理视频
    print(f"处理视频: {video_path}")
    try:
        video_reader = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"视频读取错误 {video_path}: {e}")
        return []
        
    total_frames = len(video_reader)
    print(f"总帧数: {total_frames}")
    
    dataset_entries = []
    
    # 定义坐标转换函数
    def device_to_camera(p_device, extrinsic):
        p_device = np.array(p_device + [1.0])
        p_camera = extrinsic @ p_device
        return p_camera[:3]

    def camera_to_image(p_camera, intrinsic):
        p_image = intrinsic @ p_camera
        return p_image[:2] / p_image[2]

    def draw_trajectory_on_frame(frame_bgr, trajectory, extrinsic, intrinsic, color=(0, 0, 255), radius=2):
        h, w = frame_bgr.shape[:2]
        for p in trajectory:
            p_cam = device_to_camera(p, extrinsic)
            if p_cam[2] <= 0:
                continue
            x, y = camera_to_image(p_cam, intrinsic)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_bgr, (int(x), int(y)), radius=radius, color=color, thickness=-1)
        return frame_bgr
    
    # 处理每一帧
    for frame_idx in tqdm(range(total_frames), desc=f"处理 {video_id}"):
        try:
            # 读取帧 (RGB → BGR)
            frame_rgb = video_reader[frame_idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            frame_key = str(frame_idx)

            # 提取车辆速度
            velocities_calib = states_data.get(frame_key)["velocities_calib"]
            extrinsic_matrix = states_data.get(frame_key)["extrinsic_matrix"]
            intrinsic_matrix = states_data.get(frame_key)["intrinsic_matrix"]

            # --- 轨迹处理 ---
            trajectory_data = states_data.get(frame_key)
            if trajectory_data:
                # 生成轨迹 caption
                traj_points = trajectory_data["trajectory"]
                points_str = [f"({x:.3f}, {y:.3f}, {z:.3f})" for x, y, z in traj_points]

                num_points = len(traj_points)
                if num_points <= 6:
                    # 点数不足6个，取全部
                    selected_points = traj_points
                    selected_points_str = points_str
                elif num_points % 6 == 0:
                    # 点数能被6整除，以相同间隔取6个点
                    interval = num_points // 6
                    selected_points = traj_points[::interval]
                    selected_points_str = points_str[::interval]
                else:
                    # 点数超过6个且不能被6整除，取前6个点
                    selected_points = traj_points[:6]
                    selected_points_str = points_str[:6]

                traj_text = ", ".join(selected_points_str)
                # traj_caption = f"Trajectory: {traj_text}"
                traj_caption = f"{traj_text}"

                frame_bgr = draw_trajectory_on_frame(
                    frame_bgr,
                    trajectory_data["trajectory"],  # 注意：这里仍然使用完整轨迹进行绘制，而非筛选后的点，以保持视觉效果；如果需要仅绘制筛选后的点，请替换为selected_points
                    # selected_points,  # 使用筛选后的点进行绘制
                    trajectory_data["extrinsic_matrix"],
                    trajectory_data["intrinsic_matrix"],
                )

            else:
                traj_caption = "No trajectory data available."
            
            # --- 保存带轨迹的帧 ---
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)
            
            # --- 获取 caption 字段（平铺）---
            caption_fields = captions_data.get(frame_key, {})
            
            # --- 构建多任务数据集条目 ---
            entry = {
                "image": frame_path,
                "caption": {
                    "Trajectory": traj_caption,
                    "plain_caption": caption_fields.get("plain_caption", ""),
                    "rich_caption": caption_fields.get("rich_caption", ""),
                    "risk": caption_fields.get("risk", "")
                },
                "frame_id": frame_idx,
                "video_id": video_id,
                "velocities_calib": velocities_calib,
                "extrinsic_matrix": str(extrinsic_matrix),
                "intrinsic_matrix": str(intrinsic_matrix),
                # 保留所有其他字段(除plain_caption/rich_caption/risk外)
                **{k: v for k, v in caption_fields.items() 
                   if k not in ["plain_caption", "rich_caption", "risk"]}
            }
            dataset_entries.append(entry)
            
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {e}")
            continue
    
    print(f"视频 {video_id} 处理完成，生成 {len(dataset_entries)} 条记录")
    return dataset_entries

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    '''
    需要修改的参数: num_videos; base_dir_output;
    '''
    num_videos = 15  # 处理的视频文件数量, < select-code:3-1 >
    version = 3     # 数据集版本,  < select-code:3-2 >
    base_dir_output=f"./dataset_num{num_videos}V{version}"   # 输出目录
    output_json = os.path.join(base_dir_output, "json_dataset.json")  #./dataset/json_dataset.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    base_dir = "/media/ly/WD3T/python312/CoVLA-Dataset"
    video_dir = os.path.join(base_dir, "videos")

    # 检查路径是否存在
    if not os.path.exists(video_dir):
        print(f"错误：路径 {video_dir} 不存在")
        exit(1)
    
    # 获取所有MP4文件
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    mp4_files.sort()
    
    # 处理的视频文件数量
    mp4_files = mp4_files[:num_videos]
    print(f"视频文件列表: {mp4_files}")
    
    ''''0a0fc7a5db365174.mp4'右转, '0a3b9a6430c0b2b9.mp4'直后左转, '0a4d643a26019390.mp4'左转, '0a7f851478603188.mp4'左转, '0a015f7a7ab0132c.mp4'左变道停,
    '0a16ff0a09d37024.mp4'直跟车, '0a24b82ce0cda031.mp4'左弯跟车, '0a93cf2ea5908d2e.mp4'右超车, '0a9633d92f1b92fb.mp4'左转, '0a7114103869cbab.mp4'起步右转,
    '0aa280b1c05a33db.mp4'直红停, '0ab2aafa77bf7dec.mp4'红起步右, '0c0fcb35ace08f53.mp4'会车右转, '0cd6738d1487ba04.mp4'匝道左右转, '0cff85a4c974ffb4.mp4'行人左转'''

    # 覆盖上一个 mp4_files，重新制定 mp4_files
    # < select-code:3-3 >
    mp4_files = ['0a0fc7a5db365174.mp4', '0a3b9a6430c0b2b9.mp4', '0a4d643a26019390.mp4', '0a7f851478603188.mp4', '0a015f7a7ab0132c.mp4',
                 '0a16ff0a09d37024.mp4', '0a24b82ce0cda031.mp4', '0a93cf2ea5908d2e.mp4', '0a9633d92f1b92fb.mp4', '0a7114103869cbab.mp4',
                 '0aa280b1c05a33db.mp4', '0ab2aafa77bf7dec.mp4', '0c0fcb35ace08f53.mp4', '0cd6738d1487ba04.mp4', '0cff85a4c974ffb4.mp4']
    
    # mp4_files = ['0a0fc7a5db365174.mp4', '0a3b9a6430c0b2b9.mp4', '0a4d643a26019390.mp4', '0a7f851478603188.mp4', '0cff85a4c974ffb4.mp4']

    # 移除.mp4后缀获取ID
    video_ids = [os.path.splitext(f)[0] for f in mp4_files]
    print(f"视频ID列表: {video_ids}")
    
    print(f"找到 {len(mp4_files)} 个视频文件，将处理前 {num_videos} 个:")
    for i, vid in enumerate(video_ids, 1):
        print(f"{i}. {vid}")
    
    # 收集所有数据集条目
    all_dataset_entries = []
    
    # 处理每个视频
    for i, (video_id, video_filename) in enumerate(zip(video_ids, mp4_files), 1):
        print(f"\n{'='*50}")
        print(f"处理视频 {i}/{num_videos}: {video_id}")
        print(f"{'='*50}")
        
        entries = process_single_video(video_id, video_filename, base_dir, base_dir_output)
        all_dataset_entries.extend(entries)
    
    # 保存最终数据集（追加模式：先读取已有数据再合并）
    existing_data = []
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            print(f"加载了 {len(existing_data)} 条现有记录")
        except Exception as e:
            print(f"读取现有JSON文件时出错: {e}")
    
    # 合并新旧数据
    combined_data = existing_data + all_dataset_entries
    print(f"总记录数: {len(combined_data)} (新增 {len(all_dataset_entries)} 条)")
    
    # 保存到JSON文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"处理完成！")
    print(f"总视频数: {num_videos}")
    print(f"总帧数: {len(combined_data)}")
    print(f"结果保存至: {output_json}")
    print(f"帧图像保存至: {base_dir_output}/frames/")
    print(f"{'='*50}")
    
    # 打印示例
    if combined_data:
        print("\n示例条目 (第一条记录):")
        print(json.dumps(combined_data[0], indent=2, ensure_ascii=False))


    # ======================
    # 步骤5: 保存csv数据集
    # ======================
    # 将"image": frame_path 和 "Trajectory": traj_caption 保存到csv文件
    import pandas as pd
    OUTPUT_CSV = os.path.join(base_dir_output , f"num{num_videos}_trajectory.csv")
    df = pd.DataFrame(all_dataset_entries)

    # 提取嵌套的Trajectory值
    df["Trajectory"] = df["caption"].apply(lambda x: x["Trajectory"])

    # 先重命名列名
    df.rename(columns={"image": "image_path", "Trajectory": "answer", "velocities_calib": "velocities_calib_str", 
                    "extrinsic_matrix": "extrinsic_matrix_str", "intrinsic_matrix": "intrinsic_matrix_str"}, inplace=True)

    # 再保存到CSV
    df[["image_path", "answer", "velocities_calib_str", "extrinsic_matrix_str", "intrinsic_matrix_str"]].to_csv(OUTPUT_CSV, index=False)

    print(f"Trajectory-annotated frames saved to: {OUTPUT_CSV}")