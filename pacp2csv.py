#coding=utf-8
import os
import pandas as pd
import global_path
import numpy as np
import subprocess
import re 

np.random.seed(222) 

# 处理 tshark 命令输出
def parse_tshark_conv_output(output):
    lines = output.strip().split('\n')
    data = []

    conv_pattern = re.compile(
        r"^\s*([\d.:]+(?:\[\d+\])?)\s*<->\s*([\d.:]+(?:\[\d+\])?)\s*\|"
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|\|" 
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|\|"
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|" 
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$" 
    )

    parsing = False 
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "=====" in line: 
            parsing = not parsing
            continue

        if parsing and "<->" in line and "Frames" not in line:
            match = conv_pattern.match(line)
            if match:
                # Ckient A, Server B 
                # Duration, Out_Packets (A->B), Out_Bytes (A->B), In_Packets (B->A), In_Bytes (B->A)
                duration = float(match.group(9))    # Duration
                out_packets = int(match.group(3))   # A->B Packets
                out_bytes = int(match.group(4))     # A->B Bytes
                in_packets = int(match.group(5))    # B->A Packets
                in_bytes = int(match.group(6))      # B->A Bytes
                data.append([duration, out_packets, out_bytes, in_packets, in_bytes])
            else:
                print(f"警告: 无法解析行: {line}")

    return data

# 提取特征函数
def extract_features_from_pcap(path):
    files = [f for f in os.listdir(path) if f.lower().endswith(('.pcap', '.pcapng'))]
    if not files:
        print(f"在目录 {path} 中未找到 pcap 文件。")
        return pd.DataFrame()

    try:
        subprocess.run(['tshark', '-v'], capture_output=True, text=True, check=True, shell=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: tshark 命令未找到或执行失败。")
        return pd.DataFrame() # tshark 不可用则返回空 DataFrame

    cmd_template = 'tshark -r "{pcap}" -q -z conv,tcp'
    all_features = []
    feature_columns = ['Duration', 'Out_Packets', 'Out_Bytes', 'In_Packets', 'In_Bytes']

    print(f"开始处理目录: {path}")
    for file in files:
        pcap_file_path = os.path.join(path, file)
        cmd = cmd_template.format(pcap=pcap_file_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', shell=True, check=False) # check=False 允许我们手动检查返回码

            if result.returncode != 0:
                print(f"处理文件 {file} 时运行 tshark 出错。返回码: {result.returncode}")
                if result.stderr:
                    print(f"tshark 错误信息: {result.stderr.strip()}")
                continue

            features = parse_tshark_conv_output(result.stdout)
            if features:
                all_features.extend(features)
                print(f"  - 已处理 {file}, 提取了 {len(features)} 条流。")
            else:
                print(f"  - 已处理 {file}, 未提取到有效的 TCP 流。")

        except FileNotFoundError:
             print(f"错误: 无法找到文件 {pcap_file_path}。请检查路径是否正确。")
             continue
        except Exception as e:
            print(f"处理文件 {file} 时发生未知错误: {e}")
            continue 

    if not all_features:
        print(f"在目录 {path} 的所有 pcap 文件中均未找到有效的特征。")
        return pd.DataFrame(columns=feature_columns)

    sum_df = pd.DataFrame(all_features, columns=feature_columns)
    print(f"目录 {path} 处理完成，共提取 {len(sum_df)} 条流。")
    return sum_df

# 保存CSV
def add_label_and_save(dataframe, input_path, output_csv_path):
    if dataframe.empty:
        print(f"数据框为空，跳过为 {input_path} 添加标签和保存。")
        return

    label = -1 
    if input_path == global_path.white_path:
        label = 0 
    elif input_path == global_path.black_path:
        label = 1 
    else:
        print(f"警告: 未知的输入路径类型 {input_path}，无法确定标签。将分配标签 -1。")

    dataframe['Label'] = label

    output_columns = ['Duration', 'Out_Packets', 'Out_Bytes', 'In_Packets', 'In_Bytes', 'Label']
    dataframe = dataframe[output_columns]

    print(f"为来自 {input_path} 的 {len(dataframe)} 条流添加标签 {label}。")
    # print(dataframe.head()) # 可选：供验证

    file_exists = os.path.exists(output_csv_path)

    try:
        dataframe.to_csv(output_csv_path,
                         index=False,
                         header=not file_exists,
                         mode='a',
                         encoding='utf-8')
        print(f"数据已成功追加到 {output_csv_path}")
    except IOError as e:
        print(f"错误: 无法写入 CSV 文件 {output_csv_path}。请检查权限或路径。错误信息: {e}")
    except Exception as e:
        print(f"保存到 CSV 时发生未知错误: {e}")


if __name__ == '__main__':
    output_csv_file = 'processed_flow_features.csv'
    
    print(f"脚本开始执行，输出文件为: {output_csv_file}")

    print(f"\n===== 开始处理白名单流量 =====")
    if hasattr(global_path, 'white_path') and global_path.white_path:
        print(f"白名单 Pcap 目录: {global_path.white_path}")
        white_df = extract_features_from_pcap(global_path.white_path)
        add_label_and_save(white_df, global_path.white_path, output_csv_file)
    else:
        print("警告: 未在 global_path 中找到有效的 white_path 配置。")

    print(f"\n===== 开始处理黑名单流量 =====")
    if hasattr(global_path, 'black_path') and global_path.black_path:
        print(f"黑名单 Pcap 目录: {global_path.black_path}")
        black_df = extract_features_from_pcap(global_path.black_path)
        add_label_and_save(black_df, global_path.black_path, output_csv_file)
    else:
        print("警告: 未在 global_path 中找到有效的 black_path 配置。")

    print(f"\n===== 处理完成 =====")
    if os.path.exists(output_csv_file):
        try:
            final_df = pd.read_csv(output_csv_file)
            print(f"最终数据已保存到: {output_csv_file}")
            print(f"总共提取并保存的流数量: {len(final_df)}")
            if 'Label' in final_df.columns:
                print("标签分布情况:")
                print(final_df['Label'].value_counts())
            else:
                print("警告: 输出文件中缺少 'Label' 列。")
        except pd.errors.EmptyDataError:
            print(f"输出文件 {output_csv_file} 为空。")
        except Exception as e:
            print(f"读取最终 CSV 文件时发生错误: {e}")
    else:
        print("未生成任何输出文件。")

    print("\n脚本执行结束。")
