import os
import json
import pandas as pd

def summarize_results(results_dir_list, output_csv):
    summary_data = []
    for results_dir in results_dir_list:
        if not os.path.exists(results_dir):
            print(f"Warning: {results_dir} does not exist. Skipping.")
            continue

        # 遍历子文件夹
        for subdir in os.listdir(results_dir):
            # print(f"Processing {subdir} in {results_dir}...")
            subdir_path = os.path.join(results_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # 假设每个子目录下有一个 metrics.json 文件
            history = os.path.join(subdir_path, "history.json")
            if not os.path.exists(history):
                print(f"Warning: {history} does not exist in {subdir_path}. Skipping.")
                continue

            # 读取 history.json 文件
            with open(history, "r") as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {history} is not a valid JSON file. Skipping.")
                    continue

            results_orig = metrics.get("results_orig", {})

            # print(results_orig)

            def r3(x):
                return round(x, 3) if x is not None else None


            summary_data.append({
                "Model": subdir,
                "MAE": r3(results_orig.get("mae", None)),
                "RMSE": r3(results_orig.get("rmse", None)),
                "R2": r3(results_orig.get("r2", None))
            })

    # 将总结数据保存到 CSV 文件
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary saved to {output_csv}")



if __name__ == "__main__":
    results_dirs = ["/home/huangzy/uplc-method-recommendation/results/dl/20260212"]
    output_csv = "/home/huangzy/uplc-method-recommendation/results/20260212/summary.csv"
    summarize_results(results_dirs, output_csv)