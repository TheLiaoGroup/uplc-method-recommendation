import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_absolute_error


# ====== 绘图参数配置 ======
PLOT_CONFIG = {
    'default': {
        'FIGSIZE': (6, 6),
        'LINEWIDTH': 1,
        'LABELSIZE': 10,
        'TITLESIZE': 12,
        'SAVE_DPI': 600,
        'SAVE_BBOX': 'tight',
        'SCATTER_SIZE': 50,
        'ALPHA': 0.6,
        'TICK_LENGTH': 6,
        'TICK_WIDTH': 1.2,
        'EDGE_COLOR': 'k',
        'IPHONE_COLORS': {
            "scatter": "#007AFF",
            "line": "#FF3B30",
            "residual": "#34C759",
            "text": "#1C1C1E",
            "train": "#007AFF",
            "val": "#FF3B30"
        }
    },
    'training_curves': {
        'XLABEL': 'Epoch (every 10)',
        'YLABEL': 'Loss (log scale)',
        # 如需特殊配置可在此覆盖default，如：
        # 'FIGSIZE': (8, 6)
    },
    'prediction_plot': {
        'XLABEL': 'True Retention Time (s)',
        'YLABEL': 'Predicted Retention Time (s)',
        # 如需特殊配置可在此覆盖default
    },
    'residual_plot': {
        'XLABEL': 'Predicted Retention Time (s)',
        'YLABEL': 'Residuals (Predicted - True)',
        # 如需特殊配置可在此覆盖default
    },
    
    'iphone_style': {
        'FIGSIZE': (6, 6),
        'LINEWIDTH': 3,
        'LABELSIZE': 17,
        'TITLESIZE': 16,
        'SAVE_DPI': 600,
        'SAVE_BBOX': 'tight',
        'SCATTER_SIZE': 70,
        'ALPHA': 0.8,
        'TICK_LENGTH': 6,
        'TICK_WIDTH': 2,
        'EDGE_COLOR': 'k',
        'GRID': False,
        'SPINE_VISIBLE': True,
        'IPHONE_COLORS': {
            'scatter': '#007AFF',
            'line':    '#AEAEB2',
            'text':    '#000000'
        }
    },
    'iphone_scatter': {
        'XLABEL': 'True Retention Time (s)',
        'YLABEL': 'Predicted Retention Time (s)',
        'ANNOT_FONT': 16,
        'ANNOT_POS': (0.05, 0.95)
    },
    'iphone_residual': {
        'XLABEL': 'Predicted Retention Time (s)',
        'YLABEL': 'Residuals (Predicted - True)',
        'ANNOT_FONT': 16,
        'ANNOT_POS': (0.05, 0.95),
        'TITLE_POS': (0.5, -0.15),
    }
}


def get_plot_params(plot_type='iphone_style', **overrides):
    """
    获取绘图参数：先读取default，然后读取特定图片类型参数，最后应用用户覆盖
    """
    # print("Using plot style:", plot_type)
    params = PLOT_CONFIG['default'].copy()
    if plot_type in PLOT_CONFIG:
        params.update(PLOT_CONFIG[plot_type])
    params.update(overrides)
    return params

def setup_axis_style(ax, params):
    """
    设置坐标轴样式（统一函数，避免重复代码）
    """
    ax.tick_params(axis='both', direction='out', length=params['TICK_LENGTH'], 
                   width=params['TICK_WIDTH'], color='black')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.grid(False)
    ax.set_facecolor('white')

def plot_training_curves(train_losses, val_losses, save_path=None, **overrides):
    """
    绘制训练曲线（全部点绘制，自动刻度间隔）
    """

    params = get_plot_params('training_curves', **overrides)

    plt.figure(figsize=params['FIGSIZE'])
    ax = plt.gca()

    setup_axis_style(ax, params)

    epochs = list(range(len(train_losses)))

    # 全部点绘制
    ax.plot(
        epochs, train_losses,
        label='Training Loss',
        color=params['IPHONE_COLORS']['train'],
        linewidth=params['LINEWIDTH'])

    ax.plot(
        epochs, val_losses,
        label='Validation Loss',
        color=params['IPHONE_COLORS']['val'],
        linewidth=params['LINEWIDTH'])


    # 最多显示 8~10 个主刻度
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))

    # Y轴自动漂亮间隔
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    ax.set_xlabel(params['XLABEL'], fontsize=params['LABELSIZE'])
    ax.set_ylabel(params['YLABEL'], fontsize=params['LABELSIZE'])
    ax.set_title("", fontsize=params['TITLESIZE'])

    ax.legend(fontsize=params['LABELSIZE'])
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            facecolor='white',
            edgecolor='none',
            dpi=params['SAVE_DPI'],
            bbox_inches=params['SAVE_BBOX'])

    plt.close()

def plot_predictions(results, save_path=None, **overrides):
    """
    绘制三张风格统一的图：主散点图、残差图、拼图（两图一页），全部采用 iPhone 风格。
    """


    # 取反归一化后的结果优先
    y_true = np.array(results.get('targets_orig', results['targets']))
    y_pred = np.array(results.get('predictions_orig', results['predictions']))

    IPHONE_COLORS = {
        'scatter': '#007AFF',  # iPhone 蓝
        'line': '#AEAEB2',     # iPhone 灰
        'text': '#000000'      # 黑色
    }

    # 计算指标
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # 1. 主散点图
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, y_pred, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Predicted RT (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_pred_vs_true.png') if save_path.endswith('.png') else save_path + '_pred_vs_true.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. 残差图
    residuals = y_pred - y_true
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, residuals, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    plt.axhline(y=0, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Residual (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_residual.png') if save_path.endswith('.png') else save_path + '_residual.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. 拼图（两图一页）
    plt.figure(figsize=(12, 6))
    # 左：主散点
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    ax1.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, y_pred, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    plt.plot(lims, lims, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Predicted RT (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.2f}", transform=ax1.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    # 右：残差
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    ax2.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax2.spines[spine].set_visible(True)
        ax2.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, residuals, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    plt.axhline(y=0, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Residual (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.2f}", transform=ax2.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_both.png') if save_path.endswith('.png') else save_path + '_both.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_uv_rt_histogram(csv_path, save_path=None):
    """
    读取 CSV 文件中的 'UV_RT-s' 列
    绘制 30~120 范围、每 10 为一个 bin 的直方图
    """

    # 读取数据
    df = pd.read_csv(csv_path)

    if "UV_RT-s" not in df.columns:
        raise ValueError("Column 'UV_RT-s' not found in CSV.")

    values = df["UV_RT-s"].dropna().values

    # 过滤范围
    values = values[(values >= 30) & (values <= 120)]

    # 构造 bins
    bins = np.arange(20, 91, 10)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black")

    plt.xlabel("UV_RT-s")
    plt.ylabel("Frequency")
    plt.title("UV_RT-s Distribution")
    plt.xlim(30, 90)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__": 
    data_path = "/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k4_train.csv"
    plot_uv_rt_histogram(data_path, save_path="/home/huangzy/uplc-method-recommendation/results/20260212/uv_rt_histogram.png")