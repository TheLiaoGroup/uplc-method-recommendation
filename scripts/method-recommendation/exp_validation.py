# ================= DEPENDENCIES =================
import os
import math
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from ml import calc_features_from_smarts, load_smarts_patterns


warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- GLOBAL CONFIGURATIONS ----------
SMARTS_FILE = '../../data/smarts/priority_fgs_823_newnew.txt'

FEATURE_COLS = ['MolWt', 'logP', 'TPSA', 'H_bond_donors', 'H_bond_acceptors']
FP_COLS = [f'col{i}' for i in range(823)]
MG_COLS = [f'fp_{i}' for i in range(1024)]
ALL_FEATURES = FEATURE_COLS + FP_COLS + MG_COLS

# ---------- FIXED METHOD ORDER ----------
FIXED_METHOD_ORDER = [
    'AM-I', 
    'AM-II', 
    'AM-III', 
    'AM-IV', 
    'AM-V', 
    'AM-VI'
]

# ---------- MODEL DIRECTORY MAPPING ----------

model_dir = Path('../../results/ml')

MODEL_DIR_MAP = {
    'AM-I': model_dir / 'svr-models/AM-I-svr-model',
    'AM-II': model_dir / 'svr-models/AM-II-svr-model',
    'AM-III': model_dir / 'svr-model-other4',
    'AM-IV': model_dir / 'svr-model-other4',
    'AM-V': model_dir / 'svr-model-other4',
    'AM-VI': model_dir / 'svr-model-other4',
}

# ---------- METHOD EVALUATION RANGE CONFIGURATION ----------
METHOD_RANGE_CONFIG = {
    'AM-I': (30, 120),
    'AM-II': (30, 120),
    'AM-III': (30, 150),
    'AM-IV': (30, 120),
    'AM-V': (30, 150),
    'AM-VI': (30, 180)
}

# ---------- AUTOMATIC SMARTS READING ----------
SMARTS_PATTERNS = load_smarts_patterns(SMARTS_FILE)

# ---------- FEATURE CALCULATION ----------
def calc_features(smiles: str) -> Optional[np.ndarray]:
    return calc_features_from_smarts(smiles, SMARTS_PATTERNS)

# ---------- MODEL MANAGEMENT ----------
class ModelHub:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self._load()

    def _load(self):
        model_name_patterns = {
            'AM-I': 'AM-I',
            'AM-II': 'AM-II',
            'AM-III': 'AM-III-filtered_final',
            'AM-IV': 'AM-IV-filtered_final',
            'AM-V': 'AM-V-filtered_final',
            'AM-VI': 'AM-VI-filtered_final'
        }
        
        for method_name, model_dir in MODEL_DIR_MAP.items():
            model_pattern = model_name_patterns[method_name]
            
            if method_name in ['AM-I', 'AM-II']:
                model_path = None
                scaler_path = None
                
                for file in os.listdir(model_dir):
                    if file.endswith('.joblib') and not file.endswith('_scaler.joblib'):
                        model_path = os.path.join(model_dir, file)
                    elif file.endswith('_scaler.joblib'):
                        scaler_path = os.path.join(model_dir, file)
                
                if not model_path or not scaler_path:
                    print(f'[WARN] Missing model or scaler for {method_name} in {model_dir}')
                    continue
            else:
                model_path = os.path.join(model_dir, f'{model_pattern}_svr_model.joblib')
                scaler_path = os.path.join(model_dir, f'{model_pattern}_scaler.joblib')
            
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[method_name] = joblib.load(model_path)
                    self.scalers[method_name] = joblib.load(scaler_path)
                    print(f'[INFO] Loaded {method_name} from {model_path}')
                else:
                    print(f'[WARN] Model files not found for {method_name}:')
                    print(f'       Model: {model_path} - {"Exists" if os.path.exists(model_path) else "Missing"}')
                    print(f'       Scaler: {scaler_path} - {"Exists" if os.path.exists(scaler_path) else "Missing"}')
            except Exception as e:
                print(f'[ERROR] Failed to load {method_name}: {e}')

    def predict(self, smiles: str) -> Dict[str, Optional[float]]:
        feat = calc_features(smiles)
        if feat is None:
            return {m: None for m in self.models}
        
        base = feat[:5]
        rest = feat[5:]
        preds = {}
        
        for name, model in self.models.items():
            scaler = self.scalers[name]
            base_scaled = scaler.transform(base.reshape(1, -1))[0]
            full = np.concatenate((base_scaled, rest)).reshape(1, -1)
            try:
                preds[name] = float(model.predict(full)[0])
            except Exception as e:
                print(f'Prediction error {name}: {e}')
                preds[name] = None
        
        return preds

# ---------- UNIFIED EVALUATION SYSTEM ----------
class UnifiedEvaluationSystem:
    def __init__(self,
                 min_interval: float = 9,
                 distance_weight: float = 10,
                 range_weight: float = 0.6,
                 importance_weight: float = 5.0,
                 strict_penalty: bool = True,
                 default_range: Tuple[float, float] = (30, 120)):
        """
        Unified Evaluation System
        
        Args:
            min_interval: Minimum required interval (seconds)
            distance_weight: Interval violation weight
            range_weight: Range violation weight
            importance_weight: Importance weight
            strict_penalty: Enable strict penalty
            default_range: Default retention time range
        """
        self.min_interval = min_interval
        self.distance_weight = distance_weight
        self.range_weight = range_weight
        self.importance_weight = importance_weight
        self.strict_penalty = strict_penalty
        self.default_range = default_range

    def _calculate_interval_score(self, values: List[float]) -> Tuple[float, List[Dict]]:
        """Calculate interval score"""
        violations, penalty = [], 0
        sorted_vals = sorted(values)
        n = len(values)
        for i in range(n - 1):
            gap = sorted_vals[i + 1] - sorted_vals[i]
            if gap < self.min_interval:
                shortage = self.min_interval - gap
                # Importance weighting: based on compound position in original list
                w1 = (1 / (values.index(sorted_vals[i]) + 1)) ** 3
                w2 = (1 / (values.index(sorted_vals[i + 1]) + 1)) ** 3
                w = max(w1, w2) * self.importance_weight
                p = shortage * w * self.distance_weight / n
                penalty += p
                violations.append({
                    'type': 'interval',
                    'values': [sorted_vals[i], sorted_vals[i + 1]],
                    'required': self.min_interval,
                    'actual': gap,
                    'penalty': p
                })
        return penalty, violations

    def _calculate_range_score(self, values: List[float], value_range: Tuple[float, float]) -> Tuple[float, List[Dict]]:
        """Calculate range score"""
        violations, penalty = [], 0
        min_v, max_v = value_range
        for idx, val in enumerate(values):
            if val < min_v or val > max_v:
                importance = idx + 1  # Position index starting from 1
                imp_w = math.exp(-0.5 * (importance - 1)) * self.importance_weight
                dist = min_v - val if val < min_v else val - max_v
                p = dist * imp_w * self.range_weight / len(values)
                penalty += p
                violations.append({
                    'type': 'range',
                    'value': val,
                    'importance': importance,
                    'distance': dist,
                    'penalty': p
                })
                # Strict mode: if first compound (P) is out of range, return -1 directly
                if idx == 0 and self.strict_penalty:
                    return -1, violations
        return penalty, violations

    def _normalize_score(self, d_pen: float, r_pen: float, n: int, value_range: Tuple[float, float]) -> float:
        """Normalized score calculation"""
        if r_pen == -1:
            return -1
        # Calculate maximum possible penalty
        max_d = self.min_interval * (n - 1) * sum(1 / i for i in range(1, n + 1))
        max_r = max(abs(value_range[0]), abs(value_range[1])) * sum(1 / i for i in range(1, n + 1))
        
        total_pen = self.distance_weight * d_pen + self.range_weight * r_pen
        max_total = self.distance_weight * max_d + self.range_weight * max_r
        
        return max(0, 1 - total_pen / max_total) if max_total else 1.0

    def evaluate(self, values: List[float], value_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Evaluate a set of retention times
        
        Args:
            values: List of retention times [P, S1, S2]
            value_range: Allowed time range, uses default if None
        
        Returns:
            Evaluation result dictionary
        """
        if value_range is None:
            value_range = self.default_range
        
        # Calculate interval score
        d_pen, d_vio = self._calculate_interval_score(values)
        
        # Calculate range score
        r_pen, r_vio = self._calculate_range_score(values, value_range)
        
        # Calculate final score
        score = self._normalize_score(d_pen, r_pen, len(values), value_range)
        
        return {
            'values': values,
            'distance_penalty': d_pen,
            'range_penalty': r_pen,
            'final_score': score,
            'distance_violations': d_vio,
            'range_violations': r_vio,
            'is_strict_penalty': score == -1,
            'value_range': value_range
        }
    
    def evaluate_datasets(self,
                          datasets: List[List[float]],
                          method_names: List[str],
                          exp_score: Optional[float] = None,
                          pred_score: Optional[float] = None,
                          save_csv: bool = True,
                          save_plot: bool = True,
                          output_dir: str = "./4-Exp-Reaction-data-results/") -> List[Dict[str, Any]]:
        """
        Evaluate multiple datasets for visualization and reporting
        
        Args:
            datasets: List of datasets (each dataset is a list of 3 values)
            method_names: List of method names corresponding to datasets
            exp_score: Experimental score to display (optional)
            pred_score: Prediction best score to display (optional)
            save_csv: Whether to save CSV report
            save_plot: Whether to save visualization plot
            output_dir: Output directory for saving files
        
        Returns:
            List of evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for idx, data in enumerate(datasets):
            method_name = method_names[idx]
            value_range = METHOD_RANGE_CONFIG.get(method_name, (30, 120))
            
            res = self.evaluate(data, value_range)
            res['dataset_id'] = idx
            res['method_name'] = method_name
            res['exp_score'] = exp_score
            res['pred_score'] = pred_score
            # Set x-axis limit based on range
            res['xmax'] = 210 if value_range[1] > 120 else 180
            results.append(res)
        
        if save_csv:
            self._save_csv_report(results, output_dir)
        if save_plot:
            self._create_visualization(results, output_dir)
        return results
    
    def _save_csv_report(self, results: List[Dict], out_dir: str):
        """Save detailed evaluation results to CSV"""
        rows = []
        for r in results:
            vios = []
            for v in r['distance_violations']:
                vios.append(f"Interval: {v['values']} req={v['required']} act={v['actual']:.2f}")
            for v in r['range_violations']:
                vios.append(f"Range: {v['value']} imp={v['importance']} dist={v['distance']:.2f}")
            rows.append({
                'Dataset_ID': r['dataset_id'],
                'Method': r['method_name'],
                'Values': str(r['values']),
                'Distance_Penalty': r['distance_penalty'],
                'Range_Penalty': r['range_penalty'],
                'Final_Score': r['final_score'],
                'Is_Strict_Penalty': r['is_strict_penalty'],
                'Value_Range': str(r['value_range']),
                'Violations': '; '.join(vios) if vios else 'None'
            })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'evaluation_results.csv'),
                                  index=False, encoding='utf-8-sig')
        print(f"Report saved to {out_dir}/evaluation_results.csv")
    
    def _create_visualization(self, results: List[Dict], out_dir: str):
        """Create comparison chart for all methods"""
        if not results:
            return
        
        # Sort results by fixed method order
        results_sorted = sorted(results, key=lambda x: FIXED_METHOD_ORDER.index(x['method_name']))
        n = len(results_sorted)
        max_vals = max(len(r['values']) for r in results_sorted)

        # Styling configurations
        tick_fontsize = 16
        label_fontsize = 17
        score_fontsize = 15
        axis_linewidth = 1.5
        tick_length = 8

        # Create figure with adjusted layout for score display
        fig, ax = plt.subplots(figsize=(14, max(5, n * 1.1)))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        colors = plt.cm.tab10.colors

        # Determine x-axis limits
        all_vals = [v for r in results_sorted for v in r['values']]
        gmin = min(min(all_vals), 30) - 5
        xmax = max(r['xmax'] for r in results_sorted)
        ax.set_xlim(gmin, xmax)

        # Adjust y-axis to create 10% top margin for score display
        y_min = -0.5
        y_max = n - 0.5
        total_height = y_max - y_min
        top_margin = total_height * 0.10  # 10% top margin
        ax.set_ylim(y_min, y_max + top_margin)

        # Add range background - 在30-180的x轴范围仅用一种背景
        ax.axvspan(30, 180, color='#D9D9D9', alpha=0.45, zorder=0)

        # Plot data points
        for i, res in enumerate(results_sorted):
            y = n - i - 1
            vals = res['values']
            value_range = res['value_range']
            
            # Plot each compound
            for j, v in enumerate(vals):
                size, color = 300 / (j + 1), colors[j % 10]
                # Use 'X' marker for out-of-range values
                marker = 'o' if value_range[0] <= v <= value_range[1] else 'X'
                ax.scatter(v, y, s=size, c=[color], marker=marker, alpha=0.9,
                           edgecolors='k', linewidths=1.5, zorder=3)
            
            # Highlight interval violations with red line
            sorted_vals = sorted(vals)
            for k in range(len(sorted_vals) - 1):
                if sorted_vals[k + 1] - sorted_vals[k] < self.min_interval:
                    ax.plot(sorted_vals[k:k + 2], [y, y], 'r-', lw=3, alpha=0.7, zorder=2)
            
            # Add evaluation score text (Predicted Score) - Predicted Score标签在xmax × 0.99处
            score_txt = f"{res['final_score']:.3f}" if res['final_score'] >= 0 else "Penalty"
            ax.text(xmax * 0.99, y, score_txt, ha='right', va='center',
                    fontsize=score_fontsize,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        # Add Experimental Score and Predicted Best Score in the top margin
        exp_score = results_sorted[0].get('exp_score')
        pred_score = results_sorted[0].get('pred_score')
        
        if exp_score is not None or pred_score is not None:
            # Position for scores in the top margin
            score_y_position = y_max + top_margin * 0.5
            
            # Add "Predicted Score" label at the far right (xmax × 0.99 position)
            pred_label_x = xmax * 0.99
            ax.text(pred_label_x, score_y_position, 'Predicted Score', ha='right', va='center',
                    fontsize=score_fontsize + 1, fontweight='bold',
                    bbox=dict(boxstyle="square,pad=0.3", fc="none", ec="none"))
            
            # Add Experimental Score and Predicted Best Score from left to right
            x_positions = [xmax * 0.30, xmax * 0.60]
            current_x_index = 0
            
            # Add Experimental Score if available
            if exp_score is not None:
                exp_text = f"Experimental Score: {exp_score:.3f}" if exp_score >= 0 else "Experimental: Penalty"
                ax.text(x_positions[current_x_index], score_y_position, exp_text, ha='center', va='center',
                        fontsize=score_fontsize,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))
                current_x_index += 1
            
            # Add Predicted Best Score if available
            if pred_score is not None:
                pred_text = f"Predicted Best Score: {pred_score:.3f}" if pred_score >= 0 else "Predicted: Penalty"
                ax.text(x_positions[current_x_index], score_y_position, pred_text, ha='center', va='center',
                        fontsize=score_fontsize,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.8))

        # Set labels and ticks
        ax.set_xlabel('Retention Time (s)', fontsize=label_fontsize)
        ax.set_ylabel('UPLC Method', fontsize=label_fontsize)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r['method_name'] for r in reversed(results_sorted)], fontsize=tick_fontsize)

        # Hide top y-tick labels to avoid overlap with score display
        ax.tick_params(axis='y', which='both',
                       labelsize=tick_fontsize,
                       length=tick_length,
                       width=axis_linewidth)

        ax.tick_params(axis='x', which='major',
                       labelsize=tick_fontsize,
                       length=tick_length,
                       width=axis_linewidth)

        ax.spines['top'].set_linewidth(axis_linewidth)
        ax.spines['bottom'].set_linewidth(axis_linewidth)
        ax.spines['left'].set_linewidth(axis_linewidth)
        ax.spines['right'].set_linewidth(axis_linewidth)

        ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=1.2)

        # Create legend
        labels = ['P', 'S1', 'S2'][:max_vals]
        legend = [plt.scatter([], [], s=250 // (j + 1), c=[colors[j % 10]], label=labels[j])
                  for j in range(len(labels))]
        legend += [
            plt.scatter([], [], marker='X', c='gray', s=120, label='Out Range'),
            plt.Line2D([0], [0], color='red', lw=3, label='Interval Violation')
        ]
        
        # Position legend below the score display
        ax.legend(handles=legend,
                  bbox_to_anchor=(0.5, 1.05),
                  loc='lower center',
                  ncol=len(legend),
                  fontsize=tick_fontsize - 1)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'comparison_chart.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to {out_dir}/comparison_chart.png")

# ---------- PREDICTION DATA EVALUATOR ----------
class PredictionEvaluator:
    """Prediction Data Evaluator: Find the best method among 6 methods"""
    
    def __init__(self, model_hub: ModelHub, evaluator: UnifiedEvaluationSystem):
        self.model_hub = model_hub
        self.evaluator = evaluator
    
    def evaluate_predictions(self, smiles_list: List[str], 
                           output_dir: Optional[str] = None,
                           row_index: Optional[int] = None,
                           exp_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate predicted retention times
        
        Args:
            smiles_list: List of SMILES [P, S1, S2]
            output_dir: Directory to save visualization and CSV (optional)
            row_index: Row index for naming output directory (optional)
            exp_score: Experimental score to include in visualization (optional)
        
        Returns:
            Dictionary containing best method, scores, predictions, and recommended methods
        """
        # Get all prediction results
        all_predictions = []
        valid_smiles = []
        for smiles in smiles_list:
            if smiles and pd.notna(smiles):
                preds = self.model_hub.predict(smiles)
                all_predictions.append(preds)
                valid_smiles.append(smiles)
            else:
                all_predictions.append(None)
        
        if len(valid_smiles) < 3:
            return {
                'best_methods': None,
                'best_score': None,
                'all_scores': {},
                'predictions': all_predictions,
                'predicted_values': {},
                'error': f'Insufficient valid SMILES: {len(valid_smiles)}/3'
            }
        
        # Organize predictions by method
        method_predictions = {}
        for method in FIXED_METHOD_ORDER:
            method_values = []
            all_valid = True
            for pred_dict in all_predictions:
                if pred_dict and method in pred_dict and pred_dict[method] is not None:
                    method_values.append(pred_dict[method])
                else:
                    all_valid = False
                    break
            
            if all_valid and len(method_values) == 3:
                method_predictions[method] = method_values
            else:
                method_predictions[method] = None
        
        # Evaluate predictions for each method
        method_scores = {}
        for method, values in method_predictions.items():
            if values is not None:
                # Get evaluation range for this method
                value_range = METHOD_RANGE_CONFIG.get(method, (30, 120))
                result = self.evaluator.evaluate(values, value_range)
                method_scores[method] = {
                    'score': result['final_score'],
                    'values': values,
                    'range': value_range,
                    'valid': True
                }
            else:
                method_scores[method] = {
                    'score': None,
                    'values': None,
                    'range': METHOD_RANGE_CONFIG.get(method, (30, 120)),
                    'valid': False,
                    'error': 'Incomplete predictions'
                }
        
        # Find best methods (handling ties)
        valid_scores = {k: v for k, v in method_scores.items() 
                       if v['valid'] and v['score'] is not None and v['score'] >= 0}
        
        if valid_scores:
            best_score = max(v['score'] for v in valid_scores.values())
            best_methods = [k for k, v in valid_scores.items() if v['score'] == best_score]
            # Sort best methods according to FIXED_METHOD_ORDER
            best_methods = sorted(best_methods, key=lambda x: FIXED_METHOD_ORDER.index(x))
        else:
            best_methods = []
            best_score = None
        
        # Generate visualization and report if output_dir is provided
        if output_dir and best_methods:
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare datasets for visualization
            datasets = []
            method_names = []
            for method in FIXED_METHOD_ORDER:
                if method_scores[method]['valid']:
                    datasets.append(method_scores[method]['values'])
                    method_names.append(method)
            
            if datasets:
                self.evaluator.evaluate_datasets(
                    datasets=datasets,
                    method_names=method_names,
                    exp_score=exp_score,
                    pred_score=best_score,
                    save_csv=True,
                    save_plot=True,
                    output_dir=output_dir
                )
        
        # Get predicted values for the best method(s)
        best_method_values = {}
        for method in best_methods:
            if method in method_predictions and method_predictions[method] is not None:
                best_method_values[method] = method_predictions[method]
        
        return {
            'best_methods': best_methods,  # List of best methods
            'best_methods_str': ', '.join(best_methods) if best_methods else 'None',  # String representation
            'best_score': best_score,
            'all_scores': method_scores,
            'predictions': all_predictions,
            'method_predictions': method_predictions,  # All predictions organized by method
            'best_method_values': best_method_values,  # Values for best method(s)
            'error': None
        }

# ---------- EXPERIMENTAL DATA EVALUATOR ----------
class ExperimentalEvaluator:
    """Experimental Data Evaluator: Directly evaluate experimental retention times"""
    
    def __init__(self, evaluator: UnifiedEvaluationSystem):
        self.evaluator = evaluator
    
    def evaluate_experimental(self, exp_values: List[float], 
                            value_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Evaluate experimental retention times
        
        Args:
            exp_values: List of experimental retention times [exp-RT-P, exp-RT-S1, exp-RT-S2]
            value_range: Allowed time range, uses default if None
        
        Returns:
            Evaluation result dictionary
        """
        # Check data validity
        if len(exp_values) != 3:
            return {
                'score': None,
                'error': f'Experimental data length should be 3, actual: {len(exp_values)}'
            }
        
        # Check for NaN values
        if any(pd.isna(v) for v in exp_values):
            return {
                'score': None,
                'error': 'Experimental data contains NaN values'
            }
        
        # Perform evaluation
        result = self.evaluator.evaluate(exp_values, value_range)
        
        return {
            'score': result['final_score'],
            'values': exp_values,
            'distance_penalty': result['distance_penalty'],
            'range_penalty': result['range_penalty'],
            'violations': result['distance_violations'] + result['range_violations'],
            'is_strict_penalty': result['is_strict_penalty'],
            'value_range': result['value_range']
        }

# ---------- MAIN PROCESSING CLASS ----------
class ReactionDataProcessor:
    """Main Class for Reaction Data Processing"""
    
    def __init__(self, 
                 min_interval: float = 9,
                 distance_weight: float = 5,
                 range_weight: float = 1,
                 importance_weight: float = 2.0,
                 strict_penalty: bool = True,
                 default_range: Tuple[float, float] = (30, 120)):
        
        # Initialize model hub
        self.model_hub = ModelHub()
        
        # Initialize unified evaluation system
        self.evaluator = UnifiedEvaluationSystem(
            min_interval=min_interval,
            distance_weight=distance_weight,
            range_weight=range_weight,
            importance_weight=importance_weight,
            strict_penalty=strict_penalty,
            default_range=default_range
        )
        
        # Initialize evaluators
        self.pred_evaluator = PredictionEvaluator(self.model_hub, self.evaluator)
        self.exp_evaluator = ExperimentalEvaluator(self.evaluator)
        
        # Configuration parameters
        self.config = {
            'min_interval': min_interval,
            'distance_weight': distance_weight,
            'range_weight': range_weight,
            'importance_weight': importance_weight,
            'strict_penalty': strict_penalty,
            'default_range': default_range
        }
    
    def process_row(self, row: pd.Series, row_idx: int, output_dir: str, base_filename: str) -> Dict[str, Any]:
        """
        Process a single row of data
        
        Args:
            row: Series containing SMILES and experimental RT
            row_idx: Row index (0-based)
            output_dir: Base output directory
            base_filename: Base filename for output
        
        Returns:
            Processing result dictionary
        """
        result = {
            'smiles_p': row.get('P'),
            'smiles_s1': row.get('S1'),
            'smiles_s2': row.get('S2'),
            'exp_rt_p': row.get('exp-RT-P'),
            'exp_rt_s1': row.get('exp-RT-S1'),
            'exp_rt_s2': row.get('exp-RT-S2')
        }
        
        # Create row-specific output directory
        row_dir = f'row_{row_idx}'
        row_output_dir = os.path.join(output_dir, row_dir)
        
        # 2. Experimental evaluation (do this first to get exp_score for visualization)
        exp_values = [row.get('exp-RT-P'), row.get('exp-RT-S1'), row.get('exp-RT-S2')]
        exp_result = self.exp_evaluator.evaluate_experimental(exp_values)
        result['exp_score'] = exp_result['score']
        result['exp_violations'] = exp_result['violations']
        result['exp_error'] = exp_result.get('error')
        
        # 1. Prediction evaluation (pass exp_score for visualization)
        smiles_list = [row.get('P'), row.get('S1'), row.get('S2')]
        pred_result = self.pred_evaluator.evaluate_predictions(
            smiles_list, 
            output_dir=row_output_dir,
            row_index=row_idx,
            exp_score=exp_result['score']  # Pass experimental score for visualization
        )
        
        result['pred_best_methods'] = pred_result['best_methods']
        result['pred_best_methods_str'] = pred_result['best_methods_str']
        result['pred_best_score'] = pred_result['best_score']
        result['pred_all_scores'] = pred_result['all_scores']
        result['pred_method_predictions'] = pred_result['method_predictions']
        
        # Store individual prediction values for best method(s)
        if pred_result['best_methods']:
            for method in pred_result['best_methods']:
                if method in pred_result['best_method_values']:
                    values = pred_result['best_method_values'][method]
                    result[f'pred_{method}_P'] = values[0] if len(values) > 0 else None
                    result[f'pred_{method}_S1'] = values[1] if len(values) > 1 else None
                    result[f'pred_{method}_S2'] = values[2] if len(values) > 2 else None
        
        # Add row directory information
        result['row_dir'] = row_dir
        
        return result
    
    def process_file(self, input_file: str, output_dir: str = "./4-Exp-Reaction-data-results") -> str:
        """
        Process an entire CSV file
        
        Args:
            input_file: Input CSV file path
            output_dir: Output directory
        
        Returns:
            Output file path
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Read CSV file
        try:
            print(f"Reading file: {input_file}")
            df = pd.read_csv(input_file)
            
            # Check required columns
            required_cols = ['P', 'S1', 'S2', 'exp-RT-P', 'exp-RT-S1', 'exp-RT-S2']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"File missing required columns: {missing_cols}")
            
            print(f"Successfully read {len(df)} rows of data")
            
        except Exception as e:
            print(f"Failed to read file: {e}")
            return None
        
        # Initialize results list
        all_results = []
        
        # Process data row by row
        for idx, row in df.iterrows():
            print(f"\nProcessing row {idx+1}/{len(df)}...")
            
            try:
                row_result = self.process_row(row, idx, output_dir, base_filename)
                all_results.append(row_result)
                
                # Print processing result
                print(f"  SMILES: P={row['P'][:20]}..., S1={row['S1'][:20]}..., S2={row['S2'][:20]}...")
                print(f"  Experimental RT: P={row['exp-RT-P']:.2f}, S1={row['exp-RT-S1']:.2f}, S2={row['exp-RT-S2']:.2f}")
                print(f"  Prediction best method(s): {row_result['pred_best_methods_str']}, Score: {row_result['pred_best_score']}")
                print(f"  Experimental score: {row_result['exp_score']}")
                
            except Exception as e:
                print(f"  Error processing row {idx+1}: {e}")
                # Add error information
                error_result = {
                    'smiles_p': row.get('P'),
                    'smiles_s1': row.get('S1'),
                    'smiles_s2': row.get('S2'),
                    'exp_rt_p': row.get('exp-RT-P'),
                    'exp_rt_s1': row.get('exp-RT-S1'),
                    'exp_rt_s2': row.get('exp-RT-S2'),
                    'pred_best_methods': None,
                    'pred_best_methods_str': 'None',
                    'pred_best_score': None,
                    'exp_score': None,
                    'row_dir': f'row_{idx}',
                    'error': str(e)
                }
                all_results.append(error_result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Extract key information to original DataFrame
        df['pred_best_methods'] = results_df['pred_best_methods_str']
        df['pred_best_score'] = results_df['pred_best_score']
        df['exp_score'] = results_df['exp_score']
        
        # Add row directory column as first column
        df.insert(0, 'row_dir', results_df['row_dir'])
        
        # Add predicted values for the best method(s) to the original DataFrame
        for method in FIXED_METHOD_ORDER:
            # Create columns for this method's predicted values
            p_values = []
            s1_values = []
            s2_values = []
            
            for result in all_results:
                if (result.get('pred_best_methods') and 
                    method in result['pred_best_methods']):
                    p_values.append(result.get(f'pred_{method}_P'))
                    s1_values.append(result.get(f'pred_{method}_S1'))
                    s2_values.append(result.get(f'pred_{method}_S2'))
                else:
                    p_values.append(None)
                    s1_values.append(None)
                    s2_values.append(None)
            
            # Only add columns if there are any non-None values
            if any(v is not None for v in p_values + s1_values + s2_values):
                df[f'pred_{method}_P'] = p_values
                df[f'pred_{method}_S1'] = s1_values
                df[f'pred_{method}_S2'] = s2_values
        
        # Optional: Add detailed prediction score columns for all methods
        for method in FIXED_METHOD_ORDER:
            method_scores = []
            for result in all_results:
                scores = result.get('pred_all_scores', {})
                if scores and method in scores and scores[method].get('valid', False):
                    method_scores.append(scores[method]['score'])
                else:
                    method_scores.append(None)
            df[f'pred_score_{method}'] = method_scores
        
        # Generate output filename
        output_file = os.path.join(output_dir, f"{base_filename}_evaluated.csv")
        
        # Save results as CSV
        try:
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Generate statistics report
            self._generate_statistics_report(df, output_dir, base_filename)
            
            # Generate visualization
            create_comparison_visualization(df, output_dir)
            
            return output_file
            
        except Exception as e:
            print(f"Failed to save results: {e}")
            return None
    
    def _generate_statistics_report(self, df: pd.DataFrame, output_dir: str, input_name: str):
        """Generate statistics report"""
        stats = {
            'total_rows': len(df),
            'rows_with_prediction': df['pred_best_methods'].notna().sum(),
            'rows_with_exp_score': df['exp_score'].notna().sum(),
            'avg_pred_score': df['pred_best_score'].mean() if df['pred_best_score'].notna().any() else None,
            'avg_exp_score': df['exp_score'].mean() if df['exp_score'].notna().any() else None,
            'pred_score_distribution': {
                'excellent(0.9-1.0)': ((df['pred_best_score'] >= 0.9) & (df['pred_best_score'] <= 1.0)).sum(),
                'good(0.7-0.9)': ((df['pred_best_score'] >= 0.7) & (df['pred_best_score'] < 0.9)).sum(),
                'fair(0.5-0.7)': ((df['pred_best_score'] >= 0.5) & (df['pred_best_score'] < 0.7)).sum(),
                'poor(<0.5)': (df['pred_best_score'] < 0.5).sum(),
                'penalty(-1)': (df['pred_best_score'] == -1).sum() if df['pred_best_score'].notna().any() else 0
            },
            'exp_score_distribution': {
                'excellent(0.9-1.0)': ((df['exp_score'] >= 0.9) & (df['exp_score'] <= 1.0)).sum(),
                'good(0.7-0.9)': ((df['exp_score'] >= 0.7) & (df['exp_score'] < 0.9)).sum(),
                'fair(0.5-0.7)': ((df['exp_score'] >= 0.5) & (df['exp_score'] < 0.7)).sum(),
                'poor(<0.5)': (df['exp_score'] < 0.5).sum(),
                'penalty(-1)': (df['exp_score'] == -1).sum() if df['exp_score'].notna().any() else 0,
                'missing': df['exp_score'].isna().sum()
            }
        }
        
        # Analyze method recommendations (handling multiple methods)
        if df['pred_best_methods'].notna().any():
            all_recommendations = []
            for methods_str in df['pred_best_methods'].dropna():
                methods = [m.strip() for m in methods_str.split(',')]
                all_recommendations.extend(methods)
            
            from collections import Counter
            method_counts = Counter(all_recommendations)
            stats['method_recommendation_distribution'] = dict(method_counts)
            
            # Calculate percentage of rows where each method is recommended
            total_recommendations = sum(method_counts.values())
            method_percentages = {method: count/len(df)*100 for method, count in method_counts.items()}
            stats['method_recommendation_percentage'] = method_percentages
        
        # Save statistics report
        stats_df = pd.DataFrame([stats])
        stats_file = os.path.join(output_dir, f"{input_name}_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"Statistics report saved to: {stats_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY:")
        print("="*60)
        print(f"Total rows: {stats['total_rows']}")
        print(f"Successful prediction rows: {stats['rows_with_prediction']}")
        print(f"Rows with experimental score: {stats['rows_with_exp_score']}")
        print(f"Average prediction score: {stats['avg_pred_score']:.3f}")
        print(f"Average experimental score: {stats['avg_exp_score']:.3f}")
        
        if 'method_recommendation_distribution' in stats:
            print("\nMethod recommendation distribution (including ties):")
            for method, count in stats['method_recommendation_distribution'].items():
                percentage = stats['method_recommendation_percentage'][method]
                print(f"  {method}: {count} times ({percentage:.1f}% of rows)")
        
        print("\nPrediction score distribution:")
        for category, count in stats['pred_score_distribution'].items():
            if stats['rows_with_prediction'] > 0:
                print(f"  {category}: {count} rows ({count/stats['rows_with_prediction']*100:.1f}%)")
        
        print("\nExperimental score distribution:")
        for category, count in stats['exp_score_distribution'].items():
            print(f"  {category}: {count} rows")

# ---------- VISUALIZATION FUNCTION ----------
def create_comparison_visualization(df: pd.DataFrame, output_dir: str):
    """Create comparison visualization between prediction and experimental scores"""
    if 'pred_best_score' not in df.columns or 'exp_score' not in df.columns:
        print("Cannot create visualization: Missing required score columns")
        return
    
    # Filter valid data
    valid_data = df.dropna(subset=['pred_best_score', 'exp_score'])
    if len(valid_data) == 0:
        print("Not enough data for visualization")
        return
    
    # Create scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Prediction score vs Experimental score
    ax1 = axes[0, 0]
    ax1.scatter(valid_data['pred_best_score'], valid_data['exp_score'], alpha=0.6)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
    ax1.set_xlabel('Prediction Score', fontsize=12)
    ax1.set_ylabel('Experimental Score', fontsize=12)
    ax1.set_title('Prediction Score vs Experimental Score', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. Score distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(valid_data['pred_best_score'], bins=20, alpha=0.7, label='Prediction Score', color='blue')
    ax2.hist(valid_data['exp_score'], bins=20, alpha=0.7, label='Experimental Score', color='orange')
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Score Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Method distribution analysis (for multi-method recommendations)
    ax3 = axes[1, 0]
    if 'pred_best_methods' in valid_data.columns:
        # Count occurrences of each method (including ties)
        method_counts = {}
        for methods_str in valid_data['pred_best_methods'].dropna():
            methods = [m.strip() for m in methods_str.split(',')]
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            methods = list(method_counts.keys())
            counts = [method_counts[m] for m in methods]
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            methods = [methods[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            ax3.bar(methods, counts)
            ax3.set_xlabel('Recommended Method', fontsize=12)
            ax3.set_ylabel('Count (including ties)', fontsize=12)
            ax3.set_title('Method Recommendation Distribution', fontsize=14)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
    
    # 4. Score difference box plot
    ax4 = axes[1, 1]
    score_diff = valid_data['pred_best_score'] - valid_data['exp_score']
    ax4.boxplot(score_diff.dropna())
    ax4.set_ylabel('Score Difference (Prediction-Experimental)', fontsize=12)
    ax4.set_title('Score Difference Distribution', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_file = os.path.join(output_dir, 'score_comparison.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization chart saved: {viz_file}")

# ---------- MAIN FUNCTION ----------
def main():
    """Main function"""
    print("="*60)
    print("REACTION DATA EVALUATION SYSTEM")
    print("="*60)
    
    # Configure file paths
    reaction_data_dir = "../../data/reaction"
    output_dir = "../../results/method-recommendation/Exp-Reaction-data-results"
    

    if not os.path.exists(reaction_data_dir):
        print(f"Error: Input directory '{reaction_data_dir}' does not exist!")
        print("Please create the Reaction-data directory and place CSV files in it")
        return
    
    # Get all CSV files in the directory
    csv_file = reaction_data_dir + "/exp_reaction.csv"

    
    if not csv_file:
        print(f"No CSV files found in {reaction_data_dir}")
        return

    
    # Initialize processor
    print("\nInitializing models and evaluation system...")
    processor = ReactionDataProcessor(
        min_interval=9,
        distance_weight=5,
        range_weight=1,
        importance_weight=2.0,
        strict_penalty=True,
        default_range=(30, 120)
    )
    
    # Process
    input_file = os.path.join(reaction_data_dir, csv_file)
    print(f"\n{'='*60}")
    print(f"Processing file: {csv_file}")
    print(f"{'='*60}")
    
    result_file = processor.process_file(input_file, output_dir)
    
    if result_file:
        print(f"\nProcessing completed for {csv_file}!")
        print(f"Result file: {result_file}")
        
        # Check for row-specific output directories
        row_dirs = [d for d in os.listdir(output_dir) if d.startswith('row_')]
        if row_dirs:
            print(f"Generated {len(row_dirs)} row-specific directories in {output_dir}")
    else:
        print(f"\nProcessing failed for {csv_file}!")

# ---------- COMMAND LINE INTERFACE ----------
if __name__ == "__main__":
    # Run main program
    main()