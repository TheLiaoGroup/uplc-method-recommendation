# ================= DEPENDENCIES =================
import os
import math
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import chardet

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- GLOBAL CONFIGURATIONS ----------
SMARTS_FILE = '../datas/SMARTS/priority_fgs_823_newnew.txt'

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
MODEL_DIR_MAP = {
    'AM-I': './2-svr-models/AM-I-svr-model',
    'AM-II': './2-svr-models/AM-II-svr-model',
    'AM-III': './2-svr-model-other4',
    'AM-IV': './2-svr-model-other4',
    'AM-V': './2-svr-model-other4',
    'AM-VI': './2-svr-model-other4'
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
with open(SMARTS_FILE, 'rb') as f:
    raw = f.read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
with open(SMARTS_FILE, encoding=enc, errors='ignore') as f:
    SMARTS_PATTERNS = [l.strip() for l in f if l.strip()]

# ---------- FEATURE CALCULATION ----------
def calc_features(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    base = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    fp_823 = [0] * 823
    for i, sma in enumerate(SMARTS_PATTERNS):
        patt = Chem.MolFromSmarts(sma)
        if patt and mol.HasSubstructMatch(patt):
            fp_823[i] = 1
    mg = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(base + fp_823 + list(mg), dtype=np.float32)

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
                          save_csv: bool = True,
                          save_plot: bool = True,
                          output_dir: str = "./4-All-Reaction-data-results/") -> List[Dict[str, Any]]:
        """
        Evaluate multiple datasets for visualization and reporting
        
        Args:
            datasets: List of datasets (each dataset is a list of 3 values)
            method_names: List of method names corresponding to datasets
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

        # Create figure with adjusted layout
        fig, ax = plt.subplots(figsize=(14, max(5, n * 1.1)))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        colors = plt.cm.tab10.colors

        # Determine x-axis limits
        all_vals = [v for r in results_sorted for v in r['values']]
        gmin = min(min(all_vals), 30) - 5
        xmax = max(r['xmax'] for r in results_sorted)
        ax.set_xlim(gmin, xmax)

        # Set y-axis limits
        y_min = -0.5
        y_max = n - 0.5
        ax.set_ylim(y_min, y_max)

        # Add range background
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
            
            # Add evaluation score text
            score_txt = f"{res['final_score']:.3f}" if res['final_score'] >= 0 else "Penalty"
            ax.text(xmax * 0.99, y, score_txt, ha='right', va='center',
                    fontsize=score_fontsize,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        # Set labels and ticks
        ax.set_xlabel('Retention Time (s)', fontsize=label_fontsize)
        ax.set_ylabel('UPLC Method', fontsize=label_fontsize)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r['method_name'] for r in reversed(results_sorted)], fontsize=tick_fontsize)

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
        
        # Position legend
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
    """Prediction Data Evaluator: Find the best method among six methods"""
    
    def __init__(self, model_hub: ModelHub, evaluator: UnifiedEvaluationSystem):
        self.model_hub = model_hub
        self.evaluator = evaluator
    
    def evaluate_predictions(self, smiles_list: List[str], 
                           row_index: int,
                           output_dir: str = "./4-All-Reaction-data-results") -> Dict[str, Any]:
        """
        Evaluate predicted retention times
        
        Args:
            smiles_list: List of SMILES [P, S1, S2]
            row_index: Row index for naming output directory
            output_dir: Directory to save visualization and CSV
        
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
        
        # Generate visualization and report
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare datasets for visualization (only valid methods)
        datasets = []
        method_names = []
        for method in FIXED_METHOD_ORDER:
            if method_scores[method]['valid']:
                datasets.append(method_scores[method]['values'])
                method_names.append(method)
        
        if datasets:
            row_output_dir = os.path.join(output_dir, f"row_{row_index}")
            os.makedirs(row_output_dir, exist_ok=True)
            
            self.evaluator.evaluate_datasets(
                datasets=datasets,
                method_names=method_names,
                save_csv=True,
                save_plot=True,
                output_dir=row_output_dir
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
        
        # Initialize evaluator
        self.pred_evaluator = PredictionEvaluator(self.model_hub, self.evaluator)
        
        # Configuration parameters
        self.config = {
            'min_interval': min_interval,
            'distance_weight': distance_weight,
            'range_weight': range_weight,
            'importance_weight': importance_weight,
            'strict_penalty': strict_penalty,
            'default_range': default_range
        }
    
    def process_row(self, row: pd.Series, row_idx: int) -> Dict[str, Any]:
        """
        Process a single row of data
        
        Args:
            row: Series containing SMILES
            row_idx: Row index (0-based)
        
        Returns:
            Processing result dictionary
        """
        # Prediction evaluation
        smiles_list = [row.get('P'), row.get('S1'), row.get('S2')]
        pred_result = self.pred_evaluator.evaluate_predictions(
            smiles_list, 
            row_index=row_idx
        )
        
        result = {
            'pred_best_methods': pred_result['best_methods'],
            'pred_best_methods_str': pred_result['best_methods_str'],
            'pred_best_score': pred_result['best_score'],
            'error': pred_result.get('error')
        }
        
        # Store individual prediction values for best method(s)
        if pred_result['best_methods']:
            for method in pred_result['best_methods']:
                if method in pred_result['best_method_values']:
                    values = pred_result['best_method_values'][method]
                    result[f'pred_{method}_P'] = values[0] if len(values) > 0 else None
                    result[f'pred_{method}_S1'] = values[1] if len(values) > 1 else None
                    result[f'pred_{method}_S2'] = values[2] if len(values) > 2 else None
        
        # Store all method scores
        for method in FIXED_METHOD_ORDER:
            if method in pred_result['all_scores']:
                scores = pred_result['all_scores'][method]
                if scores['valid']:
                    result[f'pred_score_{method}'] = scores['score']
                else:
                    result[f'pred_score_{method}'] = None
            else:
                result[f'pred_score_{method}'] = None
        
        return result
    
    def process_file(self, input_file: str, output_dir: str = "./4-All-Reaction-data-results") -> str:
        """
        Process an entire CSV file
        
        Args:
            input_file: Input CSV file path
            output_dir: Base output directory
        
        Returns:
            Output file path
        """
        # Read CSV file
        try:
            print(f"Reading file: {input_file}")
            df = pd.read_csv(input_file)
            
            # Check required columns
            required_cols = ['P', 'S1', 'S2']
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
                row_result = self.process_row(row, idx)
                all_results.append(row_result)
                
                # Print processing result
                print(f"  SMILES: P={row['P'][:20]}..., S1={row['S1'][:20]}..., S2={row['S2'][:20]}...")
                print(f"  Prediction best method(s): {row_result['pred_best_methods_str']}, Score: {row_result['pred_best_score']}")
                
            except Exception as e:
                print(f"  Error processing row {idx+1}: {e}")
                # Add error information
                error_result = {
                    'pred_best_methods': None,
                    'pred_best_methods_str': 'None',
                    'pred_best_score': None,
                    'error': str(e)
                }
                all_results.append(error_result)
        
        # Combine original data with results
        results_df = pd.DataFrame(all_results)
        
        # Add results to original DataFrame
        for col in results_df.columns:
            df[col] = results_df[col]
        
        # Generate output filename
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{input_name}_evaluated.csv")
        
        # Save results as CSV
        try:
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Generate statistics report
            self._generate_statistics_report(df, output_dir, input_name)
            
            return output_file
            
        except Exception as e:
            print(f"Failed to save results: {e}")
            return None
    
    def _generate_statistics_report(self, df: pd.DataFrame, output_dir: str, input_name: str):
        """Generate statistics report"""
        stats = {
            'total_rows': len(df),
            'rows_with_prediction': df['pred_best_methods_str'].notna().sum(),
            'avg_pred_score': df['pred_best_score'].mean() if df['pred_best_score'].notna().any() else None,
            'pred_score_distribution': {
                'excellent(0.9-1.0)': ((df['pred_best_score'] >= 0.9) & (df['pred_best_score'] <= 1.0)).sum(),
                'good(0.7-0.9)': ((df['pred_best_score'] >= 0.7) & (df['pred_best_score'] < 0.9)).sum(),
                'fair(0.5-0.7)': ((df['pred_best_score'] >= 0.5) & (df['pred_best_score'] < 0.7)).sum(),
                'poor(<0.5)': (df['pred_best_score'] < 0.5).sum(),
                'penalty(-1)': (df['pred_best_score'] == -1).sum() if df['pred_best_score'].notna().any() else 0
            }
        }
        
        # Analyze method recommendations (handling multiple methods)
        if df['pred_best_methods_str'].notna().any():
            all_recommendations = []
            for methods_str in df['pred_best_methods_str'].dropna():
                if methods_str != 'None':
                    methods = [m.strip() for m in methods_str.split(',')]
                    all_recommendations.extend(methods)
            
            if all_recommendations:
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
        print(f"Average prediction score: {stats['avg_pred_score']:.3f}")
        
        if 'method_recommendation_distribution' in stats:
            print("\nMethod recommendation distribution (including ties):")
            for method, count in stats['method_recommendation_distribution'].items():
                percentage = stats['method_recommendation_percentage'][method]
                print(f"  {method}: {count} times ({percentage:.1f}% of rows)")
        
        print("\nPrediction score distribution:")
        for category, count in stats['pred_score_distribution'].items():
            if stats['rows_with_prediction'] > 0:
                print(f"  {category}: {count} rows ({count/stats['rows_with_prediction']*100:.1f}%)")

# ---------- MAIN FUNCTION ----------
def main():
    """Main function"""
    print("="*60)
    print("REACTION DATA EVALUATION SYSTEM")
    print("="*60)
    
    # Configure file paths
    reaction_data_dir = "./4-All-Reaction-data"
    output_dir = "./4-All-Reaction-data-results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(reaction_data_dir):
        print(f"Error: Input directory '{reaction_data_dir}' does not exist!")
        print("Please create the Reaction-data directory and place CSV files in it")
        return
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(reaction_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {reaction_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process:")
    for file in csv_files:
        print(f"  - {file}")
    
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
    
    # Process each file
    for csv_file in csv_files:
        input_file = os.path.join(reaction_data_dir, csv_file)
        print(f"\n{'='*60}")
        print(f"Processing file: {csv_file}")
        print(f"{'='*60}")
        
        result_file = processor.process_file(input_file, output_dir)
        
        if result_file:
            print(f"\nProcessing completed for {csv_file}!")
            print(f"Result file: {result_file}")
        else:
            print(f"\nProcessing failed for {csv_file}!")

# ---------- COMMAND LINE INTERFACE ----------
if __name__ == "__main__":
    # Run main program
    main()