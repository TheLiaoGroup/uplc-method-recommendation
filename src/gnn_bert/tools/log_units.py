import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_predictions(results, save_path):
    y_true = np.array(results.get('targets_orig', results['targets']))
    y_pred = np.array(results.get('predictions_orig', results['predictions']))
    y_true_norm = np.array(results['targets'])
    y_pred_norm = np.array(results['predictions'])
            
    
    df = pd.DataFrame({
        'SMILES': results['smiles'] if results['smiles'] else [''] * len(results['predictions']),
        'True_RT': y_true,
        'Predicted_RT': y_pred,
        'Residual': y_pred - y_true,
        'Target_RT_Norm': y_true_norm,
        'Predicted_RT_Norm': y_pred_norm,
        'Residual_Norm': y_pred_norm - y_true_norm})
    
    df.to_csv(save_path, index=False)
    # print(f"Predictions saved to {save_path}")


