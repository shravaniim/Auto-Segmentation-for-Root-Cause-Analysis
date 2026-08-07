import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def compute_auc_gini(df):
    if len(df['target'].unique()) < 2:
        return np.nan, np.nan
    try:
        auc = roc_auc_score(df['target'], df['score'])
        gini = 2 * auc - 1
        return auc, gini
    except Exception:
        return np.nan, np.nan

def main():
    start_time = time.perf_counter()
    
    # 1. Load Data
    dev = pd.read_csv('development_data_5000_shap.csv')
    mon = pd.read_csv('monitoring_data_5000_shap.csv')
    
    dev['is_monitoring'] = 0
    mon['is_monitoring'] = 1
    
    combined = pd.concat([dev, mon], ignore_index=True)
    
    # 2. Features for Decision Tree
    feature_cols = ['age', 'income', 'region', 'occupation']
    
    # One-hot encode categorical features for tree
    X_df = pd.get_dummies(combined[feature_cols], columns=['region', 'occupation'], drop_first=False)
    y = combined['is_monitoring']
    
    # 3. Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05, random_state=42)
    clf.fit(X_df, y)
    
    # Leaf assignment
    combined['leaf'] = clf.apply(X_df)
    dev['leaf'] = combined[combined['is_monitoring'] == 0]['leaf'].values
    mon['leaf'] = combined[combined['is_monitoring'] == 1]['leaf'].values
    
    # Build tree rules for each leaf node
    tree_rules = {}
    
    def recurse(node, current_rules):
        if clf.tree_.children_left[node] == clf.tree_.children_right[node]: # leaf
            tree_rules[node] = list(current_rules)
            return
        
        feature_name = X_df.columns[clf.tree_.feature[node]]
        threshold = clf.tree_.threshold[node]
        
        # Left branch (<= threshold)
        left_rule = f"{feature_name} <= {threshold:.2f}"
        recurse(clf.tree_.children_left[node], current_rules + [left_rule])
        
        # Right branch (> threshold)
        right_rule = f"{feature_name} > {threshold:.2f}"
        recurse(clf.tree_.children_right[node], current_rules + [right_rule])

    recurse(0, [])
    
    total_dev_n = len(dev)
    total_mon_n = len(mon)
    
    # Overall metrics
    overall_auc_dev, overall_gini_dev = compute_auc_gini(dev)
    overall_auc_mon, overall_gini_mon = compute_auc_gini(mon)
    
    print(f"Overall Dev AUC: {overall_auc_dev:.4f}, Gini: {overall_gini_dev:.4f}, Bad Rate: {dev['target'].mean():.4f}")
    print(f"Overall Mon AUC: {overall_auc_mon:.4f}, Gini: {overall_gini_mon:.4f}, Bad Rate: {mon['target'].mean():.4f}")
    print("=" * 80)
    
    results = []
    
    for leaf_id, rules in tree_rules.items():
        dev_sub = dev[dev['leaf'] == leaf_id]
        mon_sub = mon[mon['leaf'] == leaf_id]
        
        n_dev = len(dev_sub)
        n_mon = len(mon_sub)
        
        pct_dev = n_dev / total_dev_n
        pct_mon = n_mon / total_mon_n
        
        eps = 1e-6
        psi = (pct_mon - pct_dev) * np.log((pct_mon + eps) / (pct_dev + eps))
        
        br_dev = dev_sub['target'].mean() if n_dev > 0 else np.nan
        br_mon = mon_sub['target'].mean() if n_mon > 0 else np.nan
        
        auc_dev, gini_dev = compute_auc_gini(dev_sub)
        auc_mon, gini_mon = compute_auc_gini(mon_sub)
        
        ead_dev = dev_sub['ead'].sum() if n_dev > 0 else 0
        ead_mon = mon_sub['ead'].sum() if n_mon > 0 else 0
        
        # Format human-readable rule
        clean_rules = []
        for r in rules:
            if '_<= 0.50' in r or '_<= 0.5' in r:
                col = r.split(' <= ')[0]
                val = col.split('_')[-1]
                feat = col.rsplit('_', 1)[0]
                clean_rules.append(f"{feat} != '{val}'")
            elif '_> 0.50' in r or '_> 0.5' in r:
                col = r.split(' > ')[0]
                val = col.split('_')[-1]
                feat = col.rsplit('_', 1)[0]
                clean_rules.append(f"{feat} == '{val}'")
            else:
                clean_rules.append(r)
        
        rule_str = " AND ".join(clean_rules) if clean_rules else "Root"
        
        delta_gini = gini_dev - gini_mon if not (np.isnan(gini_dev) or np.isnan(gini_mon)) else 0
        
        # Composite Severity score = PSI + delta_gini + relative exposure
        severity = psi * 10 + max(0, delta_gini) * 5 + (pct_mon - pct_dev)
        
        results.append({
            'Leaf_ID': leaf_id,
            'Rule': rule_str,
            'Dev_Count': n_dev,
            'Mon_Count': n_mon,
            'Dev_Pct': pct_dev,
            'Mon_Pct': pct_mon,
            'PSI': psi,
            'Dev_BR': br_dev,
            'Mon_BR': br_mon,
            'Dev_AUC': auc_dev,
            'Mon_AUC': auc_mon,
            'Dev_Gini': gini_dev,
            'Mon_Gini': gini_mon,
            'Delta_Gini': delta_gini,
            'Dev_EAD': ead_dev,
            'Mon_EAD': ead_mon,
            'Severity': severity
        })
        
    res_df = pd.DataFrame(results).sort_values(by='Severity', ascending=False)
    
    elapsed = time.perf_counter() - start_time
    
    print("\n--- EXTRACTED SEGMENTS RANKED BY DRIFT SEVERITY ---")
    for idx, row in res_df.iterrows():
        print(f"\nRank Leaf {row['Leaf_ID']}: {row['Rule']}")
        print(f"  Population: Dev={row['Dev_Count']} ({row['Dev_Pct']:.1%}), Mon={row['Mon_Count']} ({row['Mon_Pct']:.1%}) | PSI={row['PSI']:.4f}")
        print(f"  Bad Rate: Dev={row['Dev_BR']:.2%}, Mon={row['Mon_BR']:.2%}")
        print(f"  AUC: Dev={row['Dev_AUC']:.4f}, Mon={row['Mon_AUC']:.4f} | Gini: Dev={row['Dev_Gini']:.4f}, Mon={row['Mon_Gini']:.4f} (Delta: {row['Delta_Gini']:.4f})")
        print(f"  Exposure (EAD): Dev=${row['Dev_EAD']:,.0f}, Mon=${row['Mon_EAD']:,.0f}")

    print(f"\nExecution completed in {elapsed:.4f} seconds.")

if __name__ == '__main__':
    main()
