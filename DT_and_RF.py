import pandas as pd
import numpy as np
import re
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error



ELEMENTS = ['C', 'H', 'O', 'N', 'S']


def parse_formula(formula):
    counts = {el: 0 for el in ELEMENTS}
    if not isinstance(formula, str):
        return pd.Series(counts)
    formula = formula.strip()
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    for element, count in matches:
        if element in counts:
            counts[element] = int(count) if count != '' else 1
    return pd.Series(counts)


def _numeric_feature_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def main(train_path='data/train.txt', test_path='data/test.txt'):
    print('Loading', train_path)
    df_train = pd.read_csv(train_path, sep='\t')
    print('Loading', test_path)
    df_test = pd.read_csv(test_path, sep='\t')

    # keep only rows that have a formula
    df_train = df_train.dropna(subset=['formula']).reset_index(drop=True)
    df_test = df_test.dropna(subset=['formula']).reset_index(drop=True)

    if df_train.empty or df_test.empty:
        print('Train or test is empty; aborting')
        return

    # targets
    counts_train = df_train['formula'].apply(parse_formula)
    counts_test = df_test['formula'].apply(parse_formula)

    # select numeric features present in both sets
    num_train = _numeric_feature_columns(df_train)
    num_cols = [c for c in num_train if c in df_test.columns and c != 'formula']


    if not num_cols:
        mz_cols = [c for c in df_train.columns if c.startswith('mz')]
        if mz_cols:
            df_train['mz_mean'] = df_train[mz_cols].replace(0, np.nan).mean(axis=1).fillna(0)
            df_test['mz_mean'] = df_test[[c for c in df_test.columns if c.startswith('mz')]].replace(0, np.nan).mean(axis=1).fillna(0)
            num_cols = ['mz_mean']
        else:
            print('No numeric features found in train/test to use as model inputs')
            return

    X_train = df_train[num_cols].fillna(0).values
    X_test = df_test[num_cols].fillna(0).values
    print(X_train.shape)

    regressors = [
        ("DecisionTree", DecisionTreeRegressor(random_state=42)),
        ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
    ]

    results = []
    os.makedirs('output', exist_ok=True)

    for name, model in regressors:
        print(f'Training {name} to predict CHONS counts from numeric features: {num_cols}')

        # wrap linear or otherwise single-output regressors into multioutput wrapper
        if name in ("Ridge", "ElasticNet", "AdaBoost"):
            estimator = MultiOutputRegressor(model)
        else:
            estimator = model

        estimator.fit(X_train, counts_train.values)
        y_pred = estimator.predict(X_test)

        # post-process predictions to integer non-negative counts
        y_pred_int = np.rint(y_pred).astype(int)
        y_pred_int = np.clip(y_pred_int, 0, None)

        # evaluation: MAE per element and per-element exact-match accuracy
        maes = {}
        per_elem_acc = {}
        for i, el in enumerate(ELEMENTS):
            maes[el] = mean_absolute_error(counts_test.iloc[:, i].values, y_pred[:, i])
            per_elem_acc[el] = np.mean(y_pred_int[:, i] == counts_test.iloc[:, i].values)

        correct_count = int(np.sum(np.all(y_pred_int == counts_test.values, axis=1)))
        total_count = int(counts_test.shape[0])
        exact_match = correct_count / total_count if total_count > 0 else 0.0

        # print brief summary
        print(f'{name} MAE: ' + ', '.join([f"{el}={maes[el]:.4f}" for el in ELEMENTS]))
        print(f'{name} Exact formula (all counts) match rate: {exact_match:.4f} (test set)')

        # save results row
        row = {
            'model': name,
            'exact_match': exact_match,
            'num_correct': correct_count,
            'total': total_count,
        }
        for el in ELEMENTS:
            row[f'mae_{el}'] = maes[el]
            row[f'accuracy_{el}'] = per_elem_acc[el]

        results.append(row)

    # save all results to CSV
    df_results = pd.DataFrame(results)
    out_path = os.path.join('output', 'regressor_metrics.csv')
    df_results.to_csv(out_path, index=False)
    print(f'Saved regressor metrics to {out_path}')


if __name__ == '__main__':
    main()

