import os
import json
import pandas as pd
import qonnx.custom_op.registry as registry
from typing import List, Optional
from qonnx.core.datatype import DataType
import onnx.numpy_helper as np_helper

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

# TODO: just for testing
export_path = "/mnt/c/Users/felix/Downloads/paper_figures/"

def make_hashable(obj):
    """Recursively convert lists/dicts to tuples so the result is hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj


class QoREstimator:
    def __init__(self, operator, target_metric, database_path = None, include_commit=None, exclude_commit=None, include_pipeline_id=None, exclude_pipeline_id=None, **kwargs):
        if database_path is None:
            self.database_path = os.environ.get("FINN_MICROBENCHMARK_DATABASE")
        else:
            self.database_path = database_path
        
        self.operator = operator
        self.target_col = target_metric # e.g. "metrics.synth.resources.LUT"
        self.data = None
        self.test_data = None

        self.load_database(include_commit, exclude_commit, include_pipeline_id, exclude_pipeline_id, **kwargs)

    def get_features(self):
        # Select features based on the operator
        features = {
            "mvau": ["params.backend",
                     "params.mem_mode", "params.ram_style", "params.ram_style_thr",
                     # input data types either as string or bitwidth + sign:
                     #"params.idt", "params.wdt",
                     "idt_bitwidth", "idt_sign", "wdt_bitwidth", "wdt_sign",
                     # activation type (or None/null) as string for now:
                     "params.act",
                     "params.mw", "params.mh",
                     # folding either as SF/NF or SIMD/PE:
                     #"params.sf", "params.nf",
                     "dut_info.simd", "dut_info.pe",
                     # weight sparsity as feature:
                     "dut_info.zero_weights", 
                     #"dut_info.easy_weights",
                     ],
        }
        return features[self.operator]

    def map_features(self, model, node_type, node_inst):
        node = node_inst.onnx_node

        features_dict = {}
        if node_type in ["MVAU_hls", "MVAU_rtl"]:
            features_dict["params.backend"] = "hls" if node_type == "MVAU_hls" else "rtl"
            features_dict["params.mem_mode"] = node_inst.get_nodeattr("mem_mode")
            features_dict["params.ram_style"] = node_inst.get_nodeattr("ram_style")
            features_dict["params.ram_style_thr"] = node_inst.get_nodeattr("ram_style_thresholds")

            idt_str = node_inst.get_nodeattr("inputDataType")
            wdt_str = node_inst.get_nodeattr("weightDataType")
            features_dict["idt_bitwidth"] = DataType[idt_str].bitwidth()
            features_dict["idt_sign"] = DataType[idt_str].signed()
            features_dict["wdt_bitwidth"] = DataType[wdt_str].bitwidth()
            features_dict["wdt_sign"] = DataType[wdt_str].signed()

            if node_inst.get_nodeattr("noActivation") == 1:
                features_dict["params.act"] = None
            else:
                features_dict["params.act"] = node_inst.get_nodeattr("outputDataType")
            features_dict["params.mw"] = node_inst.get_nodeattr("MW")
            features_dict["params.mh"] = node_inst.get_nodeattr("MH")
            features_dict["dut_info.simd"] = node_inst.get_nodeattr("SIMD")
            features_dict["dut_info.pe"] = node_inst.get_nodeattr("PE")

            mvau_w_init = [x for x in model.graph.initializer if x.name == node.input[1]][0]
            W = np_helper.to_array(mvau_w_init)
            num_zeros = (W == 0).sum()
            features_dict["dut_info.zero_weights"] = round(num_zeros / W.size, 2)                    

        return pd.DataFrame.from_dict([features_dict])

    # TODO: remove later, only for testing purposes
    def map_features_test(self, node_type, node_inst):
        features_dict = {}
        if node_type in ["MVAU_hls", "MVAU_rtl"]:
            features_dict["params.backend"] = "hls" if node_type == "MVAU_hls" else "rtl"
            features_dict["params.mem_mode"] = node_inst.get_nodeattr("mem_mode")
            features_dict["params.ram_style"] = node_inst.get_nodeattr("ram_style")
            features_dict["params.ram_style_thr"] = node_inst.get_nodeattr("ram_style_thresholds")
            features_dict["params.idt"] = node_inst.get_nodeattr("inputDataType")
            features_dict["params.wdt"] = node_inst.get_nodeattr("weightDataType")
            if node_inst.get_nodeattr("noActivation") == 1:
                features_dict["params.act"] = None
            else:
                features_dict["params.act"] = node_inst.get_nodeattr("outputDataType")
            features_dict["params.mw"] = node_inst.get_nodeattr("MW")
            features_dict["params.mh"] = node_inst.get_nodeattr("MH")
            features_dict["params.sf"] = node_inst.get_nodeattr("MW") // node_inst.get_nodeattr("SIMD")
            features_dict["params.nf"] = node_inst.get_nodeattr("MH") // node_inst.get_nodeattr("PE")

        return pd.DataFrame.from_dict([features_dict]), features_dict

    def get_regressor_grid(self):
        """Return the list of (regressor class, param_grid) tuples used for model selection."""
        #TODO: use random_state instance instead of integer to allow the models to use different seeds for each fold?
        return [
            (KNeighborsRegressor, {
                "regressor__n_neighbors": [3, 5, 7, 10, 15, 20],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2],
                "regressor__algorithm": ["auto", "ball_tree", "kd_tree"]
            }),

            # Linear models don't perform at all for this use case
            # (LinearRegression, {
            #     "regressor__positive": [False, True],
            #     "regressor__fit_intercept": [True, False]
            # }),
            # (Ridge, {
            #     "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            #     "regressor__fit_intercept": [True, False],
            #     "regressor__solver": ["auto", "svd", "cholesky", "lsqr"]
            # }),
            # (Lasso, {
            #     "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            #     "regressor__fit_intercept": [True, False],
            #     "regressor__max_iter": [1000, 2000],
            #     "regressor__selection": ["cyclic", "random"]
            # }),

            (DecisionTreeRegressor, {
                "regressor__max_depth": [None, 3, 5, 7, 10, 15],
                "regressor__min_samples_split": [2, 5, 10, 20],
                "regressor__min_samples_leaf": [1, 2, 5, 10],
                "regressor__max_features": ["sqrt", "log2", None],
                "regressor__random_state": [42]
            }),

            (RandomForestRegressor, {
                "regressor__n_estimators": [50, 100, 200, 300],
                "regressor__max_depth": [None, 5, 10, 15, 20],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__max_features": ["sqrt", "log2", None],
                "regressor__bootstrap": [True, False],
                "regressor__random_state": [42]
            }),
            (GradientBoostingRegressor, {
                "regressor__n_estimators": [50, 100, 200, 300],
                "regressor__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "regressor__max_depth": [3, 4, 5, 6],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__subsample": [0.8, 0.9, 1.0],
                "regressor__random_state": [42]
            }),
            (HistGradientBoostingRegressor, {
                "regressor__max_iter": [100, 200, 300, 500],
                "regressor__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "regressor__max_depth": [None, 3, 5, 7, 10],
                "regressor__min_samples_leaf": [1, 5, 10, 20],
                "regressor__l2_regularization": [0.0, 0.1, 1.0],
                "regressor__random_state": [42]
            }),
            # (SVR, {
            #     "regressor__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            #     "regressor__kernel": ["rbf", "linear", "poly"],
            #     "regressor__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            #     "regressor__epsilon": [0.01, 0.1, 0.2, 0.5],
            #     "regressor__degree": [2, 3, 4]  # Only used for poly kernel
            # }),
            # (MLPRegressor, {
            #     #"regressor__hidden_layer_sizes": [(50,), (100,), (150,), (50, 50), (100, 50), (100, 100)],
            #     "regressor__hidden_layer_sizes": [(100, 100), (200, 200)],
            #     #"regressor__activation": ["relu", "tanh", "logistic"],
            #     "regressor__activation": ["relu", "tanh"],
            #     #"regressor__alpha": [0.0001, 0.001, 0.01, 0.1],
            #     #"regressor__alpha": [0.001, 0.01],
            #     #"regressor__learning_rate": ["constant", "adaptive"],
            #     #"regressor__learning_rate_init": [0.001, 0.01, 0.1],
            #     #"regressor__max_iter": [300, 500, 800],
            #     "regressor__max_iter": [300],
            #     "regressor__random_state": [42]
            # }),
        ]

    def load_database(self, include_commit: Optional[List[str]] = None, exclude_commit: Optional[List[str]] = None, include_pipeline_id: Optional[List[int]] = None, exclude_pipeline_id: Optional[List[int]] = None, **kwargs):
        """Load the empirical QOR database from the specified path and aggregate runs with metadata.
        Optionally filter the data based on given keyword arguments and optional include/exclude 
        lists for commit and pipeline_id."""

        if not self.database_path or not os.path.exists(self.database_path):
            raise ValueError("Database path is not set or does not exist.")
        
        all_rows = []
        files = sorted([f for f in os.listdir(os.path.join(self.database_path, self.operator)) if f.endswith('.json')])
        for path in files:
            with open(os.path.join(self.database_path, self.operator, path), 'r') as file:
                entry = json.load(file)
                # Separate global metadata and runs
                runs = entry.get("runs", [])
                metadata = {k: v for k, v in entry.items() if k != "runs"}
                for run in runs:
                    row = {**metadata, **run}
                    all_rows.append(row)

        print(f"Loaded microbenchmark data for operator {self.operator} from {len(files)} files")

        # Use json_normalize to flatten nested dicts into columns
        self.data = pd.json_normalize(all_rows)

        # set types for known columns
        self.data["pipeline_id"] = self.data["pipeline_id"].astype(int)

        # remove unneeded columns
        if "params.generate_outputs" in self.data.columns:
            self.data = self.data.drop(columns=["params.generate_outputs"])
        if "params.store_results_in_dvc_experiment" in self.data.columns:
            self.data = self.data.drop(columns=["params.store_results_in_dvc_experiment"])
        # num_inp_vectors ([N] or [N, H, W]) doesn't really affect the MVAU implementation itself, as it is only the outer loop bound
        if "params.nhw" in self.data.columns:
            self.data = self.data.drop(columns=["params.nhw"])
        # convert nhw list to individual columns
        #self.data[['params.n', 'params.h', 'params.w']] = pd.DataFrame(self.data['params.nhw'].tolist(), index=self.data.index)

        # Apply manual filters
        print(f"Total number of database entries: {len(self.data)}")
        # Apply include/exclude filters for commit (support short hashes)
        if include_commit is not None:
            self.data = self.data[self.data["commit"].str.startswith(tuple(include_commit))]
        if exclude_commit is not None:
            self.data = self.data[~self.data["commit"].str.startswith(tuple(exclude_commit))]
        # Apply include/exclude filters for pipeline_id
        if include_pipeline_id is not None:
            self.data = self.data[self.data["pipeline_id"].isin(include_pipeline_id)]
        if exclude_pipeline_id is not None:
            self.data = self.data[~self.data["pipeline_id"].isin(exclude_pipeline_id)]
        # Apply additional filters from kwargs
        for key, value in kwargs.items():
            if key in self.data.columns:
                self.data = self.data[self.data[key] == value]
            else:
                raise KeyError(f"Column '{key}' not found in data.")
        self.data = self.data
        
        # Filter out skipped and failed runs
        print(f"Total number of database entries after applying manual filters: {len(self.data)}")
        n_skipped = (self.data["metrics.status"] == "skipped").sum()
        n_failed = (self.data["metrics.status"] == "failed").sum()
        print(f"Filtering out {n_skipped} 'skipped' and {n_failed} 'failed' runs")
        self.data = self.data[(self.data["metrics.status"] != "skipped") & (self.data["metrics.status"] != "failed")]

        # Filter out broken experiments
        # 1) RTL MVAU with PE=1 (instrumentation degeneration bug, output stream unconnected)
        if "params.backend" in self.data.columns and "dut_info.pe" in self.data.columns:
            rtl_pe1_mask = (self.data["params.backend"] == "rtl") & (self.data["dut_info.pe"] == 1)
            n_filtered_rtl_pe1 = rtl_pe1_mask.sum()
            self.data = self.data[~rtl_pe1_mask]
            print(f"Filtered out {n_filtered_rtl_pe1} rows containing broken experiments")

        num_rows_before = len(self.data)
        # Deduplicate: keep only the newest row for each unique set of params.* columns
        if not self.data.empty and "date" in self.data.columns:
            # Find all columns that start with 'params.' and sort them for consistent tuple keys
            params_cols = sorted([col for col in self.data.columns if col.startswith("params.")])
            if params_cols:
                # Create a tuple key from all params.* columns for each row, making each element hashable
                self.data["params_tuple"] = self.data[params_cols].apply(lambda row: tuple(make_hashable(v) for v in row), axis=1)
                self.data = self.data.sort_values(["date", "params_tuple"], ascending=[False, True])
                self.data = self.data.drop_duplicates(subset=["params_tuple"], keep="first")
        print(f"Removed {num_rows_before - len(self.data)} duplicate entries")
        print(f"Final number of entries: {len(self.data)}")
        # Reset index to ensure it is clean after removing rows
        self.data = self.data.reset_index(drop=True)
        # Now self.data contains one row per run, with all nested dicts flattened into columns and deduplication based on params.*

        # Add artificial column: load power minus baseline power
        col = 'metrics.measurement.power.power_pl_ps_load'
        self.data['power'] = self.data[col] - (self.data[col].min() - 0.01) # avoid zero values
        print(f"Measured baseline (min observed) power {self.data[col].min()}, shift all measurements by {self.data[col].min() - 0.01}")

        # Transform certain feature columns for better model performance
        self.data["idt_bitwidth"] = self.data["params.idt"].apply(lambda x: DataType[x].bitwidth())
        self.data["idt_sign"] = self.data["params.idt"].apply(lambda x: DataType[x].signed())

        self.data["wdt_bitwidth"] = self.data["params.wdt"].apply(lambda x: DataType[x].bitwidth())
        self.data["wdt_sign"] = self.data["params.wdt"].apply(lambda x: DataType[x].signed())

        # TODO: special handling for activation type (can be None/null)?
        #self.data["act_bitwidth"] = self.data["act"].apply(lambda x: DataType[x].bitwidth)

        # Select features depending on the operator
        self.feature_cols = self.get_features()
        # Automatically determine which columns are categorical
        self.categorical_cols = [col for col in self.feature_cols if self.data[col].dtype == object or self.data[col].dtype.name == "category"]

    def split_data(self, train_size=1.0, random_state=42):
        """Split the data into training and testing sets. If train_size=1.0, all data is assigned to train, test is empty."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_database() first.")

        if train_size == 1.0:
            train_data = self.data.copy()
            test_data = self.data.iloc[0:0].copy()  # empty DataFrame with same columns
        else:
            train_data, test_data = train_test_split(self.data, train_size=train_size, random_state=random_state)
        self.data = train_data
        self.test_data = test_data
        return train_data, test_data
    
    def make_pipeline(self, model_cls=None, model_kwargs=None):
        """Create a sklearn pipeline for regression, but do not fit it yet."""
        if self.data is None or self.data.empty:
            raise ValueError("Data not loaded or empty. Call load_database() and ensure data is present.")
        if self.target_col is None:
            raise ValueError("target_col must be specified.")
        if self.feature_cols is None:
            raise ValueError("feature_cols must be specified.")

        # Build preprocessor
        # TODO: try other scalers?
        numeric_cols = [col for col in self.feature_cols if col not in self.categorical_cols]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
            ]
        )

        # Choose regressor
        if model_cls is None:
            model_cls = KNeighborsRegressor
        if model_kwargs is None:
            model_kwargs = {}
        regressor = model_cls(**model_kwargs)

        # Assemble pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        self.pipeline = pipeline
        return pipeline

    def fit_pipeline(self, pipeline):
        """Fit a given pipeline on self.data."""
        X = self.data[self.feature_cols]
        y = self.data[self.target_col].values
        pipeline.fit(X, y)
        print(f"Fitted {pipeline.named_steps['regressor'].__class__.__name__} in pipeline on {X.shape[0]} samples with features: {self.feature_cols} (categorical: {self.categorical_cols}) and target: {self.target_col}")
        self.pipeline = pipeline
        return pipeline

    def predict(self, X):
        """Predict using the fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_pipeline() or auto_select_regressor() first.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        if not all(col in X.columns for col in self.feature_cols):
            raise ValueError(f"Input DataFrame must contain all feature columns: {self.feature_cols}")
        prediction = self.pipeline.predict(X[self.feature_cols])
        if "power" in self.target_col:
            return prediction.item()
        else:
            return int(prediction.item())

    def auto_select_regressor(self, scoring="r2", cv=5, verbose=True, n_jobs=-1, plot_scores=False, n_repeats=1):
        """Try different regressors with hyperparameter optimization and select the best one. 
        Optionally plot all regressor scores as a bar chart and box plot, including additional 
        metrics (MAPE, MSE, RMSE) for comprehensive evaluation."""
        regressors = self.get_regressor_grid()
        best_score = -np.inf
        best_model = None
        best_name = None
        all_scores = []
        all_names = []
        all_worst_scores = []
        all_stds = []
        all_worst_stds = []
        all_best_mins = []
        all_best_maxs = []
        all_worst_mins = []
        all_worst_maxs = []
        best_fold_scores_list = []
        
        # Additional metrics collection
        additional_metrics = {
            'MAPE': [],
            'MSE': [],
            'RMSE': []
        }
        additional_fold_scores = {
            'MAPE': [],
            'MSE': [],
            'RMSE': []
        }
        # New: individual sample metrics collection
        individual_sample_metrics = {
            'MAE': [],
            'MSE': [],
            'RMSE': [],
            'MAPE': []  # Will contain NaN for zero true values
        }
        
        cv_obj = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=42)
        for reg_cls, param_grid in regressors:
            try:
                pipeline = self.make_pipeline(
                    model_cls=reg_cls,
                    model_kwargs={}
                )
                X = self.data[self.feature_cols]
                y = self.data[self.target_col].values
                
                # Compute additional metrics using cross-validation
                def compute_additional_metrics(pipeline, X, y, cv_obj):
                    """Compute MAPE, MSE, RMSE using cross-validation with individual sample predictions"""
                    from sklearn.model_selection import cross_validate, cross_val_predict
                    from sklearn.metrics import make_scorer, mean_squared_error
                    
                    def mape_score(y_true, y_pred):
                        """Mean Absolute Percentage Error"""
                        # Avoid division by zero by replacing zero values with small epsilon
                        y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
                        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
                    
                    def rmse_score(y_true, y_pred):
                        """Root Mean Squared Error"""
                        return np.sqrt(mean_squared_error(y_true, y_pred))
                    
                    # Define custom scorers for fold-wise metrics
                    scorers = {
                        'mape': make_scorer(mape_score, greater_is_better=False),
                        'mse': make_scorer(mean_squared_error, greater_is_better=False),
                        'rmse': make_scorer(rmse_score, greater_is_better=False)
                    }
                    
                    # Perform cross-validation with multiple metrics (fold-wise)
                    cv_results = cross_validate(pipeline, X, y, cv=cv_obj, scoring=scorers, n_jobs=n_jobs)
                    
                    # Generate out-of-fold predictions for individual sample analysis
                    y_pred = cross_val_predict(pipeline, X, y, cv=cv_obj, n_jobs=n_jobs)
                    
                    # Calculate individual sample errors
                    sample_mae = np.abs(y_pred - y)
                    sample_mse = (y_pred - y) ** 2
                    sample_rmse = np.sqrt(sample_mse)
                    
                    # Calculate individual sample MAPE (handle zeros carefully)
                    sample_mape = np.full(len(y), np.nan)
                    non_zero_mask = y != 0
                    if non_zero_mask.sum() > 0:
                        sample_mape[non_zero_mask] = np.abs((y_pred[non_zero_mask] - y[non_zero_mask]) / y[non_zero_mask]) * 100
                    
                    return {
                        'MAPE': -cv_results['test_mape'],  # Fold-wise means (negative to positive)
                        'MSE': -cv_results['test_mse'],
                        'RMSE': -cv_results['test_rmse'],
                        'sample_predictions': y_pred,  # Individual predictions for all samples
                        'sample_mae': sample_mae,      # Individual MAE for all samples
                        'sample_mse': sample_mse,      # Individual MSE for all samples  
                        'sample_rmse': sample_rmse,    # Individual RMSE for all samples
                        'sample_mape': sample_mape     # Individual MAPE for all samples (NaN where y=0)
                    }
                
                if param_grid:
                    search = GridSearchCV(pipeline, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs, return_train_score=False)
                    search.fit(X, y)
                    # Best
                    model = search.best_estimator_
                    best_mean_score = search.best_score_
                    best_idx = search.best_index_
                    std_score = search.cv_results_["std_test_score"][best_idx]
                    n_splits = cv_obj.get_n_splits()
                    best_fold_scores = [search.cv_results_[f"split{i}_test_score"][best_idx] for i in range(n_splits)]
                    best_fold_scores_list.append(best_fold_scores)
                    
                    # Print best hyperparameters for this regressor
                    if verbose:
                        print(f"  🏆 Best hyperparameters for {reg_cls.__name__}:")
                        for param, value in search.best_params_.items():
                            print(f"    • {param}: {value}")
                        print(f"    • Total parameter combinations tested: {len(search.cv_results_['params'])}")
                    
                    # Compute additional metrics for the best model
                    additional_cv_results = compute_additional_metrics(model, X, y, cv_obj)
                    
                    # Get all fold scores for best param set
                    n_splits = cv_obj.get_n_splits()
                    best_fold_scores = [search.cv_results_[f"split{i}_test_score"][best_idx] for i in range(n_splits)]
                    all_best_mins.append(np.min(best_fold_scores))
                    all_best_maxs.append(np.max(best_fold_scores))
                    # Worst
                    mean_test_scores = search.cv_results_["mean_test_score"]
                    worst_idx = np.argmin(mean_test_scores)
                    worst_mean_score = mean_test_scores[worst_idx]
                    worst_std = search.cv_results_["std_test_score"][worst_idx]
                    worst_fold_scores = [search.cv_results_[f"split{i}_test_score"][worst_idx] for i in range(n_splits)]
                    all_worst_mins.append(np.min(worst_fold_scores))
                    all_worst_maxs.append(np.max(worst_fold_scores))
                else:
                    scores = cross_val_score(pipeline, X, y, cv=cv_obj, scoring=scoring, n_jobs=n_jobs)
                    model = pipeline
                    best_mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    # For consistency: worst_mean_score is the mean score (only one parameter set)
                    worst_mean_score = best_mean_score
                    worst_std = std_score
                    best_fold_scores_list.append(scores)
                    all_best_mins.append(np.min(scores))
                    all_best_maxs.append(np.max(scores))
                    all_worst_mins.append(np.min(scores))
                    all_worst_maxs.append(np.max(scores))
                    
                    # Compute additional metrics
                    additional_cv_results = compute_additional_metrics(pipeline, X, y, cv_obj)
                
                # Store additional metrics (fold-wise means)
                for metric_name in ['MAPE', 'MSE', 'RMSE']:
                    additional_metrics[metric_name].append(np.mean(additional_cv_results[metric_name]))
                    additional_fold_scores[metric_name].append(additional_cv_results[metric_name])
                
                # Store individual sample metrics
                individual_sample_metrics['MAE'].append(additional_cv_results['sample_mae'])
                individual_sample_metrics['MSE'].append(additional_cv_results['sample_mse'])
                individual_sample_metrics['RMSE'].append(additional_cv_results['sample_rmse'])
                individual_sample_metrics['MAPE'].append(additional_cv_results['sample_mape'])
                
                if verbose:
                    print(f"{reg_cls.__name__}: mean {scoring} = {best_mean_score:.4f}, worst = {worst_mean_score:.4f}, std = {std_score:.4f}, worst_std = {worst_std:.4f}")
                    print(f"  MAPE: {np.mean(additional_cv_results['MAPE']):.2f}%, MSE: {np.mean(additional_cv_results['MSE']):.4f}, RMSE: {np.mean(additional_cv_results['RMSE']):.4f}")
                
                all_scores.append(best_mean_score)
                all_worst_scores.append(worst_mean_score)
                all_stds.append(std_score)
                all_worst_stds.append(worst_std)
                all_names.append(reg_cls.__name__)
                if best_mean_score > best_score:
                    best_score = best_mean_score
                    best_model = model
                    best_name = reg_cls.__name__
            except Exception as e:
                if verbose:
                    print(f"{reg_cls.__name__} failed: {e}")
                all_scores.append(np.nan)
                all_worst_scores.append(np.nan)
                all_stds.append(np.nan)
                all_worst_stds.append(np.nan)
                all_names.append(reg_cls.__name__)
                best_fold_scores_list.append([])
                # Add NaN for additional metrics (fold-wise)
                for metric_name in ['MAPE', 'MSE', 'RMSE']:
                    additional_metrics[metric_name].append(np.nan)
                    additional_fold_scores[metric_name].append([])
                # Add empty arrays for individual sample metrics
                for metric_name in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                    individual_sample_metrics[metric_name].append(np.array([]))

        if plot_scores:
            # --- Bar plot with error bars (min/max) ---
            x = np.arange(len(all_names))
            plt.figure(figsize=(14, 7))
            best_err = [np.array(all_scores) - np.array(all_best_mins), np.array(all_best_maxs) - np.array(all_scores)]
            plt.bar(x - 0.2, all_scores, 0.4, label='Best parameter set', color='skyblue', yerr=best_err, capsize=5)
            worst_err = [np.array(all_worst_scores) - np.array(all_worst_mins), np.array(all_worst_maxs) - np.array(all_worst_scores)]
            plt.bar(x + 0.2, all_worst_scores, 0.4, label='Worst parameter set', color='salmon', yerr=worst_err, capsize=5)
            plt.ylabel(f"Mean CV {scoring}")
            plt.xlabel("Regressor")
            plt.title("Regressor Scores, Cross-Validated (Bars for Best/Worst Fold)")
            plt.xticks(x, all_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.show()

            # --- Create subplot with multiple box plots (Individual Sample Performance) ---
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Primary scoring metric (top-left) - Keep fold-wise for consistency
            box_data = best_fold_scores_list
            box_labels = all_names
            bp1 = axes[0,0].boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True, showfliers=False, whis=[0, 100])
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
            axes[0,0].set_ylabel(f"CV {scoring}")
            axes[0,0].set_xlabel("Regressor")
            axes[0,0].set_title(f"Cross-Validated {scoring.upper()} Scores (Fold Variance)")
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # MAPE Individual Samples (top-right)
            mape_sample_data = []
            mape_sample_labels = []
            for i, name in enumerate(all_names):
                sample_mape = individual_sample_metrics['MAPE'][i]
                if len(sample_mape) > 0:
                    # Remove NaN values (from zero true values)
                    valid_mape = sample_mape[~np.isnan(sample_mape)]
                    if len(valid_mape) > 0:
                        mape_sample_data.append(valid_mape)
                        mape_sample_labels.append(name)
            
            if mape_sample_data:
                #bp2 = axes[0,1].boxplot(mape_sample_data, labels=mape_sample_labels, patch_artist=True, showmeans=True, showfliers=False, whis=[5, 95])
                bp2 = axes[0,1].boxplot(mape_sample_data, labels=mape_sample_labels, patch_artist=True, showmeans=True, showfliers=True, whis=1.5) # same settings as for FINN/HLS estimator evaluation
                for patch in bp2['boxes']:
                    patch.set_facecolor('lightyellow')
                axes[0,1].set_ylabel("MAPE (%)")
                #axes[0,1].set_ylim(-10, 500) # same settings as for FINN/HLS estimator evaluation
                axes[0,1].set_ylim(-10, 100)
                axes[0,1].set_xlabel("Regressor")
                axes[0,1].set_title("Individual Sample MAPE Distribution")
                axes[0,1].tick_params(axis='x', rotation=45)
            else:
                axes[0,1].text(0.5, 0.5, 'No valid MAPE data\n(all targets are zero)', 
                             ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_title("Individual Sample MAPE Distribution")
            
            # MSE Individual Samples (bottom-left)
            mse_sample_data = [individual_sample_metrics['MSE'][i] for i in range(len(all_names)) if len(individual_sample_metrics['MSE'][i]) > 0]
            mse_sample_labels = [all_names[i] for i in range(len(all_names)) if len(individual_sample_metrics['MSE'][i]) > 0]
            
            if mse_sample_data:
                bp3 = axes[1,0].boxplot(mse_sample_data, labels=mse_sample_labels, patch_artist=True, showmeans=True, showfliers=False, whis=[5, 95])
                for patch in bp3['boxes']:
                    patch.set_facecolor('lightgreen')
                axes[1,0].set_ylabel("MSE")
                axes[1,0].set_xlabel("Regressor")
                axes[1,0].set_title("Individual Sample MSE Distribution")
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # RMSE Individual Samples (bottom-right)
            rmse_sample_data = [individual_sample_metrics['RMSE'][i] for i in range(len(all_names)) if len(individual_sample_metrics['RMSE'][i]) > 0]
            rmse_sample_labels = [all_names[i] for i in range(len(all_names)) if len(individual_sample_metrics['RMSE'][i]) > 0]
            
            if rmse_sample_data:
                bp4 = axes[1,1].boxplot(rmse_sample_data, labels=rmse_sample_labels, patch_artist=True, showmeans=True, showfliers=False, whis=[5, 95])
                for patch in bp4['boxes']:
                    patch.set_facecolor('lightgreen')
                axes[1,1].set_ylabel("RMSE")
                axes[1,1].set_xlabel("Regressor")
                axes[1,1].set_title("Individual Sample RMSE Distribution")
                axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.suptitle("Regressor Performance: Fold Variance vs Individual Sample Distribution", fontsize=16, y=0.98)
            plt.tight_layout()
            plt.savefig(export_path + "regressor_comparison_comprehensive.pdf", bbox_inches="tight")
            plt.show()
            
            # --- Save MAPE subplot as separate PDF ---
            if mape_sample_data:
                plt.figure(figsize=(10, 6))
                bp_mape = plt.boxplot(mape_sample_data, labels=mape_sample_labels, patch_artist=True, showmeans=True, showfliers=True, whis=1.5)
                for patch in bp_mape['boxes']:
                    patch.set_facecolor('lightyellow')
                plt.ylabel("MAPE (%)", fontsize=12)
                plt.ylim(-10, 100)
                plt.xlabel("Regressor", fontsize=12)
                plt.title("Individual Sample MAPE Distribution", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(export_path + "regressor_mape_distribution.pdf", bbox_inches="tight")
                plt.show()
                print(f"Saved MAPE distribution plot to {export_path}regressor_mape_distribution.pdf")
            
            # --- Summary table ---
            print(f"\n{'='*120}")
            print("COMPREHENSIVE PERFORMANCE SUMMARY")
            print(f"{'='*120}")
            print(f"{'Regressor':<15} {'Mean '+scoring.upper():<10} {'Fold MAPE':<10} {'Fold MSE':<10} {'Fold RMSE':<10} {'Sample MAPE':<12} {'Sample MSE':<12} {'Sample RMSE':<12}")
            print("-" * 120)
            for i, name in enumerate(all_names):
                # Fold-wise metrics
                primary_score = all_scores[i] if not np.isnan(all_scores[i]) else "N/A"
                fold_mape = additional_metrics['MAPE'][i] if not np.isnan(additional_metrics['MAPE'][i]) else "N/A"
                fold_mse = additional_metrics['MSE'][i] if not np.isnan(additional_metrics['MSE'][i]) else "N/A"
                fold_rmse = additional_metrics['RMSE'][i] if not np.isnan(additional_metrics['RMSE'][i]) else "N/A"
                
                # Individual sample metrics (mean of all samples)
                sample_mape_arr = individual_sample_metrics['MAPE'][i]
                sample_mse_arr = individual_sample_metrics['MSE'][i]
                sample_rmse_arr = individual_sample_metrics['RMSE'][i]
                
                if len(sample_mape_arr) > 0:
                    valid_mape = sample_mape_arr[~np.isnan(sample_mape_arr)]
                    sample_mape = np.mean(valid_mape) if len(valid_mape) > 0 else "N/A"
                else:
                    sample_mape = "N/A"
                    
                sample_mse = np.mean(sample_mse_arr) if len(sample_mse_arr) > 0 else "N/A"
                sample_rmse = np.mean(sample_rmse_arr) if len(sample_rmse_arr) > 0 else "N/A"
                
                # Format strings
                primary_str = f"{primary_score:.4f}" if isinstance(primary_score, (int, float)) else str(primary_score)
                fold_mape_str = f"{fold_mape:.2f}%" if isinstance(fold_mape, (int, float)) else str(fold_mape)
                fold_mse_str = f"{fold_mse:.4f}" if isinstance(fold_mse, (int, float)) else str(fold_mse)
                fold_rmse_str = f"{fold_rmse:.4f}" if isinstance(fold_rmse, (int, float)) else str(fold_rmse)
                sample_mape_str = f"{sample_mape:.2f}%" if isinstance(sample_mape, (int, float)) else str(sample_mape)
                sample_mse_str = f"{sample_mse:.4f}" if isinstance(sample_mse, (int, float)) else str(sample_mse)
                sample_rmse_str = f"{sample_rmse:.4f}" if isinstance(sample_rmse, (int, float)) else str(sample_rmse)
                
                print(f"{name:<15} {primary_str:<10} {fold_mape_str:<10} {fold_mse_str:<10} {fold_rmse_str:<10} {sample_mape_str:<12} {sample_mse_str:<12} {sample_rmse_str:<12}")
            
            print(f"\nNote: 'Fold' metrics show variance between CV folds (model stability).")
            print(f"      'Sample' metrics show distribution across individual predictions (prediction quality).")
            print(f"      Sample MAPE excludes cases where true value = 0 to avoid division by zero.")
        
        print(f"Best regressor: {best_name} (mean {scoring} = {best_score:.4f})")
        self.pipeline = best_model
        return best_model
    
    def regressor_performance_vs_data_size(self, data_fractions, scoring="r2", cv=5, n_jobs=-1, plot_scores=True, n_repeats=1, test_size=0.2, random_state=42, evaluation_mode="holdout"):
        """Evaluate regressor performance as a function of training data size.
        
        For each fraction in data_fractions (e.g. [0.1, 0.2, ...]), train regressors on 
        a subset of the training data and evaluate performance. Multiple evaluation modes
        are supported to balance between statistical rigor and practical utility.
        
        Args:
            data_fractions: List of fractions of training data to use (e.g. [0.1, 0.2, 0.5, 1.0])
            test_size: Fraction of data to reserve for testing (default 0.2)
            evaluation_mode: How to evaluate models:
                - "holdout": Strict train/test split (most rigorous, default)
                - "full_dataset": Use full dataset for testing (most data, some bias)
                - "both": Compute both and return comparison (recommended for analysis)
            Other args: Same as auto_select_regressor
        """
        regressors = self.get_regressor_grid()
        results = {reg_cls.__name__: [] for reg_cls, _ in regressors}
        if evaluation_mode == "both":
            results_holdout = {reg_cls.__name__: [] for reg_cls, _ in regressors}
            results_full = {reg_cls.__name__: [] for reg_cls, _ in regressors}
        fractions = []
        
        # Split data into train/test once at the beginning
        orig_data = self.data
        if test_size > 0 and evaluation_mode in ["holdout", "both"]:
            train_data, test_data = train_test_split(
                self.data, test_size=test_size, random_state=random_state
            )
        else:
            # If test_size=0 or full_dataset mode, use all data for training
            train_data = self.data
            test_data = self.data
            
        print(f"Evaluation mode: {evaluation_mode}")
        if evaluation_mode == "holdout":
            print(f"Using {len(train_data)} samples for training scaling, {len(test_data)} samples for evaluation")
        elif evaluation_mode == "full_dataset":
            print(f"Using {len(train_data)} samples for training scaling, {len(self.data)} samples for evaluation")
        elif evaluation_mode == "both":
            print(f"Using {len(train_data)} samples for training scaling")
            print(f"Holdout evaluation: {len(test_data)} samples")
            print(f"Full dataset evaluation: {len(self.data)} samples")
        
        # Prepare test features and targets for different evaluation modes
        if evaluation_mode in ["holdout", "both"]:
            X_test_holdout = test_data[self.feature_cols]
            y_test_holdout = test_data[self.target_col].values
        
        if evaluation_mode in ["full_dataset", "both"]:
            X_test_full = self.data[self.feature_cols]
            y_test_full = self.data[self.target_col].values
        
        for frac in data_fractions:
            n_train_samples = int(len(train_data) * frac)
            if n_train_samples < cv:
                print(f"Skipping fraction {frac} (not enough training samples for {cv}-fold CV: {n_train_samples})")
                for reg in results:
                    results[reg].append(np.nan)
                continue
                
            # Sample subset of training data
            train_subset = train_data.sample(n=n_train_samples, random_state=random_state)
            
            print(f"Training on {len(train_subset)} samples (fraction {frac:.2f}), evaluating on {len(test_data)} samples")
            
            cv_obj = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_state)
            
            for reg_cls, param_grid in regressors:
                try:
                    pipeline = self.make_pipeline(
                        model_cls=reg_cls,
                        model_kwargs={}
                    )
                    X_train = train_subset[self.feature_cols]
                    y_train = train_subset[self.target_col].values
                    
                    if param_grid:
                        # Use cross-validation on training subset to find best parameters
                        search = GridSearchCV(pipeline, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs, return_train_score=False)
                        search.fit(X_train, y_train)
                        best_pipeline = search.best_estimator_
                    else:
                        # No hyperparameters to tune, just fit the pipeline
                        pipeline.fit(X_train, y_train)
                        best_pipeline = pipeline
                    
                    # Evaluate the best model based on evaluation mode
                    from sklearn.metrics import get_scorer
                    scorer = get_scorer(scoring)
                    
                    if evaluation_mode == "holdout":
                        test_score = scorer(best_pipeline, X_test_holdout, y_test_holdout)
                        results[reg_cls.__name__].append(test_score)
                    elif evaluation_mode == "full_dataset":
                        test_score = scorer(best_pipeline, X_test_full, y_test_full)
                        results[reg_cls.__name__].append(test_score)
                    elif evaluation_mode == "both":
                        holdout_score = scorer(best_pipeline, X_test_holdout, y_test_holdout)
                        full_score = scorer(best_pipeline, X_test_full, y_test_full)
                        results_holdout[reg_cls.__name__].append(holdout_score)
                        results_full[reg_cls.__name__].append(full_score)
                        # For primary results, use holdout (more conservative)
                        results[reg_cls.__name__].append(holdout_score)
                except Exception as e:
                    print(f"Error with {reg_cls.__name__} at fraction {frac}: {e}")
                    if evaluation_mode == "both":
                        results_holdout[reg_cls.__name__].append(np.nan)
                        results_full[reg_cls.__name__].append(np.nan)
                    results[reg_cls.__name__].append(np.nan)
            
            fractions.append(frac)
        
        # Restore original data
        self.data = orig_data
        if plot_scores:
            if evaluation_mode == "both":
                # Create comparison plot with both evaluation modes
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot holdout results
                for reg_name, scores in results_holdout.items():
                    valid_fractions = [f for f, s in zip(fractions, scores) if not np.isnan(s)]
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if valid_scores:
                        ax1.plot(valid_fractions, valid_scores, marker='o', label=reg_name)
                ax1.set_xlabel('Fraction of Training Data Used')
                ax1.set_ylabel(f'Holdout Test Set {scoring}')
                ax1.set_title('Performance vs. Training Data Size (Holdout Evaluation)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot full dataset results
                for reg_name, scores in results_full.items():
                    valid_fractions = [f for f, s in zip(fractions, scores) if not np.isnan(s)]
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if valid_scores:
                        ax2.plot(valid_fractions, valid_scores, marker='s', label=reg_name)
                ax2.set_xlabel('Fraction of Training Data Used')
                ax2.set_ylabel(f'Full Dataset {scoring}')
                ax2.set_title('Performance vs. Training Data Size (Full Dataset Evaluation)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(export_path + "regressor_training_data_comparison.pdf", bbox_inches="tight")
                plt.show()
                
                # Print comparison summary
                print(f"\n{'='*80}")
                print("EVALUATION MODE COMPARISON SUMMARY")
                print(f"{'='*80}")
                for reg_name in results_holdout.keys():
                    holdout_final = results_holdout[reg_name][-1] if results_holdout[reg_name] and not np.isnan(results_holdout[reg_name][-1]) else np.nan
                    full_final = results_full[reg_name][-1] if results_full[reg_name] and not np.isnan(results_full[reg_name][-1]) else np.nan
                    if not np.isnan(holdout_final) and not np.isnan(full_final):
                        bias = full_final - holdout_final
                        print(f"{reg_name:20s}: Holdout={holdout_final:.4f}, Full={full_final:.4f}, Bias={bias:+.4f}")
            else:
                # Single evaluation mode plot
                plt.figure(figsize=(12, 6))
                for reg_name, scores in results.items():
                    valid_fractions = [f for f, s in zip(fractions, scores) if not np.isnan(s)]
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if valid_scores:
                        plt.plot(valid_fractions, valid_scores, marker='o', label=reg_name)
                
                eval_label = "Holdout Test Set" if evaluation_mode == "holdout" else "Full Dataset"
                plt.xlabel('Fraction of Training Data Used')
                plt.ylabel(f'{eval_label} {scoring}')
                plt.title(f'Regressor Performance vs. Training Data Size ({eval_label} Evaluation)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(export_path + f"regressor_training_data_{evaluation_mode}.pdf", bbox_inches="tight")
                plt.show()
            
        # Prepare return values
        if evaluation_mode == "both":
            return {
                "holdout": (results_holdout, fractions),
                "full_dataset": (results_full, fractions),
                "primary": (results, fractions)  # Primary uses holdout for conservative estimates
            }
        else:
            return results, fractions


# TODO: This could be just an analysis function, but those don't have a constructor
# Proper transformations can't return an arbitrary dict
class empirical_resource_estimation():
    def __init__(self, fpgapart):
        super().__init__()
        self.fpgapart = fpgapart # only used for fallback

        # Fit all estimators in advance
        # TODO: replace auto selection by sensible default
        #estimator = QoREstimator("mvau", "metrics.synth.resources.LUT", exclude_commit=["fcc5d7a3"])
        #estimator.auto_select_regressor(plot_scores=True, n_repeats=1, scoring="neg_mean_absolute_percentage_error")

        import pickle
        path = '/mnt/c/Users/felix/Downloads/paper_figures/saved_models/qor_estimator_mvau_metrics_synth_resources_LUT.pkl'
        with open(path, 'rb') as f:
            estimator = pickle.load(f)
            print(f"Loaded saved MVAU LUT estimator from {path}")

        self.estimators = {
            "MVAU_hls": {"LUT": estimator},
            "MVAU_rtl": {"LUT": estimator},
        }

    def analysis_pass(self, model):
        """Perform empirical resource estimation on a model using the fitted QoREstimators."""
        res_dict = {}
        for node in model.graph.node:
            if is_hls_node(node) or is_rtl_node(node):
                inst = registry.getCustomOp(node)

                res_dict[node.name] = {}
                for resource in ["LUT", "DSP", "BRAM_18K", "URAM"]:
                    if node.op_type in self.estimators and resource in self.estimators[node.op_type]:
                        # Use the QoREstimator for this operator and resource
                        estimator = self.estimators[node.op_type][resource]

                        features = estimator.map_features(model, node.op_type, inst)

                        res_dict[node.name][resource] = estimator.predict(features)
                    else:
                        # Fall back to FINN estimate if no QoREstimator is available
                        res_dict[node.name][resource] = inst.node_res_estimation(self.fpgapart)[resource]

        res_dict["total"] = aggregate_dict_keys(res_dict)
        return res_dict


# TODO: This could be just an analysis function, but those don't have a constructor
# Proper transformations can't return an arbitrary dict
class empirical_power_estimation():
    def __init__(self, fpgapart):
        super().__init__()
        self.fpgapart = fpgapart # only used for fallback

        # Fit all estimators in advance
        # TODO: replace auto selection by sensible default
        #estimator = QoREstimator("mvau", "metrics.measurement.power.power_pl_ps_dyn", exclude_commit=["fcc5d7a3"])
        #estimator.auto_select_regressor(plot_scores=True, n_repeats=1, scoring="r2")

        import pickle
        path = '/mnt/c/Users/felix/Downloads/paper_figures/saved_models/qor_estimator_mvau_power.pkl'
        with open(path, 'rb') as f:
            estimator = pickle.load(f)
            print(f"Loaded saved MVAU power estimator from {path}")

        self.estimators = {
            "MVAU_hls": estimator,
            "MVAU_rtl": estimator,
        }

    def analysis_pass(self, model):
        """Perform empirical power estimation on a model using the fitted QoREstimator."""
        res_dict = {}
        for node in model.graph.node:
            if is_hls_node(node) or is_rtl_node(node):
                inst = registry.getCustomOp(node)

                res_dict[node.name] = {}
                if node.op_type in self.estimators:
                    # Use the QoREstimator for this operator and resource
                    estimator = self.estimators[node.op_type]

                    features = estimator.map_features(model, node.op_type, inst)

                    res_dict[node.name]["power_estimated"] = estimator.predict(features)
                else:
                    # No fall-back available
                    res_dict[node.name]["power_estimated"] = 0

        res_dict["total"] = aggregate_dict_keys(res_dict)
        return res_dict
