"""
ADIS - Data Cleaning Module
Handles missing values, duplicates, outlier detection, and type coercions.
All changes are logged for the explanation engine.
"""
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class CleaningLog:
    """Tracks every change made during cleaning."""
    
    def __init__(self):
        self.entries: List[Dict] = []
    
    def log(self, operation: str, column: Optional[str], detail: str, rows_affected: int = 0):
        self.entries.append({
            "operation": operation,
            "column": column,
            "detail": detail,
            "rows_affected": rows_affected,
        })
    
    def to_list(self) -> List[Dict]:
        return self.entries


def handle_missing_values(
    df: pd.DataFrame,
    column_info: Dict[str, Dict],
    log: CleaningLog,
    strategy: str = "auto",
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Impute missing values based on detected column type and chosen strategy.
    
    Strategies:
        - 'auto': median for numeric, mode for categorical
        - 'median': use median for all numeric
        - 'mean': use mean for all numeric
        - 'mode': use mode for all
        - 'knn': KNN imputation for numeric (falls back to mode for categorical)
        - 'drop': drop rows with any missing values
    """
    df = df.copy()
    
    numeric_cols = [c for c, info in column_info.items() 
                    if info["detected_type"] == "numeric" and c in df.columns]
    categorical_cols = [c for c, info in column_info.items() 
                        if info["detected_type"] in ("categorical", "boolean") and c in df.columns]
    datetime_cols = [c for c, info in column_info.items() 
                     if info["detected_type"] == "datetime" and c in df.columns]
    
    if strategy == "drop":
        initial = len(df)
        df = df.dropna()
        dropped = initial - len(df)
        log.log("drop_missing_rows", None, f"Dropped {dropped} rows containing missing values.", dropped)
        return df
    
    # Handle KNN imputation for numeric columns
    if strategy == "knn" and numeric_cols:
        missing_numeric = [c for c in numeric_cols if df[c].isna().any()]
        if missing_numeric:
            try:
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df_numeric = df[missing_numeric].copy()
                imputed_vals = imputer.fit_transform(df_numeric)
                df[missing_numeric] = imputed_vals
                for col in missing_numeric:
                    n = column_info[col]["missing_count"]
                    log.log("knn_imputation", col, 
                            f"KNN (k={knn_neighbors}) imputed {n} missing numeric values.", n)
            except ImportError:
                strategy = "auto"
                log.log("fallback", None, "sklearn not available; falling back to auto strategy.", 0)
    
    # Numeric imputation
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        
        if strategy in ("auto", "median") or strategy == "knn":
            fill_val = df[col].median()
            method = "median"
        elif strategy == "mean":
            fill_val = df[col].mean()
            method = "mean"
        elif strategy == "mode":
            fill_val = df[col].mode()[0] if not df[col].mode().empty else 0
            method = "mode"
        else:
            fill_val = df[col].median()
            method = "median"
        
        df[col] = df[col].fillna(fill_val)
        log.log(f"{method}_imputation", col, 
                f"Filled {n_missing} missing values with {method} ({fill_val:.4g}).", n_missing)
    
    # Categorical imputation (always mode)
    for col in categorical_cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode_val)
        log.log("mode_imputation", col, 
                f"Filled {n_missing} missing categorical values with mode ('{mode_val}').", n_missing)
    
    # Datetime: forward-fill then back-fill
    for col in datetime_cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        df[col] = df[col].ffill().bfill()
        log.log("ffill_bfill", col, 
                f"Forward/back-filled {n_missing} missing datetime values.", n_missing)
    
    # Text and ID columns: fill with placeholder
    for col in df.columns:
        if col not in numeric_cols and col not in categorical_cols and col not in datetime_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna("__MISSING__")
                log.log("placeholder_fill", col, 
                        f"Filled {n_missing} missing text/id values with '__MISSING__'.", n_missing)
    
    return df


def remove_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Remove fully duplicate rows."""
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    
    if removed > 0:
        log.log("remove_duplicates", None, 
                f"Removed {removed} duplicate rows ({round(removed/initial_count*100, 2)}% of data).", 
                removed)
    else:
        log.log("remove_duplicates", None, "No duplicate rows found.", 0)
    
    return df


def detect_and_handle_outliers(
    df: pd.DataFrame,
    column_info: Dict[str, Dict],
    log: CleaningLog,
    method: str = "iqr",
    action: str = "clip",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outliers using IQR or Z-score, then clip or remove them.
    
    Args:
        method: 'iqr' or 'zscore'
        action: 'clip' (cap to bounds) or 'remove' (drop rows)
        z_threshold: Z-score cutoff (default 3.0)
        iqr_multiplier: IQR fence multiplier (default 1.5)
    """
    df = df.copy()
    numeric_cols = [c for c, info in column_info.items() 
                    if info["detected_type"] == "numeric" and c in df.columns]
    
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        if method == "iqr":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
        else:  # zscore
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                continue
            lower = mean - z_threshold * std
            upper = mean + z_threshold * std
        
        col_outliers = ((df[col] < lower) | (df[col] > upper))
        n_out = col_outliers.sum()
        
        if n_out == 0:
            continue
        
        if action == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
            log.log("outlier_clip", col, 
                    f"Clipped {n_out} outliers using {method.upper()} "
                    f"(bounds: [{lower:.4g}, {upper:.4g}]).", n_out)
        else:
            outlier_mask |= col_outliers
    
    if action == "remove" and outlier_mask.any():
        n_removed = outlier_mask.sum()
        df = df[~outlier_mask]
        log.log("outlier_remove", None, 
                f"Removed {n_removed} rows containing outliers ({method.upper()} method).", n_removed)
    
    return df


def fix_data_types(df: pd.DataFrame, column_info: Dict[str, Dict], log: CleaningLog) -> pd.DataFrame:
    """Coerce columns to their detected types where beneficial."""
    df = df.copy()
    
    for col, info in column_info.items():
        if col not in df.columns:
            continue
        detected = info["detected_type"]
        
        if detected == "datetime" and df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                log.log("type_coercion", col, 
                        "Converted column to datetime dtype.", 0)
            except Exception:
                pass
        
        elif detected == "numeric" and df[col].dtype == object:
            # Try to clean and convert strings like "$1,234.56" 
            cleaned = df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.isna().sum() < df[col].isna().sum() + len(df) * 0.1:
                df[col] = converted
                log.log("type_coercion", col, 
                        "Coerced string column to numeric (removed non-numeric characters).", 0)
        
        elif detected == "boolean":
            bool_map = {
                'true': True, 'false': False, 'yes': True, 'no': False,
                'y': True, 'n': False, '1': True, '0': False,
                1: True, 0: False
            }
            if df[col].dtype == object:
                df[col] = df[col].str.lower().map(bool_map)
                log.log("type_coercion", col, 
                        "Converted boolean-like column to bool dtype.", 0)
    
    return df


def strip_whitespace(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string columns."""
    df = df.copy()
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    
    affected = []
    for col in str_cols:
        original = df[col].copy()
        df[col] = df[col].str.strip()
        if not df[col].equals(original):
            affected.append(col)
    
    if affected:
        log.log("strip_whitespace", None, 
                f"Stripped whitespace from {len(affected)} columns: {affected}", 0)
    
    return df


def run_cleaning(
    df: pd.DataFrame,
    column_info: Dict[str, Dict],
    strategy: str = "auto",
    outlier_method: str = "iqr",
    outlier_action: str = "clip",
) -> Dict[str, Any]:
    """
    Full cleaning pipeline.
    
    Returns:
        Dict with cleaned df, cleaning log, and explanation.
    """
    log = CleaningLog()
    initial_shape = df.shape
    
    # Step 1: Strip whitespace
    df = strip_whitespace(df, log)
    
    # Step 2: Fix data types
    df = fix_data_types(df, column_info, log)
    
    # Step 3: Remove duplicates
    df = remove_duplicates(df, log)
    
    # Step 4: Handle missing values
    df = handle_missing_values(df, column_info, log, strategy=strategy)
    
    # Step 5: Detect & handle outliers
    df = detect_and_handle_outliers(df, column_info, log, 
                                     method=outlier_method, action=outlier_action)
    
    final_shape = df.shape
    rows_changed = initial_shape[0] - final_shape[0]
    
    explanation = _generate_cleaning_explanation(log, initial_shape, final_shape, 
                                                   strategy, outlier_method, outlier_action)
    
    return {
        "df": df,
        "log": log.to_list(),
        "initial_shape": initial_shape,
        "final_shape": final_shape,
        "rows_removed": rows_changed,
        "explanation": explanation,
        "step": "cleaning",
    }


def _generate_cleaning_explanation(log: CleaningLog, initial_shape, final_shape,
                                    strategy, outlier_method, outlier_action) -> dict:
    """Generate human-readable explanation of all cleaning steps."""
    total_ops = len(log.entries)
    imputation_ops = [e for e in log.entries if "imputation" in e["operation"] or "fill" in e["operation"]]
    outlier_ops = [e for e in log.entries if "outlier" in e["operation"]]
    
    return {
        "title": "Data Cleaning",
        "what_happened": (
            f"Applied {total_ops} cleaning operations. "
            f"Dataset went from {initial_shape[0]:,}×{initial_shape[1]} → "
            f"{final_shape[0]:,}×{final_shape[1]}."
        ),
        "why": (
            "Raw data almost always contains noise: duplicates skew statistics, "
            "missing values break most ML algorithms, and outliers distort model weights. "
            "Systematic cleaning improves both reliability and model performance."
        ),
        "decisions": {
            "missing_value_strategy": strategy,
            "outlier_method": outlier_method,
            "outlier_action": outlier_action,
            "imputation_operations": len(imputation_ops),
            "outlier_operations": len(outlier_ops),
        },
        "change_log": log.to_list(),
        "impact": (
            f"Removed {initial_shape[0] - final_shape[0]} duplicate/outlier rows. "
            f"Imputed {sum(e['rows_affected'] for e in imputation_ops)} missing cell values. "
            f"{'Clipped' if outlier_action == 'clip' else 'Removed'} outliers via {outlier_method.upper()}."
        ),
    }
