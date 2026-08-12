"""
techniques/kmeans.py
====================

K-Means Clustering Segmentation Technique.

Discovers sub-populations by clustering feature space on Development data,
assigning Monitoring data to the nearest centroids, and distilling
human-readable tree rules for each cluster.

Architecture
------------
1. Centralized schema detection
2. Development-only preprocessing fit
3. K-Means / MiniBatchKMeans clustering
4. Monitoring assignment using Development centroids
5. Human-readable rule distillation
6. Centralized candidate filtering
7. Centralized candidate evaluation
8. Feature-level PSI Root Cause analysis
9. Centralized severity scoring
10. Significance-first ranking
11. Portfolio view generation

Important
---------
Root Cause analysis is performed by candidate_evaluation.py using the
schema-derived feature list.

Therefore ID-like columns such as customer_id are excluded from Root Cause
analysis and are not allowed to become the Root_Cause_Feature.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

from models.config import KMeansConfig
from models.result import TechniqueResult

from techniques.base import BaseSegmentationTechnique

from utils.logging_config import get_logger
from utils.schema_detection import (
    detect_schema,
    auto_detect_shap_columns,
)

from utils.preprocessing import (
    canonicalize_path_conditions,
)

from metrics.drift_metrics import (
    detect_shap_shift,
)

from core.candidate_evaluation import (
    evaluate_segment,
)

from core.candidate_filtering import (
    derive_min_abs_count,
    derive_min_support,
    passes_support_filter,
    passes_event_count_filter,
)

from core.candidate_ranking import (
    compute_severity_scores,
    rank_records_significance_first,
    build_portfolio_view,
)

logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

STABILITY_SEED = 777

LARGE_DATA_ROWS = 200_000

MISSING_LABEL = "__MISSING__"


# ============================================================================
# ENCODING
# ============================================================================

def fit_encoder(
    dev_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """
    Fit numerical scaling and categorical encoding using Development data only.

    The fitted encoder is reused unchanged for Monitoring data.

    This prevents Monitoring data from influencing the feature representation
    used by the clustering model.
    """

    # ------------------------------------------------------------------
    # Numerical scaler
    # ------------------------------------------------------------------

    scaler = StandardScaler()

    numeric_medians = (
        dev_df[numeric_cols].median()
        if numeric_cols
        else None
    )

    if numeric_cols:

        numeric_fit = (
            dev_df[numeric_cols]
            .fillna(numeric_medians)
        )

        scaler.fit(
            numeric_fit
        )

        num_fit = scaler

    else:

        num_fit = None

    # ------------------------------------------------------------------
    # Categorical values
    # ------------------------------------------------------------------

    cat_values = {}

    for col in categorical_cols:

        # IMPORTANT:
        # Replace missing values before converting to string.
        vals = (
            dev_df[col]
            .where(
                dev_df[col].notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )

        cat_values[col] = sorted(
            vals.unique()
        )

    return {
        "scaler": num_fit,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_values": cat_values,
        "numeric_medians": numeric_medians,
    }


def transform(
    df: pd.DataFrame,
    enc: dict,
) -> np.ndarray:
    """
    Transform a dataframe using a previously fitted Development encoder.
    """

    parts = []

    # ==================================================================
    # NUMERIC FEATURES
    # ==================================================================

    if enc["numeric_cols"]:

        X_num = (
            df[
                enc["numeric_cols"]
            ]
            .fillna(
                enc["numeric_medians"]
            )
        )

        parts.append(
            enc["scaler"].transform(
                X_num
            )
        )

    # ==================================================================
    # CATEGORICAL FEATURES
    # ==================================================================

    for col in enc["categorical_cols"]:

        vals = (
            df[col]
            .where(
                df[col].notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )

        onehot = (
            pd.get_dummies(
                vals
            )
            .reindex(
                columns=enc[
                    "cat_values"
                ][col],
                fill_value=0,
            )
        )

        parts.append(
            onehot.to_numpy(
                dtype=float
            )
        )

    # ==================================================================
    # COMBINE
    # ==================================================================

    if parts:

        return np.hstack(
            parts
        )

    return np.zeros(
        (
            len(df),
            0,
        )
    )


# ============================================================================
# ESTIMATOR SELECTION
# ============================================================================

def _kmeans_cls(
    n_rows: int,
):
    """
    Use MiniBatchKMeans for very large datasets.
    """

    if n_rows > LARGE_DATA_ROWS:

        return MiniBatchKMeans

    return KMeans


# ============================================================================
# K SELECTION
# ============================================================================

def select_optimal_k(
    X: np.ndarray,
    k_range: range,
    min_cluster_pct: float,
    seed: int,
    X_mon: Optional[np.ndarray] = None,
    drift_aware: bool = True,
    drift_weight: float = 0.4,
) -> tuple[int, pd.DataFrame]:
    """
    Select optimal K using:

        - Silhouette score
        - Calinski-Harabasz score
        - Davies-Bouldin score
        - Clustering stability
        - Development-to-Monitoring cluster drift

    A minimum cluster-size floor is applied before scoring.
    """

    n = len(X)

    if n < 2:

        raise ValueError(
            "K-Means requires at least two Development observations."
        )

    Estimator = _kmeans_cls(
        n
    )

    rows = []

    # ==================================================================
    # EVALUATE EACH K
    # ==================================================================

    for k in k_range:

        if k < 2:
            continue

        if k >= n:
            continue

        # --------------------------------------------------------------
        # First model
        # --------------------------------------------------------------

        n_init = (
            10
            if Estimator is KMeans
            else 3
        )

        km_a = Estimator(
            n_clusters=k,
            random_state=seed,
            n_init=n_init,
        )

        labels_a = (
            km_a.fit_predict(X)
        )

        sizes = (
            np.bincount(
                labels_a,
                minlength=k,
            )
            / n
        )

        min_cluster_size = float(
            sizes.min()
        )

        # --------------------------------------------------------------
        # Minimum cluster-size rejection
        # --------------------------------------------------------------

        if (
            min_cluster_size
            < min_cluster_pct
        ):

            rows.append(
                {
                    "k": k,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "stability_ari": np.nan,
                    "max_cluster_drift": np.nan,
                    "min_cluster_pct": min_cluster_size,
                    "composite": -np.inf,
                    "rejected_reason": (
                        f"smallest cluster "
                        f"{min_cluster_size * 100:.1f}% "
                        f"< floor "
                        f"{min_cluster_pct * 100:.0f}%"
                    ),
                }
            )

            continue

        # --------------------------------------------------------------
        # Stability model
        # --------------------------------------------------------------

        km_b = Estimator(
            n_clusters=k,
            random_state=STABILITY_SEED,
            n_init=n_init,
        )

        labels_b = (
            km_b.fit_predict(X)
        )

        stability = (
            adjusted_rand_score(
                labels_a,
                labels_b,
            )
        )

        # --------------------------------------------------------------
        # Internal clustering metrics
        # --------------------------------------------------------------

        sample_size = min(
            n,
            10_000,
        )

        try:

            sil = silhouette_score(
                X,
                labels_a,
                sample_size=sample_size,
                random_state=seed,
            )

        except Exception:

            sil = np.nan

        try:

            ch = calinski_harabasz_score(
                X,
                labels_a,
            )

        except Exception:

            ch = np.nan

        try:

            db = davies_bouldin_score(
                X,
                labels_a,
            )

        except Exception:

            db = np.nan

        # --------------------------------------------------------------
        # Development vs Monitoring cluster drift
        # --------------------------------------------------------------

        max_drift = np.nan

        if (
            drift_aware
            and X_mon is not None
            and len(X_mon) > 0
        ):

            mon_labels = (
                km_a.predict(
                    X_mon
                )
            )

            dev_pcts = (
                np.bincount(
                    labels_a,
                    minlength=k,
                )
                / len(labels_a)
            )

            mon_pcts = (
                np.bincount(
                    mon_labels,
                    minlength=k,
                )
                / len(mon_labels)
            )

            relative_drifts = (
                np.abs(
                    mon_pcts
                    - dev_pcts
                )
                / np.maximum(
                    dev_pcts,
                    1e-6,
                )
            )

            max_drift = float(
                np.max(
                    relative_drifts
                )
            )

        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "stability_ari": stability,
                "max_cluster_drift": max_drift,
                "min_cluster_pct": min_cluster_size,
                "rejected_reason": None,
            }
        )

    # ==================================================================
    # DIAGNOSTIC DATAFRAME
    # ==================================================================

    diag = pd.DataFrame(
        rows
    )

    if diag.empty:

        raise ValueError(
            "No candidate K values were available."
        )

    valid = diag[
        diag[
            "rejected_reason"
        ].isna()
    ].copy()

    if valid.empty:

        raise ValueError(
            "No candidate k satisfied the minimum cluster-size floor."
        )

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    def norm(series):

        series = series.astype(
            float
        )

        finite_values = series[
            np.isfinite(
                series
            )
        ]

        if finite_values.empty:

            return pd.Series(
                1.0,
                index=series.index,
            )

        min_value = (
            finite_values.min()
        )

        max_value = (
            finite_values.max()
        )

        if max_value <= min_value:

            return pd.Series(
                1.0,
                index=series.index,
            )

        return (
            series
            - min_value
        ) / (
            max_value
            - min_value
        )

    # ==================================================================
    # INTERNAL VALIDITY
    # ==================================================================

    valid["sil_n"] = norm(
        valid["silhouette"]
    )

    valid["ch_n"] = norm(
        valid[
            "calinski_harabasz"
        ]
    )

    valid["db_n"] = (
        1.0
        - norm(
            valid[
                "davies_bouldin"
            ]
        )
    )

    valid["stability_n"] = norm(
        valid[
            "stability_ari"
        ]
    )

    internal_validity = (
        valid[
            [
                "sil_n",
                "ch_n",
                "db_n",
            ]
        ]
        .mean(axis=1)
    )

    # ==================================================================
    # COMPOSITE SCORE
    # ==================================================================

    if (
        drift_aware
        and X_mon is not None
    ):

        valid["drift_n"] = norm(
            valid[
                "max_cluster_drift"
            ]
        )

        valid["composite"] = (
            (1.0 - drift_weight)
            * (
                0.5
                * internal_validity
                + 0.5
                * valid[
                    "stability_n"
                ]
            )
            + drift_weight
            * valid[
                "drift_n"
            ]
        )

    else:

        valid["composite"] = (
            0.5
            * internal_validity
            + 0.5
            * valid[
                "stability_n"
            ]
        )

    # ==================================================================
    # MERGE COMPOSITE INTO DIAGNOSTICS
    # ==================================================================

    diag = diag.merge(
        valid[
            [
                "k",
                "composite",
            ]
        ],
        on="k",
        how="left",
    )

    best_k = int(
        valid.loc[
            valid[
                "composite"
            ].idxmax(),
            "k",
        ]
    )

    return (
        best_k,
        diag.sort_values(
            "k"
        ).reset_index(
            drop=True
        ),
    )


# ============================================================================
# RULE DISTILLATION
# ============================================================================

def distill_cluster_rules(
    raw_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    max_depth: int = 4,
):
    """
    Distill human-readable decision rules for each cluster.

    A shallow DecisionTreeClassifier is trained only on Development data
    to approximate the cluster assignments.

    NOTE:
    The tree is only used for explanation. It does not determine the actual
    cluster assignment.
    """

    X = pd.DataFrame(
        index=raw_df.index
    )

    # ==================================================================
    # NUMERIC FEATURES
    # ==================================================================

    for col in numeric_cols:

        median_value = (
            raw_df[col].median()
        )

        X[col] = (
            raw_df[col]
            .fillna(
                median_value
            )
        )

    # ==================================================================
    # CATEGORICAL FEATURES
    # ==================================================================

    onehot_cols = {}

    for col in categorical_cols:

        vals = (
            raw_df[col]
            .where(
                raw_df[col].notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )

        for value in sorted(
            vals.unique()
        ):

            encoded_col = (
                f"{col}__{value}"
            )

            X[
                encoded_col
            ] = (
                vals == value
            ).astype(int)

            onehot_cols[
                encoded_col
            ] = (
                col,
                value,
            )

    # ==================================================================
    # NO FEATURES
    # ==================================================================

    if X.shape[1] == 0:

        return {
            int(cluster_id): {
                "rule_text": (
                    f"Cluster_{cluster_id} "
                    "(centroid)"
                ),
                "precision": 1.0,
                "recall": 1.0,
            }
            for cluster_id in np.unique(
                cluster_labels
            )
        }

    # ==================================================================
    # DECISION TREE
    # ==================================================================

    min_samples_leaf = max(
        20,
        int(
            0.01
            * len(X)
        ),
    )

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    tree.fit(
        X,
        cluster_labels,
    )

    leaf_id = tree.apply(
        X
    )

    t = tree.tree_

    # ==================================================================
    # FIND PATH TO LEAF
    # ==================================================================

    def path_conditions(
        leaf
    ):

        def walk(
            node,
            path,
        ):

            if node == leaf:

                return path

            if (
                t.children_left[node]
                == t.children_right[node]
            ):

                return None

            feature_index = (
                t.feature[node]
            )

            if (
                feature_index < 0
                or feature_index
                >= len(X.columns)
            ):

                return None

            feature_name = (
                X.columns[
                    feature_index
                ]
            )

            threshold = (
                t.threshold[node]
            )

            # ----------------------------------------------------------
            # Left branch
            # ----------------------------------------------------------

            left_path = walk(
                t.children_left[node],
                path
                + [
                    (
                        feature_name,
                        "<=",
                        threshold,
                    )
                ],
            )

            if left_path is not None:

                return left_path

            # ----------------------------------------------------------
            # Right branch
            # ----------------------------------------------------------

            return walk(
                t.children_right[node],
                path
                + [
                    (
                        feature_name,
                        ">",
                        threshold,
                    )
                ],
            )

        return walk(
            0,
            [],
        )

    # ==================================================================
    # BUILD RULES
    # ==================================================================

    results = {}

    for cluster_id in np.unique(
        cluster_labels
    ):

        cluster_mask = (
            cluster_labels
            == cluster_id
        )

        if not cluster_mask.any():
            continue

        # --------------------------------------------------------------
        # Leaves occupied by this cluster
        # --------------------------------------------------------------

        leaves_for_cluster = (
            pd.Series(
                leaf_id[
                    cluster_mask
                ]
            )
            .value_counts()
        )

        if leaves_for_cluster.empty:

            results[
                int(cluster_id)
            ] = {
                "rule_text": (
                    f"Cluster_{cluster_id}"
                ),
                "precision": 1.0,
                "recall": 1.0,
            }

            continue

        best_leaf = (
            leaves_for_cluster.index[
                0
            ]
        )

        leaf_mask = (
            leaf_id
            == best_leaf
        )

        leaf_count = int(
            leaf_mask.sum()
        )

        precision = (
            (
                cluster_labels[
                    leaf_mask
                ]
                == cluster_id
            ).mean()
            if leaf_count
            else 0.0
        )

        recall = (
            leaves_for_cluster.iloc[
                0
            ]
            / cluster_mask.sum()
        )

        # --------------------------------------------------------------
        # Extract path
        # --------------------------------------------------------------

        raw_conditions = (
            path_conditions(
                best_leaf
            )
            or []
        )

        # canonicalize_path_conditions expects (feature, threshold,
        # direction); path_conditions produces (feature, direction,
        # threshold) -- reorder, then let the shared utility collapse
        # redundant same-feature bounds and redundant categorical
        # negatives (same logic used by Drift Tree / Gradient Boosting).
        rendered = canonicalize_path_conditions(
            [
                (feature_name, threshold, direction)
                for feature_name, direction, threshold in raw_conditions
            ],
            onehot_cols,
        ) or []

        rule_text = (
            " AND ".join(
                rendered
            )
            if rendered
            else (
                f"Cluster_{cluster_id} "
                "(centroid)"
            )
        )

        results[
            int(cluster_id)
        ] = {
            "rule_text": rule_text,
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
        }

    return results


# ============================================================================
# K-MEANS TECHNIQUE
# ============================================================================

class KMeansSegmentationTechnique(
    BaseSegmentationTechnique
):
    """
    K-Means Segmentation Technique implementation.
    """

    @property
    def name(self) -> str:
        return "K-Means Clustering"

    # ----------------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------------

    def run(
        self,
        dev_data: Union[
            str,
            pd.DataFrame,
        ],
        mon_data: Union[
            str,
            pd.DataFrame,
        ],
        config: Optional[
            KMeansConfig
        ] = None,
    ) -> TechniqueResult:

        cfg = (
            config
            or KMeansConfig()
        )

        start_time = (
            time.perf_counter()
        )

        # ==================================================================
        # LOAD DATA
        # ==================================================================

        dev_df = (
            pd.read_csv(dev_data)
            if isinstance(
                dev_data,
                str,
            )
            else dev_data.copy()
        )

        mon_df = (
            pd.read_csv(mon_data)
            if isinstance(
                mon_data,
                str,
            )
            else mon_data.copy()
        )

        # ==================================================================
        # SCHEMA DETECTION
        # ==================================================================

        schema = detect_schema(
            dev_df,
            cfg.schema,
        )

        target_col = schema[
            "target_col"
        ]

        score_col = schema[
            "score_col"
        ]

        weight_col = schema[
            "weight_col"
        ]

        num_cols = schema[
            "numeric_cols"
        ]

        cat_cols = schema[
            "categorical_cols"
        ]

        # ==================================================================
        # EXPLICIT ROOT CAUSE FEATURES
        # ==================================================================
        #
        # CRITICAL:
        # Use the authoritative feature_cols returned by schema detection.
        #
        # Do NOT reconstruct this as:
        #
        #     num_cols + cat_cols
        #
        # because schema["feature_cols"] is the single source of truth for
        # legitimate segmentation / Root Cause Analysis features.
        #
        # This prevents ID-like columns such as customer_id, account_id and
        # application_id, as well as target, score, exposure, time/vintage
        # columns and SHAP columns, from becoming Root_Cause_Feature.
        # ==================================================================

        feature_cols = schema[
            "feature_cols"
        ]

        logger.info(
            "K-Means Root Cause features: %s",
            feature_cols,
        )

        # ==================================================================
        # DATA COUNTS
        # ==================================================================

        total_dev_n = len(
            dev_df
        )

        total_mon_n = len(
            mon_df
        )

        if total_dev_n == 0:

            raise ValueError(
                "Development dataset is empty."
            )

        if total_mon_n == 0:

            raise ValueError(
                "Monitoring dataset is empty."
            )

        # ==================================================================
        # EXPOSURE TOTALS
        # ==================================================================

        if (
            weight_col
            and weight_col
            in dev_df.columns
        ):

            total_dev_weight = float(
                dev_df[
                    weight_col
                ].sum()
            )

        else:

            total_dev_weight = 0.0

        if (
            weight_col
            and weight_col
            in mon_df.columns
        ):

            total_mon_weight = float(
                mon_df[
                    weight_col
                ].sum()
            )

        else:

            total_mon_weight = 0.0

        # ==================================================================
        # EVENT RATE
        # ==================================================================

        dev_event_rate = float(
            dev_df[
                target_col
            ].mean()
        )

        # ==================================================================
        # SUPPORT FILTERING
        # ==================================================================

        min_abs_count = (
            derive_min_abs_count(
                dev_event_rate,
                cfg.min_events_per_slice,
                cfg.min_abs_count,
            )
        )

        min_support = (
            derive_min_support(
                min_abs_count,
                total_dev_n,
                total_mon_n,
                cfg.min_support,
            )
        )

        # ==================================================================
        # SHAP DIAGNOSTICS
        # ==================================================================

        combined_df = pd.concat(
            [
                dev_df,
                mon_df,
            ],
            ignore_index=True,
        )

        shap_cols = (
            auto_detect_shap_columns(
                combined_df
            )
        )

        if shap_cols:

            try:

                shap_shift = (
                    detect_shap_shift(
                        dev_df,
                        mon_df,
                        shap_cols,
                    )
                )

                logger.info(
                    "K-Means detected SHAP shift: %s",
                    shap_shift.get(
                        "top_shift_feature"
                    ),
                )

            except Exception as exc:

                logger.warning(
                    "K-Means SHAP shift detection failed: %s",
                    exc,
                )

        # ==================================================================
        # CHECK FEATURES
        # ==================================================================

        if not num_cols and not cat_cols:

            raise ValueError(
                "K-Means could not identify any usable segmentation features."
            )

        # ==================================================================
        # FIT DEVELOPMENT-ONLY ENCODER
        # ==================================================================

        enc = fit_encoder(
            dev_df,
            num_cols,
            cat_cols,
        )

        # ==================================================================
        # TRANSFORM DATA
        # ==================================================================

        X_dev = transform(
            dev_df,
            enc,
        )

        X_mon = transform(
            mon_df,
            enc,
        )

        if (
            X_dev.shape[1]
            == 0
        ):

            raise ValueError(
                "K-Means feature matrix contains zero columns."
            )

        # ==================================================================
        # SELECT OPTIMAL K
        # ==================================================================

        best_k, k_diagnostics = (
            select_optimal_k(
                X_dev,
                cfg.k_range,
                cfg.min_cluster_pct,
                cfg.seed,
                X_mon=X_mon,
                drift_aware=cfg.drift_aware,
                drift_weight=cfg.drift_weight,
            )
        )

        logger.info(
            "K-Means selected optimal k=%s",
            best_k,
        )

        # ==================================================================
        # FIT FINAL K-MEANS MODEL
        # ==================================================================

        Estimator = _kmeans_cls(
            len(X_dev)
        )

        n_init = (
            10
            if Estimator is KMeans
            else 3
        )

        km = Estimator(
            n_clusters=best_k,
            random_state=cfg.seed,
            n_init=n_init,
        )

        dev_clusters = (
            km.fit_predict(
                X_dev
            )
        )

        mon_clusters = (
            km.predict(
                X_mon
            )
        )

        # ==================================================================
        # DISTILL HUMAN-READABLE RULES
        # ==================================================================

        rules = (
            distill_cluster_rules(
                dev_df,
                dev_clusters,
                num_cols,
                cat_cols,
                max_depth=(
                    cfg.max_tree_depth
                ),
            )
        )

        # ==================================================================
        # EVALUATE CLUSTERS
        # ==================================================================

        candidates_evaluated: list[
            dict
        ] = []

        for cluster_id in range(
            best_k
        ):

            # --------------------------------------------------------------
            # Masks
            # --------------------------------------------------------------

            d_mask = (
                dev_clusters
                == cluster_id
            )

            m_mask = (
                mon_clusters
                == cluster_id
            )

            n_dev = int(
                d_mask.sum()
            )

            n_mon = int(
                m_mask.sum()
            )

            # --------------------------------------------------------------
            # Empty cluster protection
            # --------------------------------------------------------------

            if (
                n_dev == 0
                or n_mon == 0
            ):

                logger.debug(
                    "Skipping cluster %s because it has "
                    "no Development or Monitoring observations.",
                    cluster_id,
                )

                continue

            # --------------------------------------------------------------
            # Population support
            # --------------------------------------------------------------

            pct_dev = (
                n_dev
                / total_dev_n
            )

            pct_mon = (
                n_mon
                / total_mon_n
            )

            # --------------------------------------------------------------
            # Support filter
            # --------------------------------------------------------------

            if not passes_support_filter(
                n_dev,
                n_mon,
                pct_dev,
                pct_mon,
                min_abs_count,
                min_support,
                cfg.max_segment_pct,
            ):

                logger.debug(
                    "Cluster %s failed support filter.",
                    cluster_id,
                )

                continue

            # --------------------------------------------------------------
            # Event counts
            # --------------------------------------------------------------

            x_dev = int(
                dev_df.loc[
                    d_mask,
                    target_col,
                ].sum()
            )

            x_mon = int(
                mon_df.loc[
                    m_mask,
                    target_col,
                ].sum()
            )

            # --------------------------------------------------------------
            # Event count filter
            # --------------------------------------------------------------

            if not passes_event_count_filter(
                x_dev,
                x_mon,
                cfg.min_events_per_slice,
            ):

                logger.debug(
                    "Cluster %s failed event-count filter.",
                    cluster_id,
                )

                continue

            # ==============================================================
            # CENTRALIZED CANDIDATE EVALUATION
            # ==============================================================
            #
            # IMPORTANT:
            #
            # feature_cols is explicitly passed.
            #
            # This prevents customer_id or other metadata fields from being
            # selected as Root_Cause_Feature.
            # ==============================================================

            eval_res = evaluate_segment(
                dev_df=dev_df,
                mon_df=mon_df,
                dev_mask=d_mask,
                mon_mask=m_mask,
                target_col=target_col,
                score_col=score_col,
                weight_col=weight_col,
                total_dev_n=total_dev_n,
                total_mon_n=total_mon_n,
                total_dev_weight=total_dev_weight,
                total_mon_weight=total_mon_weight,
                shap_cols=shap_cols,
                feature_cols=feature_cols,
            )

            if eval_res is None:
                continue

            # --------------------------------------------------------------
            # Rule information
            # --------------------------------------------------------------

            rule_info = rules.get(
                cluster_id,
                {
                    "rule_text": (
                        f"Cluster_{cluster_id}"
                    ),
                    "precision": 1.0,
                    "recall": 1.0,
                },
            )

            # --------------------------------------------------------------
            # Candidate record
            # --------------------------------------------------------------

            cand_record = {
                **eval_res,

                "Cluster_ID": (
                    cluster_id
                ),

                "Segment_Definition": (
                    f"Cluster_{cluster_id}: "
                    f"{rule_info['rule_text']}"
                ),

                "Rule_Precision": (
                    rule_info.get(
                        "precision",
                        1.0,
                    )
                ),

                "Rule_Recall": (
                    rule_info.get(
                        "recall",
                        1.0,
                    )
                ),

                "_mon_mask": (
                    m_mask
                ),
            }

            candidates_evaluated.append(
                cand_record
            )

        # ==================================================================
        # NO VALID CANDIDATES
        # ==================================================================

        if not candidates_evaluated:

            exec_time = (
                time.perf_counter()
                - start_time
            )

            return TechniqueResult(
                technique_name=self.name,

                overall={
                    "total_dev_n": total_dev_n,
                    "total_mon_n": total_mon_n,
                    "optimal_k": best_k,
                    "candidates_evaluated": 0,
                    "candidates_returned": 0,
                },

                segments_df=pd.DataFrame(),

                execution_time=exec_time,

                schema=schema,
            )

        # ==================================================================
        # CENTRALIZED SEVERITY SCORING
        # ==================================================================

        scored_records = (
            compute_severity_scores(
                candidates_evaluated,

                w_psi=cfg.w_psi,

                w_business_impact=(
                    cfg.w_business_impact
                ),

                w_gini_drop=(
                    cfg.w_gini_drop
                ),

                w_ks_drop=(
                    cfg.w_ks_drop
                ),

                w_br_shift=(
                    cfg.w_br_shift
                ),

                significance_alpha=(
                    cfg.significance_alpha
                ),
            )
        )

        # ==================================================================
        # SIGNIFICANCE-FIRST RANKING
        # ==================================================================

        ranked_records = (
            rank_records_significance_first(
                scored_records
            )
        )

        # ==================================================================
        # ASSIGN RANK
        # ==================================================================

        top_records = (
            ranked_records[
                : cfg.top_n
            ]
        )

        for i, record in enumerate(
            top_records,
            1,
        ):

            record[
                "Rank"
            ] = i

            # Mask is no longer required after ranking.
            record.pop(
                "_mon_mask",
                None,
            )

        # ==================================================================
        # DATAFRAME
        # ==================================================================

        segments_df = pd.DataFrame(
            top_records
        )

        # ==================================================================
        # STANDARDIZE OUTPUT NAMES
        # ==================================================================

        col_rename = {

            # --------------------------------------------------------------
            # Population
            # --------------------------------------------------------------

            "n_dev":
                "Dev_Count",

            "n_mon":
                "Mon_Count",

            "pct_dev":
                "Dev_Pct",

            "pct_mon":
                "Mon_Pct",

            # --------------------------------------------------------------
            # PSI
            # --------------------------------------------------------------

            "psi":
                "PSI",

            # --------------------------------------------------------------
            # Bad Rate
            # --------------------------------------------------------------

            "br_dev":
                "Dev_BR",

            "br_mon":
                "Mon_BR",

            "delta_br":
                "Delta_BR",

            "br_pvalue":
                "BR_PValue",

            # --------------------------------------------------------------
            # AUC
            # --------------------------------------------------------------

            "auc_dev":
                "Dev_AUC",

            "auc_mon":
                "Mon_AUC",

            "delta_auc":
                "Delta_AUC",

            # --------------------------------------------------------------
            # Gini
            # --------------------------------------------------------------

            "gini_dev":
                "Dev_Gini",

            "gini_mon":
                "Mon_Gini",

            "delta_gini":
                "Delta_Gini",

            # --------------------------------------------------------------
            # KS
            # --------------------------------------------------------------

            "ks_dev":
                "Dev_KS",

            "ks_mon":
                "Mon_KS",

            "delta_ks":
                "Delta_KS",

            "score_shift_pvalue":
                "Score_Shift_PValue",

            # --------------------------------------------------------------
            # Exposure
            # --------------------------------------------------------------

            "weight_dev":
                "Dev_EAD",

            "weight_mon":
                "Mon_EAD",

            "dev_weight_pct":
                "Dev_Exposure_Pct",

            "mon_weight_pct":
                "Mon_Exposure_Pct",

            "exposure_drift":
                "Exposure_Drift",

            # --------------------------------------------------------------
            # Calibration
            # --------------------------------------------------------------

            "calibration_drift":
                "Calibration_Drift",

            # --------------------------------------------------------------
            # Root Cause
            # --------------------------------------------------------------

            "top_drift_feature":
                "Root_Cause_Feature",

            "top_drift_psi":
                "Root_Cause_PSI",

            "root_cause_score":
                "Root_Cause_Score",

            # --------------------------------------------------------------
            # Backward compatibility
            # --------------------------------------------------------------

            "top_shap_shift_feature":
                "Top_SHAP_Feature",

            "top_shap_shift_psi":
                "Top_SHAP_PSI",
        }

        segments_df = (
            segments_df.rename(
                columns=col_rename
            )
        )

        # ==================================================================
        # PORTFOLIO VIEW
        # ==================================================================

        portfolio_df = (
            build_portfolio_view(
                segments_df,
                len(
                    segments_df
                ),
            )
        )

        # ==================================================================
        # EXECUTION TIME
        # ==================================================================

        exec_time = (
            time.perf_counter()
            - start_time
        )

        # ==================================================================
        # OVERALL RESULT
        # ==================================================================

        overall = {

            "total_dev_n":
                total_dev_n,

            "total_mon_n":
                total_mon_n,

            "optimal_k":
                best_k,

            "candidates_evaluated":
                len(
                    candidates_evaluated
                ),

            "candidates_returned":
                len(
                    segments_df
                ),
        }

        # ==================================================================
        # RETURN
        # ==================================================================

        return TechniqueResult(

            technique_name=self.name,

            overall=overall,

            segments_df=segments_df,

            execution_time=exec_time,

            schema=schema,

            portfolio_view_df=(
                portfolio_df
            ),

            extra={
                "k_diagnostics":
                    k_diagnostics,
            },
        )


# ============================================================================
# FUNCTIONAL WRAPPER
# ============================================================================

def run_kmeans_segmentation(
    dev_path: Union[
        str,
        pd.DataFrame,
    ],
    mon_path: Union[
        str,
        pd.DataFrame,
    ],
    cfg: Optional[
        KMeansConfig
    ] = None,
) -> dict:
    """
    Functional wrapper for K-Means segmentation execution.

    Uses the common parameter optimization framework.
    """

    from core.parameter_optimization import (
        optimize_parameters,
    )

    cfg = (
        cfg
        or KMeansConfig()
    )

    dev_df = (
        pd.read_csv(dev_path)
        if isinstance(
            dev_path,
            str,
        )
        else dev_path.copy()
    )

    mon_df = (
        pd.read_csv(mon_path)
        if isinstance(
            mon_path,
            str,
        )
        else mon_path.copy()
    )

    def _run_single(
        d_df,
        m_df,
        config,
    ):

        technique = (
            KMeansSegmentationTechnique()
        )

        return technique.run(
            d_df,
            m_df,
            config=config,
        )

    res = optimize_parameters(
        _run_single,
        cfg,
        dev_df,
        mon_df,
    )

    return res.to_dict()