import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Q(R) = -(|R_l|/|R|) H(R_l) - (|R_r|/|R|) H(R_r),
    H(R) = 1 - p_0^2 - p_1^2
    """

    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    diff_idx = np.nonzero(x_sorted[1:] != x_sorted[:-1])[0]
    if diff_idx.size == 0:
        return np.array([]), np.array([]), None, None

    thresholds = 0.5 * (x_sorted[diff_idx] + x_sorted[diff_idx + 1])

    n = len(x_sorted)
    n_left = diff_idx + 1
    n_right = n - n_left

    total_ones = y_sorted.sum()
    ones_left = np.cumsum(y_sorted)[diff_idx]
    ones_right = total_ones - ones_left

    p1_left = ones_left / n_left
    p0_left = 1.0 - p1_left

    p1_right = ones_right / n_right
    p0_right = 1.0 - p1_right

    H_left = 1.0 - p0_left**2 - p1_left**2
    H_right = 1.0 - p0_right**2 - p1_right**2

    ginis = -(n_left / n) * H_left - (n_right / n) * H_right

    best_idx = np.argmax(ginis)
    best_threshold = thresholds[best_idx]
    best_gini = ginis[best_idx]

    return thresholds, ginis, best_threshold, best_gini


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in ("real", "categorical") for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def _make_terminal(self, node, sub_y):
        node["type"] = "terminal"
        node["class"] = Counter(sub_y).most_common(1)[0][0]

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if self._max_depth is not None and depth >= self._max_depth:
            self._make_terminal(node, sub_y)
            return
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            self._make_terminal(node, sub_y)
            return
        if (
            self._min_samples_leaf is not None
            and len(sub_y) < 2 * self._min_samples_leaf
        ):
            self._make_terminal(node, sub_y)
            return
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        best_feature = None
        best_threshold = None
        best_gini = None
        best_split_mask = None

        n_features = sub_X.shape[1]

        for feat_idx in range(n_features):
            f_type = self._feature_types[feat_idx]
            cat_to_num = None

            if f_type == "real":
                f_values = sub_X[:, feat_idx]
            elif f_type == "categorical":
                counts = Counter(sub_X[:, feat_idx])
                clicks = Counter(sub_X[sub_y == 1, feat_idx])
                ratios = {}
                for key, cnt in counts.items():
                    pos_cnt = clicks.get(key, 0)
                    ratios[key] = 0.0 if pos_cnt == 0 else cnt / pos_cnt
                ordered_cats = [
                    k for k, _ in sorted(ratios.items(), key=lambda kv: kv[1])
                ]
                cat_to_num = {cat: i for i, cat in enumerate(ordered_cats)}
                f_values = np.array([cat_to_num[val] for val in sub_X[:, feat_idx]])
            else:
                raise ValueError

            _, _, thr, gini = find_best_split(f_values, sub_y)
            if thr is None:
                continue

            if best_gini is None or gini > best_gini:
                best_gini = gini
                best_feature = feat_idx
                mask = f_values < thr
                best_split_mask = mask

                if f_type == "real":
                    best_threshold = thr
                    best_categories = None
                else:
                    best_categories = [
                        cat for cat, idx in cat_to_num.items() if idx < thr
                    ]
                    best_threshold = best_categories

        if best_feature is None:
            self._make_terminal(node, sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_feature

        if self._feature_types[best_feature] == "real":
            node["threshold"] = best_threshold
        else:
            node["categories_split"] = best_threshold

        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(
            sub_X[best_split_mask],
            sub_y[best_split_mask],
            node["left_child"],
            depth=depth + 1,
        )
        self._fit_node(
            sub_X[~best_split_mask],
            sub_y[~best_split_mask],
            node["right_child"],
            depth=depth + 1,
        )

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feat = node["feature_split"]
        if self._feature_types[feat] == "real":
            thr = node["threshold"]
            if x[feat] < thr:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            cats = node["categories_split"]
            if x[feat] in cats:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
