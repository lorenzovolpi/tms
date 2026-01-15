import collections
import itertools as IT
import os
from abc import ABC
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

import env
from config import (
    get_acc_names,
    get_classifier_class_names,
    get_classifier_names,
    get_method_names,
)


class Results(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def load(
        cls,
        base_dir=env.root_dir,
        acc_name="*",
        dataset="*",
        filter_methods: list[str] | None = None,
        set_problem=True,
    ) -> "Results":
        problem = env.PROBLEM if set_problem else "*"
        dfs = []
        for path in glob(
            os.path.join(base_dir, problem, acc_name, dataset, "**", "*.json"),
            recursive=True,
        ):
            if filter_methods is None or Path(path).parent.name in filter_methods:
                dfs.append(pd.read_json(path))

        return Results(pd.concat(dfs, axis=0))

    @classmethod
    def concat(cls, res: list["Results"], axis=0) -> "Results":
        return Results(pd.concat([_r.df for _r in res], axis=axis))

    def filter_by_shift(self, filter: float = 0.5, high=True):
        methods = self.df["method"].unique()
        datasets = self.df["dataset"].unique()
        classifiers = self.df["classifier"].unique()
        acc_names = self.df["acc_name"].unique()

        new_res = []
        for m, d, cls_name, acc in IT.product(methods, datasets, classifiers, acc_names):
            _df = self.df.loc[
                (self.df["method"] == m)
                & (self.df["dataset"] == d)
                & (self.df["classifier"] == cls_name)
                & (self.df["acc_name"] == acc),
                :,
            ]
            _df["filter_index"] = np.arange(len(_df))

            _shifts = _df.loc[:, "shifts"].to_numpy()
            _filter_idx = np.nonzero(_shifts >= filter if high else _shifts <= filter)[0]
            new_res.append(_df.loc[_df["filter_index"].isin(_filter_idx), :].drop(columns=["filter_index"]))

        return Results(pd.concat(new_res))

    def split_by_shift(self, prevs: float | list[float] = 0.5, high_to_low=True, return_last=False) -> "Results":
        """
        Splits a pandas.DataFrame of results based on the shift values. prevs is used to determine by what portions
        to split the pandas.DataFrame.

        :param res:
        :param prevs:
        :param high_to_low:
        :param return_last:
        :raises ValueError: if the sum of the values in prevs exceeds 1
        :return:
        """
        if isinstance(prevs, float) and np.isclose(prevs, 1):
            return self

        if isinstance(prevs, float):
            prevs = [prevs]

        if np.isclose(np.sum(prevs), 1):
            prevs = prevs[:-1]
        elif np.sum(prevs) > 1:
            raise ValueError("Invalid prevs array: it should sum up to 1 or leave the last prevalence implicit.")

        new_ress = [[] for _ in prevs]
        if return_last:
            new_ress += []

        methods = self.df["method"].unique()
        datasets = self.df["dataset"].unique()
        classifiers = self.df["classifier"].unique()
        acc_names = self.df["acc_name"].unique()
        for m, d, cls_name, acc in IT.product(methods, datasets, classifiers, acc_names):
            _df = self.df.loc[
                (self.df["method"] == m)
                & (self.df["dataset"] == d)
                & (self.df["classifier"] == cls_name)
                & (self.df["acc_name"] == acc),
                :,
            ]
            _df["filter_index"] = np.arange(len(_df))

            _shifts = _df.loc[:, "shifts"].to_numpy()
            _sh_sort_idx = np.argsort(_shifts)[::-1] if high_to_low else np.argsort(_shifts)
            _cum_prevs = np.cumsum(prevs)
            _ths = [int(_cp * len(_df)) for _cp in _cum_prevs]
            _filter_idxs = [_sh_sort_idx[: _ths[0]]] + [_sh_sort_idx[_thl:_thr] for _thl, _thr in zip(_ths, _ths[1:])]
            if return_last:
                _filter_idxs += [_sh_sort_idx[_ths[-1] :]]
            for new_r, _fidx in zip(new_ress, _filter_idxs):
                new_r.append(_df.loc[_df["filter_index"].isin(_fidx), :].drop(columns=["filter_index"]))

        return (
            [Results(pd.concat(new_r)) for new_r in new_ress] if len(new_ress) > 1 else Results(pd.concat(new_ress[0]))
        )

    def method_ms(self, method: str, classifier_class=None, rank_label="ranking_vals") -> "Results":
        # methods = get_CAP_method_names()
        accs = get_acc_names()

        dfs = []
        # apply model selection for each method and each acc measure
        for acc_name in accs:
            # filter by method and acc and make a copy
            if classifier_class is None:
                mdf = self.df.loc[(self.df["method"] == method) & (self.df["acc_name"] == acc_name), :].copy()
            else:
                mdf = self.df.loc[
                    (self.df["method"] == method)
                    & (self.df["acc_name"] == acc_name)
                    & (self.df["classifier_class"] == classifier_class),
                    :,
                ].copy()

            # filter out classifiers not intended for model selection
            mdf = mdf.loc[~mdf["ms_ignore"], :]
            # index data by dataset, sample_id (uids), and classifier
            mdf = mdf.set_index(["dataset", "uids", "classifier"])
            # group data by sample_id and dataset and take the index of the maximum in the self.estim_acc_label column
            best_idx = mdf.groupby(["uids", "dataset"])[rank_label].idxmax()
            # use the index to filter the data, resetting the index
            mdf = mdf.loc[best_idx, :].reset_index(drop=False)
            if classifier_class is not None:
                mdf["method"] = [f"{method}-{classifier_class}"] * len(mdf)
            dfs.append(mdf)

        return Results(pd.concat(dfs, axis=0))

    def oracle_ms(self) -> "Results":
        accs = get_acc_names()

        dfs = []
        for acc in accs:
            odf = self.df.loc[self.df["acc_name"] == acc, :].groupby(["dataset", "uids", "classifier"]).first()
            best_idx = odf.groupby(["uids", "dataset"])["true_accs"].idxmax()
            odf = odf.loc[best_idx, :].reset_index(drop=False)
            odf["method"] = ["oracle"] * len(odf)
            dfs.append(odf)

        return Results(pd.concat(dfs, axis=0))

    def default_classifier_ms(self) -> "Results":
        accs = get_acc_names()
        classifiers = get_classifier_names()

        dfs = []
        for acc, classifier in IT.product(accs, classifiers):
            ndf = (
                self.df.loc[
                    (self.df["acc_name"] == acc) & (self.df["classifier"] == classifier),
                    :,
                ]
                .groupby(["dataset", "uids"])
                .first()
                .reset_index(drop=False)
            )
            if not np.all(ndf["default_c"].to_numpy()):
                continue
            ndf["method"] = ndf["classifier"]
            dfs.append(ndf)

        return Results(pd.concat(dfs, axis=0))

    def model_selection(self, oracle=False, rank_label="ranking_vals") -> "Results":
        classifier_classes = get_classifier_class_names()
        spread_methods = ["IMS"]
        methods = get_method_names()

        dfs = (
            [self.default_classifier_ms()]
            + [
                self.method_ms(method=m, classifier_class=cls_class, rank_label=rank_label)
                for m, cls_class in IT.product(spread_methods, classifier_classes)
            ]
            + [self.method_ms(method=m, rank_label=rank_label) for m in methods]
        )

        if oracle:
            dfs.append(self.oracle_ms())

        return Results.concat(dfs, axis=0)

    def filter_column_values(
        self,
        col: str,
        type: Literal["eq", "ne", "gt", "lt", "ge", "le", "isin"],
        val: Any,
    ) -> "Results":
        """
        Filters the Results object based on the specified column and value.

        :param col: the column of the Results to filter by.
        :param type: the type of the filter to apply. Can be one of: "eq", "ne", "gt", "lt", "ge", "le", "isin"
        :param val: the value to filter by. If type is "isin", this should be a list of values.
        :return: the filtered Results object.
        """
        match type:
            case "eq":
                return Results(self.df.loc[self.df[col] == val, :])
            case "ne":
                return Results(self.df.loc[self.df[col] != val, :])
            case "gt":
                return Results(self.df.loc[self.df[col] > val, :])
            case "lt":
                return Results(self.df.loc[self.df[col] < val, :])
            case "ge":
                return Results(self.df.loc[self.df[col] >= val, :])
            case "le":
                return Results(self.df.loc[self.df[col] <= val, :])
            case "isin":
                if not isinstance(val, list):
                    raise ValueError("Value for 'isin' filter must be a list.")
                return Results(self.df.loc[self.df[col].isin(val), :])
            case _:
                raise ValueError(f"Invalid type '{type}' for filtering results.")

    def select_columns(self, cols: list[str]) -> "Results":
        """
        Selects only the specified columns from the DataFrame.

        :param cols: List of column names to select.
        :return: A new Results object with only the selected columns.
        """
        return Results(self.df.loc[:, cols])

    def add_column(self, col_name: str, values: Any | list[Any] | np.ndarray) -> "Results":
        """
        Adds a new column to the Results.

        :param col_name: Name of the new column.
        :param values: List of values for the new column.
        :return: A new Results object with the added column.
        """
        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = [values] * len(self.df)

        new_df = self.df.copy()
        new_df[col_name] = values
        return Results(new_df)

    def map_column_values(self, col: str, mapping: dict[str, Any]) -> "Results":
        """
        Maps the values in a specified column using a provided mapping dictionary.

        :param col: The column to map.
        :param mapping: A dictionary where keys are current values and values are the new values.
        :return: A new Results object with the mapped column.
        """
        new_df = self.df.copy()
        # maps the values in col with the provided mapping;
        # the values not present in the mapping are kept as is
        new_df[col] = new_df[col].map(mapping).fillna(new_df[col])
        return Results(new_df)

    def apply_to_column(self, col: str, func: callable) -> "Results":
        """
        Applies a function to a specified column in the DataFrame.

        :param col: The column to apply the function to.
        :param func: A function to apply to the column values.
        :return: A new Results object with the modified column.
        """
        new_df = self.df.copy()
        new_df[col] = new_df[col].apply(func)
        return Results(new_df)

    def unique_column_values(self, col: str) -> Iterable[Any]:
        """
        Returns the unique values of a specified column.

        :param col: The column to get values from.
        :return: An iterable of unique values in the specified column.
        """
        return self.df[col].unique()
