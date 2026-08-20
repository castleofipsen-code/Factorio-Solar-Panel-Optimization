"""Reusable HiGHS evaluators for the staged bound-and-penalty search."""

from concurrent.futures import ThreadPoolExecutor
import math
import threading

import highspy
import numpy as np
import scipy.sparse as sp

import objectives
from staged_core.network_highs import DD
from staged_core.target_highs import (
    ACCUMULATOR_PLACEMENT,
    SOLAR_PLACEMENT,
    TARGET_ACCUMULATORS,
    TARGET_SOLAR,
    packing_parameters,
)


OPTIMAL = highspy.HighsModelStatus.kOptimal
INFEASIBLE = highspy.HighsModelStatus.kInfeasible
NETWORK_VARIABLES = np.concatenate(
    (
        np.arange(2 * DD, 3 * DD, dtype=np.int32),
        np.arange(4 * DD, 5 * DD, dtype=np.int32),
    )
)


def _finite_highs_bounds(values):
    """Replace NumPy infinities with the explicit HiGHS infinity value."""
    values = np.asarray(values, dtype=float)
    return np.where(
        np.isneginf(values),
        -highspy.kHighsInf,
        np.where(np.isposinf(values), highspy.kHighsInf, values),
    )


def build_highs_model(
    matrix,
    row_lower,
    row_upper,
    costs,
    column_lower,
    column_upper,
    integrality,
    *,
    output=False,
    time_limit=math.inf,
):
    """Pass one sparse matrix model directly to native HiGHS."""
    matrix = sp.csr_matrix(matrix)
    model_data = highspy.HighsLp()
    model_data.num_col_ = matrix.shape[1]
    model_data.num_row_ = matrix.shape[0]
    model_data.col_cost_ = np.asarray(costs, dtype=float)
    model_data.col_lower_ = _finite_highs_bounds(column_lower)
    model_data.col_upper_ = _finite_highs_bounds(column_upper)
    model_data.row_lower_ = _finite_highs_bounds(row_lower)
    model_data.row_upper_ = _finite_highs_bounds(row_upper)
    model_data.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model_data.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    model_data.a_matrix_.index_ = matrix.indices.astype(np.int32)
    model_data.a_matrix_.value_ = matrix.data.astype(float)
    model_data.integrality_ = [
        (
            highspy.HighsVarType.kInteger
            if integer
            else highspy.HighsVarType.kContinuous
        )
        for integer in np.asarray(integrality) != 0
    ]

    model = highspy.Highs()
    model.setOptionValue("output_flag", bool(output))
    model.setOptionValue("threads", 1)
    if math.isfinite(time_limit):
        model.setOptionValue("time_limit", float(time_limit))
    status = model.passModel(model_data)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected the model: {status}")
    return model


class ExactStageBLPHighs:
    """Reusable continuous Stage-B power bound for one fixed network."""

    def __init__(
        self,
        initial_network,
        time_limit=math.inf,
        periodic_electric_coverage=True,
    ):
        result = objectives.construct_matrix_coverage_fixed_network(
            50,
            initial_network[:DD],
            initial_network[DD:],
            min_power=8000,
            roboport_substitution_factor=0,
            periodic_electric_coverage=periodic_electric_coverage,
        )
        matrix, row_lower, row_upper, _, costs, lower, upper, _ = result
        lower = np.asarray(lower, dtype=float).copy()
        upper = np.asarray(upper, dtype=float).copy()
        lower[NETWORK_VARIABLES] = 0
        upper[NETWORK_VARIABLES] = 1
        self.model = build_highs_model(
            matrix,
            row_lower,
            row_upper,
            costs,
            lower,
            upper,
            np.zeros(len(costs)),
            time_limit=time_limit,
        )

    def evaluate(self, layout):
        network = layout.network_vector().astype(float)
        self.model.changeColsBounds(
            len(NETWORK_VARIABLES),
            NETWORK_VARIABLES,
            network,
            network,
        )
        self.model.run()
        status = self.model.getModelStatus()
        runtime = float(self.model.getRunTime())
        if status != OPTIMAL:
            return math.nan, None, runtime, status
        return -float(self.model.getObjectiveValue()), None, runtime, status


class ParallelStageBEvaluatorHighs:
    def __init__(
        self,
        initial_network,
        workers,
        time_limit=math.inf,
        periodic_electric_coverage=True,
    ):
        self.initial_network = initial_network
        self.time_limit = time_limit
        self.periodic_electric_coverage = periodic_electric_coverage
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _evaluate(self, layout):
        evaluator = getattr(self.local, "evaluator", None)
        if evaluator is None:
            evaluator = ExactStageBLPHighs(
                self.initial_network,
                self.time_limit,
                self.periodic_electric_coverage,
            )
            self.local.evaluator = evaluator
        return layout, *evaluator.evaluate(layout)

    def evaluate(self, layouts):
        return list(self.executor.map(self._evaluate, layouts))

    def close(self):
        self.executor.shutdown(wait=True)


PACKING_MATRIX = sp.vstack(
    (
        sp.hstack((SOLAR_PLACEMENT, ACCUMULATOR_PLACEMENT), format="csr"),
        sp.hstack(
            (sp.csr_matrix(np.ones((1, DD))), sp.csr_matrix((1, DD))),
            format="csr",
        ),
        sp.hstack(
            (sp.csr_matrix((1, DD)), sp.csr_matrix(np.ones((1, DD)))),
            format="csr",
        ),
    ),
    format="csr",
)
PACKING_COLUMNS = np.arange(2 * DD, dtype=np.int32)
PACKING_ROWS = np.arange(DD + 2, dtype=np.int32)


class ExactCoverPenaltyLPHighs:
    """Reusable fractional unsupported-cover penalty model."""

    def __init__(self, solar_penalty):
        self.solar_penalty = solar_penalty
        rhs = np.concatenate(
            (np.ones(DD), (TARGET_SOLAR, TARGET_ACCUMULATORS))
        )
        self.model = build_highs_model(
            PACKING_MATRIX,
            rhs,
            rhs,
            np.zeros(2 * DD),
            np.zeros(2 * DD),
            np.ones(2 * DD),
            np.zeros(2 * DD),
        )

    def evaluate(self, layout, oracle):
        free, solar_eligible, accumulator_eligible = packing_parameters(
            layout, oracle
        )
        costs = np.concatenate(
            (
                self.solar_penalty * (~solar_eligible).astype(float),
                (~accumulator_eligible).astype(float),
            )
        )
        rhs = np.concatenate(
            (free, (TARGET_SOLAR, TARGET_ACCUMULATORS))
        )
        self.model.changeColsCost(2 * DD, PACKING_COLUMNS, costs)
        self.model.changeRowsBounds(DD + 2, PACKING_ROWS, rhs, rhs)
        self.model.run()
        status = self.model.getModelStatus()
        runtime = float(self.model.getRunTime())
        if status != OPTIMAL:
            return math.inf, status, runtime
        return float(self.model.getObjectiveValue()), status, runtime


class ParallelPenaltyEvaluatorHighs:
    def __init__(self, oracle, workers, solar_penalty):
        self.oracle = oracle
        self.solar_penalty = solar_penalty
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _evaluate(self, layout):
        evaluator = getattr(self.local, "evaluator", None)
        if evaluator is None:
            evaluator = ExactCoverPenaltyLPHighs(self.solar_penalty)
            self.local.evaluator = evaluator
        score, status, runtime = evaluator.evaluate(layout, self.oracle)
        return layout, score, status, runtime

    def evaluate(self, layouts):
        return list(self.executor.map(self._evaluate, layouts))

    def close(self):
        self.executor.shutdown(wait=True)


class ExactTargetPackingHighs:
    """Binary exact-cover verification used only for zero-penalty layouts."""

    def __init__(self, workers):
        del workers  # All parallel LP evaluators and this model use one thread.
        rhs = np.concatenate(
            (np.ones(DD), (TARGET_SOLAR, TARGET_ACCUMULATORS))
        )
        self.model = build_highs_model(
            PACKING_MATRIX,
            rhs,
            rhs,
            np.zeros(2 * DD),
            np.zeros(2 * DD),
            np.ones(2 * DD),
            np.ones(2 * DD),
            output=True,
        )
        self.status = highspy.HighsModelStatus.kNotset

    def solve(self, layout, oracle):
        free, solar_eligible, accumulator_eligible = packing_parameters(
            layout, oracle
        )
        upper = np.concatenate(
            (solar_eligible, accumulator_eligible)
        ).astype(float)
        rhs = np.concatenate(
            (free, (TARGET_SOLAR, TARGET_ACCUMULATORS))
        )
        self.model.changeColsBounds(
            2 * DD,
            PACKING_COLUMNS,
            np.zeros(2 * DD),
            upper,
        )
        self.model.changeRowsBounds(DD + 2, PACKING_ROWS, rhs, rhs)
        self.model.run()
        self.status = self.model.getModelStatus()
        solution = self.model.getSolution()
        if self.status != OPTIMAL or not solution.value_valid:
            return None
        values = np.rint(np.asarray(solution.col_value)).astype(int)
        return values[:DD], values[DD:]

    def close(self):
        self.model.clear()
