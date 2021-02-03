import constraint
import sys
import os
import math
import copy
il = 1

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils


def get_factors(maxv, product):
    """ Make a list of factors of product, less than maxv.

    :param maxv: Maximum factor size
    :param product: Product to factor
    """
    max_bound = min(maxv, math.ceil(math.sqrt(product)))
    var_range = []
    for i in range(1, max_bound + 1):
        if i not in var_range:
            var_range += [i]
        if (math.ceil(product / i) < maxv) and \
           (math.ceil(product / i) not in var_range):
            var_range += [math.ceil(product / i)]
    return var_range


def score_solution(solution, num_MACs, loop_bounds, preload_i, preload_o):
    """ Score a possible solution based on estimated cycle count.

    :param solution: Mapping vector
    :param num_MACs: Total number of MACs per PE
    :param loop_bounds: Layer loop bounds
    :param preload_i: number of weight chains per PE
    :param preload_o: number of PE weight chains
    """
    # calculate an approximate cycle count
    product_cycles = get_product(
        solution, [loop_bound + 'T' for loop_bound in loop_bounds])
    num_used_PEs = get_product(
        solution,
        [loop_bound + 'O' for loop_bound in loop_bounds])
    num_used_MACs = get_product(
        solution,
        [loop_bound + 'I' for loop_bound in loop_bounds])
    preload_o = min(preload_o, num_used_PEs)
    if (preload_o < 0):
        preload_o = num_used_PEs
    if (preload_i < 0):
        preload_i = num_used_MACs
    preload_chain_len = math.ceil(num_MACs / preload_i) * \
        math.ceil(num_used_PEs/preload_o)
    if (solution['CT']*solution['RYT']*solution['RXT'] > 1):
        # In this case (output stationary) we
        # need to reload the weights after every addition.
        preload_cycles = preload_chain_len * \
            get_product(solution, ['ET', 'RXT', 'RYT', 'CT', 'BT', 'PXT',
                                   'PYT'])
    else:
        preload_cycles = preload_chain_len * \
            get_product(solution, ['ET', 'RXT', 'RYT', 'CT'])
    pipeline_count = get_product(solution, ['RXO', 'RYO', 'CO'])

    total_cycles = product_cycles + preload_cycles + pipeline_count
    return total_cycles


def ApproximateProductConstraint(x, val0=1, val1=1, val2=1, val3=1, val4=1,
                                 val5=1, val6=1):
    """ Make sure that product of given factors is greater than x,
        but not unneccessarily big.

    :param x: Expected product
    :param val0-val6: Factors
    """
    min_product = (val0 * val1 * val2 * val3 * val4 * val5 * val6)
    max_product = max((val0 - 1) * val1 * val2 * val3 * val4 * val5 * val6,
                      val0 * (val1 - 1) * val2 * val3 * val4 * val5 * val6,
                      val0 * val1 * (val2 - 1) * val3 * val4 * val5 * val6,
                      val0 * val1 * val2 * (val3 - 1) * val4 * val5 * val6,
                      val0 * val1 * val2 * val3 * (val4 - 1) * val5 * val6,
                      val0 * val1 * val2 * val3 * val4 * (val5 - 1) * val6,
                      val0 * val1 * val2 * val3 * val4 * val5 * (val6 - 1))
    return ((min_product >= x) and (max_product <= x))


def get_product(var_dict, var_keys):
    """ Find the product of values in the dict with keys in ver_keys

    :param var_dict: Dict of factors
    :param var_keys: List of keys
    """
    product = 1
    for var_key in var_keys:
        product *= var_dict[var_key]
    return product


def find_mappings(hwb, workload, pe_count, enable_soft_logic=True,
                  suggested_solution=None, preload_o=1, preload_i=1,
                  num_solutions=1, cost_function=score_solution,
                  buffer_count=-1):
    """ Find the best set of mappings (according to a given cost function)
    that are achievable given the ML blocks available.

    :param hwb: Tensor block definition
    :param pe_count: Number of PEs available
    :param enable_soft_logic: Add soft-logic to PEs to enable more mappings.
    :param suggested_solution: Proposed mapping vector
    :param preload_o: Information about weight preload method
    :param preload_i: Information about weight preload method
    :param num_solutions: number of solutions to return
    :param cost_function: function used to sort possible solutions
    """
    utils.printi(il, "Solving for optimal mapping vector. " +
                 "This may take several minutes.")
    utils.printi(il, "Workload definition: " + str(workload))
    problem = constraint.Problem()
    hwbp = copy.deepcopy(hwb['access_patterns'])
    ws = not (hwb.get('output_accumulator', False))
    loop_bounds = ['B', 'C', 'E', 'PX', 'PY', 'RX', 'RY']
    levels = ['O', 'I', 'T']

    # Mappings between access patterns and corresponding dimensions.
    access_patterns = {'AP1': ['RX'], 'AP2': ['RY', 'C'], 'AP3': ['E'],
                       'AP4': ['B', 'PX', 'PY'], 'AP5': []}
    access_patterns_r = {'B': ['AP4'], 'C': ['AP2'], 'E': ['AP3'],
                         'PX': ['AP4'], 'PY': ['AP4'], 'RX': ['AP1'],
                         'RY': ['AP2']}

    if enable_soft_logic:
        if (hwbp['AP1'] < 2):
            access_patterns['AP2'] += access_patterns['AP1']
            for lb in access_patterns['AP1']:
                access_patterns_r[lb] += ['AP2']
                access_patterns_r[lb].remove('AP1')
            access_patterns['AP1'] = []
        # if (hwbp['AP2'] == 1):
        #     access_patterns['AP5'] += access_patterns['AP2']
        #     for lb in access_patterns['AP2']:
        #         access_patterns_r[lb] += ['AP5']
        #         access_patterns_r[lb].remove('AP2')
        #     access_patterns['AP2'] = []
        if (hwbp['AP3'] == 1):
            access_patterns['AP5'] += access_patterns['AP3']
            for lb in access_patterns['AP3']:
                access_patterns_r[lb] += ['AP5']
                access_patterns_r[lb].remove('AP3')
            access_patterns['AP3'] = []
        if (hwbp['AP4'] == 1):
            access_patterns['AP5'] += access_patterns['AP4']
            for lb in access_patterns['AP4']:
                access_patterns_r[lb] += ['AP5']
                access_patterns_r[lb].remove('AP4')
            access_patterns['AP4'] = []
    if (hwbp['AP1'] == 0):
        hwbp['AP1'] = 1

    # Windowing can't be done in both dimensions inside the PE or across PEs.
    # problem.addConstraint(constraint.InSetConstraint([1]), ['PXI'])

    # Ensure that product of tiling factors is the workload dimension
    for loop_bound in loop_bounds:
        curr_bound = workload.get(loop_bound, 1)

        # The inner unrolling factor shouldn't exceed the max unrolling
        # physically possible. It also won't be greater than the total
        # loop bound.
        ap_product = 1
        for ap in access_patterns_r[loop_bound]:
            ap_product *= hwbp[ap]
        max_bound_i = min(curr_bound, ap_product)

        # We probably want to unroll within the PE as much as possible
        # if this is the only loop bound corresponding to an access pattern.
        min_bound_i = 1
        if (len(access_patterns_r[loop_bound]) == 1) and \
           (len(access_patterns[access_patterns_r[loop_bound][0]]) == 1):
            min_bound_i = min(max(math.floor(max_bound_i / 2), 1), curr_bound)
            if (max_bound_i == curr_bound):
                min_bound_i = curr_bound

        for level in levels:
            if (suggested_solution):
                problem.addVariable(loop_bound+level,
                                    [suggested_solution[loop_bound + level]])
            else:
                if (level == 'O'):
                    var_range = get_factors(pe_count,
                                            math.ceil(curr_bound /
                                                      min_bound_i))
                elif (level == 'I'):
                    var_range = range(min_bound_i,
                                      min(curr_bound, max_bound_i) + 1)
                else:
                    var_range = get_factors(sys.maxsize,
                                            math.ceil(curr_bound /
                                                      min_bound_i))
                problem.addVariable(loop_bound + level, var_range)

        nested_bounds = [loop_bound + level for level in levels]
        problem.addConstraint(lambda val0=1, val1=1, val2=1, val3=1,
                              val4=1, val5=1, val6=1, maxv=curr_bound:
                              ApproximateProductConstraint(maxv, val0, val1,
                                                           val2, val3, val4,
                                                           val5),
                              nested_bounds)

    # Ensure that product of outer tiling factors is <= # PEs
    problem.addConstraint(
        lambda val0=1, val1=1, val2=1, val3=1, val4=1, val5=1, val6=1,
        maxv=pe_count:
        ((val0 * val1 * val2 * val3 * val4 * val5 * val6) <= maxv),
        [loop_bound + 'O' for loop_bound in loop_bounds])

    # Ensure that the inner tiling factors are compatible with the PE access
    # patterns
    for access_pattern in access_patterns:
        nested_bounds = [loop_bound + 'I' for loop_bound in
                         access_patterns[access_pattern]]
        if len(nested_bounds) > 0:
            problem.addConstraint(
                lambda val0=1, val1=1, val2=1, val3=1, val4=1, val5=1, val6=1,
                maxv=hwbp[access_pattern]:
                ((val0 * val1 * val2 * val3 * val4 * val5 * val6) <= maxv),
                nested_bounds)

    problem.addConstraint(constraint.InSetConstraint([1]), ['RXT'])
    problem.addConstraint(constraint.InSetConstraint([1]), ['RYT'])

    # Only allow this if there is a single batch.
    if ws:
        problem.addConstraint(constraint.InSetConstraint([1]), ['CT'])

    solutions = problem.getSolutions()
    assert len(solutions) > 0

    # Sort by estimated cycle count
    min_product = sys.maxsize
    num_MACs = hwb["MAC_info"]["num_units"]
    sorted_solutions = sorted(solutions,
                              key=lambda soln, nm=num_MACs, lbs=loop_bounds,
                              pli=preload_i, plo=preload_o:
                              cost_function(soln, nm, lbs, pli, plo))
    min_solutions = sorted_solutions[0:num_solutions]
    min_product = cost_function(min_solutions[0], num_MACs, loop_bounds,
                                preload_i, preload_o)
    utils.printi(il, "Found best " + str(num_solutions) +
                 " solutions out of " + str(len(solutions)) +
                 " possibilities, with estimated cycle count " +
                 str(min_product))
    for min_solution in min_solutions:
        utils.printi(il, "Solution: " + utils.print_mapping(min_solution,
                                                            il + 1))
    return min_solutions, min_product
