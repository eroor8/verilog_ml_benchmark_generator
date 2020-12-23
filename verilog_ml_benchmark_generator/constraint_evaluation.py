import constraint
import sys
import math

#NUM_PES = 1

def our_constraint(x,y):
    if x + y >= 5:
        return True
    
def ExactProductConstraint(x,val0=1,val1=1,val2=1,val3=1,val4=1,val5=1,val6=1):
    return ((val0 * val1 * val2 * val3 * val4 * val5 * val6) == x)

def MaxProductConstraint(x,val0=1,val1=1,val2=1,val3=1,val4=1,val5=1,val6=1):
    return ((val0 * val1 * val2 * val3 * val4 * val5 * val6) <= x)

def MACProductConstraint(val0=1,val1=1,val2=1,val3=1,val4=1,val5=1,val6=1):
    return ((val0 * val1 * val2 * val3 * val4 * val5 * val6) <= NUM_MACS)

def MinProductConstraint(x,val0=1,val1=1,val2=1,val3=1,val4=1,val5=1,val6=1):
    return ((val0 * val1 * val2 * val3 * val4 * val5 * val6) >= x)

def ApproximateProductConstraint(x,val0=1,val1=1,val2=1,val3=1,val4=1,val5=1,val6=1):
    min_product = (val0 * val1 * val2 * val3 * val4 * val5 * val6)
    max_product = max((val0-1) * val1 * val2 * val3 * val4 * val5 * val6,
                      val0 * (val1-1) * val2 * val3 * val4 * val5 * val6,
                      val0 * val1 * (val2-1) * val3 * val4 * val5 * val6,
                      val0 * val1 * val2 * (val3-1) * val4 * val5 * val6,
                      val0 * val1 * val2 * val3 * (val4-1)  * val5 * val6,
                      val0 * val1 * val2 * val3 * val4 * (val5-1)  * val6,
                      val0 * val1 * val2 * val3 * val4 * val5 * (val6-1))
    return ((min_product >= x) and (max_product <= x))
    
def get_product(var_dict, var_keys):
    product = 1
    for var_key in var_keys:
        product *= var_dict[var_key]
    return product

def find_mappings(hwb, workload, pe_count, enable_soft_logic=False, suggested_solution=None):
    print("Solving for optimal mapping vector. This may take several minutes.")
    problem = constraint.Problem()
    loop_bounds = ['B','C','E','PX','PY','RX','RY']
    access_patterns = {'URW':['RX'], 'URN':['RY','C'], 'UE':['E'], 'UB':['B','PX','PY'], 'UG':[]}
    access_patterns_r = {'B':'UB','C':'URN','E':'UE','PX':'UB','PY':'UB','RX':'URW','RY':'URN'}
    levels = ['O','I','T']
    hwbp = hwb['possible_projections']

    # Windowing can't be done in both dimensions inside the PE or across PEs.
    #problem.addConstraint(constraint.InSetConstraint([1]), ['PXO'])
    #problem.addConstraint(constraint.InSetConstraint([1]), ['PXI'])

    # Ensure that product of tiling factors is the workload dimension
    for loop_bound in loop_bounds:
        curr_bound = workload.get(loop_bound,1)
        #problem.addVariable(loop_bound, [curr_bound])
        
        # The inner unrolling factor shouldn't exceed the max unrolling physically possible
        # It also won't be greater than the total loop bound.
        max_bound_i = hwbp[access_patterns_r[loop_bound]]
        max_bound_i = min(max_bound_i, curr_bound)
        
        # We probably want to unroll within the PE as much as possible if this
        # is the only loop bound corresponding to an access pattern.
        min_bound_i = 1
        if (len(access_patterns[access_patterns_r[loop_bound]]) == 1):
            min_bound_i = min(max(math.floor(max_bound_i/2),1), curr_bound)
            if (max_bound_i == curr_bound):
                min_bound_i = curr_bound
        for level in levels:
            if (suggested_solution):
                problem.addVariable(loop_bound+level, [suggested_solution[loop_bound+level]])
            else:
                min_bound = 1
                if (level == 'O'):
                    max_bound = min(pe_count, math.ceil(curr_bound/min_bound_i))
                elif (level == 'I'):
                    max_bound = max_bound_i
                    min_bound = min_bound_i
                else:
                    max_bound = math.ceil(curr_bound/min_bound_i)
                problem.addVariable(loop_bound + level,
                                    range(min_bound, min(curr_bound,max_bound)+1))
        nested_bounds = [loop_bound + level for level in levels]
        #problem.addConstraint(ApproximateProductConstraint, [loop_bound] + nested_bounds)
        problem.addConstraint(lambda val0=1, val1=1, val2=1, val3=1,
                              val4=1, val5=1, val6=1, maxv=curr_bound:
                              ApproximateProductConstraint(maxv,val0,val1,val2,val3,val4,val5),
                              nested_bounds)
        
    # Ensure that product of outer tiling factors is <= # PEs
    problem.addConstraint(lambda val0=1, val1=1, val2=1, val3=1,
                          val4=1, val5=1, val6=1, maxv=pe_count:
                          ((val0 * val1 * val2 * val3 * val4 * val5 * val6) <= maxv),
                          [loop_bound + 'O' for loop_bound in loop_bounds])
    
    # Ensure that the inner tiling factors are compatible with the PE access patterns
    for access_pattern in access_patterns:
        nested_bounds = [loop_bound + 'I' for loop_bound in access_patterns[access_pattern]]
        if len(nested_bounds) > 0:
            problem.addConstraint(lambda val0=1, val1=1, val2=1, val3=1,
                                  val4=1, val5=1, val6=1, maxv=hwbp[access_pattern]:
                                  ((val0 * val1 * val2 * val3 * val4 * val5 * val6) <= maxv),
                                  nested_bounds)
        
    solutions = problem.getSolutions()
    assert len(solutions) > 0
    print("Finding Min solution")
    min_product = sys.maxsize
    min_solutions = []
    for solution in solutions:
        product_cycles = get_product(solution, [loop_bound + 'T' for loop_bound in loop_bounds])
        num_used_PEs = get_product(solution, [loop_bound + 'O' for loop_bound in loop_bounds])
        
        if product_cycles < min_product:
            min_product = product_cycles 
            min_solutions = [solution]
        elif product_cycles == min_product:
            min_solutions += [solution]
    print("Best solutions (of " + str(len(solutions)) + ", with throughput " + str(min_product))
    for min_solution in min_solutions:
        print(min_solution)
    return min_solutions

if __name__ == "__main__":
    find_mappings({},{})
