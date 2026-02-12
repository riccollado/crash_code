# Partitioning Algorithm Documentation

## Data Structures

Each subproblem is represented as a dictionary with the following structure:

```sh
subproblem = {
    "variables": [x1, x2, ..., xn],          # Decision variables
    "constraints": [x1=0, x2=1, ...],       # Constraint expressions
    "gurobi_model": None,                   # Reference to the optimization model
    "lb": None,                             # Lower bound for this subproblem
    "ub": None,                             # Upper bound for this subproblem
    "is_singleton": False,                  # Whether the subproblem has a single solution
    "is_record_set": False                  # Whether the subproblem contains the best solution found
}
```

---

## Global Variables

```sh
Global_Obj_Fun_Value    # Best objective function value found
Global_lower_bound      # Global lower bound across all subproblems
PartitionList = []      # List of subproblems
```

---

## Algorithm Steps

### Step 1: Initialization

```sh
1. Create an initial subproblem:
   - Variables: Binary decision variables [x1, x2, ..., xn]
   - Constraints: None (empty list)
   - is_singleton: False
   - is_record_set: True

2. Add the initial subproblem to PartitionList.

3. Update global bounds and the objective function value.
```

---

### Step 2: Partitioning

```sh
K = 1
While (stopping_criteria is not met):
    For each subproblem in PartitionList:
        If (subproblem.is_record_set == True AND subproblem.is_singleton == False):
            constraints_list = subproblem.constraints
            index_of_last_constraint = length(constraints_list)

            If (index_of_last_constraint < n - 1):
                # Create two new subproblems by branching
                Subproblem1 = Copy of subproblem
                Add constraint "x[index_of_last_constraint + 1] = 0" to Subproblem1

                Subproblem2 = Copy of subproblem
                Add constraint "x[index_of_last_constraint + 1] = 1" to Subproblem2

                Remove the original subproblem from PartitionList
                Add Subproblem1 and Subproblem2 to PartitionList
            Else:
                Set subproblem.is_singleton = True

    # Call the optimizer for each subproblem
    For each subproblem in PartitionList:
        subproblem.gurobi_model = CreateOptimizationProblem(subproblem)
        EstimateBounds(subproblem)

    # Update the record set
    Select the new record set based on updated bounds.

    K = K + 1
```

---

### Step 4: Final Bound Estimation

```sh
1. Initialize a list to store the minimum function value for each scenario:
   Min_function_value_per_scenario = []

2. For each scenario (w) in scenarios:
    - Initialize a list to store function values for each partition:
      function_values_per_partition = []

    - For each partition (subproblem) in PartitionList:
        - Estimate the function value F(x, w) by calling the optimizer.
        - Append F(x, w) to function_values_per_partition.

    - Find the minimum function value for the current scenario:
      min_value = min(function_values_per_partition)
    - Append min_value to Min_function_value_per_scenario.

3. Compute the optimal value as the average of the minimum function values:
   Optimal_value = sum(Min_function_value_per_scenario) / length(Min_function_value_per_scenario)
```

---

## Function Definitions

### CreateOptimizationProblem

```sh
Input:
    - Project network
    - List of activity durations
    - Crashing activity durations
    - Crashing activity costs
    - Subproblem

Output:
    - Gurobi model

Steps:
1. Extract variables from the subproblem (e.g., x1, x2, ..., xn).
2. Define start times for each activity (e.g., s1, s2, ..., sn).
3. Add constraints:
    - Set s1 = 0 (start time of the first activity).
    - For each activity in the project network:
        - Get successors of the activity.
        - Add precedence constraints:
          s(successor) >= s(activity) + (d(activity) - crash_value * crash_time)
    - Add constraints from the subproblem.
4. Define the objective function:
    Minimize (sn + sum(crashing_costs * crashing_durations)).
5. Return the Gurobi model.
```

---

### EstimateBounds

```sh
Input:
    - Subproblem
    - ScenarioList

Output:
    - Lower bound for the objective function value
    - Solution

Steps:
1. Initialize a list to store objective function values for each scenario:
   OBJ_Function_value_per_scenario = []

2. For each scenario (w) in ScenarioList:
    - Create an optimization problem for the subproblem:
      subproblem.gurobi_model = CreateOptimizationProblem(subproblem)
    - Solve the optimization problem using Gurobi:
      (Min_ObjFun_value, solution) = Solve the problem
    - Append Min_ObjFun_value to OBJ_Function_value_per_scenario.

3. Compute the average objective function value across scenarios:
   Min_Obj_Function_value = sum(OBJ_Function_value_per_scenario) / length(ScenarioList)

4. Return the bounds and the solution.
```

---
