# Adapted from: https://github.com/Pyomo/PyomoGallery/blob/master/p_median/p-median.py
import pyomo.environ as pyo
import random

random.seed(42)

# These could also be pyo.Param objects
number_of_candidates = 10 # Number of candidate locations
number_of_customers = 6   # Number of customers
number_of_facilities = 3  # Number of facilities 

## Create the model
model = pyo.ConcreteModel()

## Sets
# Set of candidate locations (could be just a pyo.Set if names are better)
model.candidate_locations = pyo.RangeSet(1,number_of_candidates)

# Set of customer nodes
model.customer_nodes = pyo.RangeSet(1,number_of_customers)

## Parameters
# demand[j] - demand of customer j
model.demand = pyo.Param(model.customer_nodes, initialize=lambda m,j : random.uniform(5.0,10.0))

# cost[i,j] - cost of satisfying a unit of demand for customer j from facility i
model.cost = pyo.Param(model.candidate_locations, model.customer_nodes,
        initialize=lambda m,i,j : random.uniform(1.0,2.0))

## (decision) Variables
# fraction_production[i,j] - fraction of demand of customer j that is supplied byfacility i
model.fraction_production = pyo.Var(model.candidate_locations, model.customer_nodes, bounds=(0.,1.))

# build_facility[i] - a binary variable that is 1 if a facility is located at location i
model.build_facility = pyo.Var(model.candidate_locations, within=pyo.Binary)

## Constraints
# Exactly p facilities are located
def facilities_rule(m):
    return sum(m.build_facility[i] for i in m.candidate_locations) == number_of_facilities
model.facilities_limit = pyo.Constraint(rule=facilities_rule)

# All the demand for customer j must be satisfied (with indexed object slice)
def demand_satisfaction_rule(m, j):
    return sum(m.fraction_production[:,j]) == 1.
model.demand_satisfaction = pyo.Constraint(model.customer_nodes, rule=demand_satisfaction_rule)

# Using the @Constraint decorator
# Creates constraint with the same name as the decorated function
@model.Constraint(model.candidate_locations, model.customer_nodes)
def open_demand_served(m, i, j):
    return m.fraction_production[i,j] <= m.build_facility[i]

## Objective 
model.total_cost = pyo.Objective(
        expr=sum(model.demand[j]*model.cost[i,j]*model.fraction_production[i,j]
                for i in model.candidate_locations for j in model.customer_nodes))
