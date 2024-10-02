import pulp
import numpy as np
import matplotlib.pyplot as plt

# Number of locations including the depot
num_locations = 6
# Number of vehicles
num_vehicles = 2
np.random.seed(0)

# Generate random coordinates for locations (including depot at index 0)
locations = np.random.rand(num_locations, 2) * 100

# Calculate the distance matrix
distance_matrix = np.linalg.norm(locations[:, np.newaxis] - locations, axis=2)

# Create the MILP problem
vrp = pulp.LpProblem("Vehicle_Routing_Problem", pulp.LpMinimize)

# Decision variables
x = pulp.LpVariable.dicts('x', ((i, j, k) for i in range(num_locations) for j in range(num_locations) for k in range(num_vehicles)), cat='Binary')

# Objective function: Minimize total travel distance
vrp += pulp.lpSum(distance_matrix[i][j] * x[i, j, k] for i in range(num_locations) for j in range(num_locations) for k in range(num_vehicles))

# Each location is visited exactly once by one vehicle
for j in range(1, num_locations):
    vrp += pulp.lpSum(x[i, j, k] for i in range(num_locations) for k in range(num_vehicles)) == 1

# Ensure vehicles leave and return to the depot
for k in range(num_vehicles):
    vrp += pulp.lpSum(x[0, j, k] for j in range(1, num_locations)) == 1
    vrp += pulp.lpSum(x[i, 0, k] for i in range(1, num_locations)) == 1

# Subtour elimination constraints
u = pulp.LpVariable.dicts('u', (i for i in range(num_locations)), lowBound=0, cat='Integer')

for k in range(num_vehicles):
    for i in range(1, num_locations):
        for j in range(1, num_locations):
            if i != j:
                vrp += u[i] - u[j] + num_locations * x[i, j, k] <= num_locations - 1

# Solve the problem
vrp.solve()

# Output the results
print(f"Status: {pulp.LpStatus[vrp.status]}")
print(f"Total Distance: {pulp.value(vrp.objective)}")

# Extract the routes
routes = {k: [] for k in range(num_vehicles)}

for k in range(num_vehicles):
    current_location = 0
    visited = set()
    while True:
        routes[k].append(current_location)
        visited.add(current_location)
        try:
            next_location = next(j for j in range(num_locations) if pulp.value(x[current_location, j, k]) == 1 and j not in visited)
        except StopIteration:
            routes[k].append(0)
            break
        if next_location == 0:
            routes[k].append(0)
            break
        current_location = next_location

# Print the routes
for k, route in routes.items():
    print(f"Route for vehicle {k}: {route}")

# Visualization
colors = ['r', 'g', 'b', 'y', 'c']
for k in range(num_vehicles):
    route = routes[k]
    locs = locations[route]
    plt.plot(locs[:, 0], locs[:, 1], colors[k % len(colors)], marker='o', linestyle='-', label=f'Vehicle {k}')
    plt.text(locs[0, 0], locs[0, 1], 'Depot', fontsize=12, color='black')
plt.scatter(locations[:, 0], locations[:, 1], c='black')
plt.title('Optimized Vehicle Routes')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()
