def generate_tuples(fuel_min, fuel_max, fuel_step, position_min, position_max, position_step, velocity_min,
                    velocity_max, velocity_step, list_of_tuples):
    for k in range(5):
        for i in range(fuel_min, fuel_max + 1, fuel_step):
            for j in range(position_min, position_max + 1, position_step):
                for z in range(velocity_min, velocity_max + 1, velocity_step):
                    list_of_tuples.append((i, j, z))


fuel_min = 200
fuel_max = 1300
fuel_step = 400
position_min = 200
position_max = 2000
position_step = 400
velocity_min = -50
velocity_max = 50
velocity_step = 10
list_of_tuples = []
generate_tuples(fuel_min, fuel_max, fuel_step, position_min, position_max, position_step, velocity_min,
                velocity_max, velocity_step, list_of_tuples)

