import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import random


def find_Hill_Climbing(div_function, start, end):
    random_number = random.uniform(start, end)
    cur_x = random_number
    rate = 0.001  # Learning rate
    precision = 0.000001  # This tells us when to stop the algorithm
    previous_step_size = 1  #

    while previous_step_size > precision:
        prev_x = cur_x
        diff = div_function(cur_x)  # diversion of main function
        cur_x = cur_x - rate * diff  # find new position x
        previous_step_size = abs(cur_x - prev_x)

    return cur_x


def find_Iterative_Hill_Climbing(div_function, main_function, start, end):
    main_x = find_Hill_Climbing(div_function, start, end)
    main_y = main_function(main_x)

    for i in range(1, 100):  # use hill climbing in 100 number of time to find best position in global

        # find 2 random number expect our before number we choice

        temp_x = find_Hill_Climbing(div_function, start, main_x)
        temp_y = main_function(temp_x)

        temp_x2 = find_Hill_Climbing(div_function, main_x, end)
        temp_y2 = main_function(temp_x)

        if temp_y > temp_y2:
            temp_y = temp_y2
            temp_x = temp_x2

        if temp_y < main_y:
            main_y = temp_y
            main_x = temp_x

    return main_x


def get_neighbors(current_state, start, end, distance): #find a naighbors between start and end with distacne we give
    upper_neighbor = current_state + distance

    if upper_neighbor > end:
        upper_neighbor = end

    low_neighbor = current_state - distance

    if start > low_neighbor:
        low_neighbor = start
    neighbor = random.uniform(low_neighbor, upper_neighbor)

    return neighbor


def find_Simulated_Annealing(main_function, start, end):
    current_state = random.uniform(start, end)  # random state we are here at first time
    current_temp = 100  # our max temperature
    final_temp = 0.1  # when we arrive to this degree we break our for
    alpha = 0.95  # define how much our temperature most be change after each process

    solution = current_state

    while current_temp > final_temp:

        for i in range(250):
            neighbor = get_neighbors(current_state, start, end, 0.1)

            cost_diff = main_function(solution) - main_function(neighbor)

            if cost_diff > 0:

                solution = neighbor

            else:

                if random.uniform(0, 1) <= np.exp((cost_diff / current_temp) * -1):
                    current_state = neighbor

        current_temp *= alpha
        # print(solution)
        # print(main_function(solution))
        # print("___________________________")
    return solution


def main():


    x = Symbol('x')
    f = (sin(10 * np.pi * x) / (2 * x)) + ((x - 1) ** 4) # our function
    derivative_func = f.diff(x)  # diff our function
    derivative_func = lambdify(x, derivative_func)  # our diff function can give number and give result
    main_function = lambdify(x, f) # our function can give number and give result

    x = find_Hill_Climbing(derivative_func, 0.5, 2.5)  # our result that find in hill clilmbing
    y = find_Iterative_Hill_Climbing(derivative_func, main_function, 0.5, 2.6) # our result that find in Iter hill climbing
    z = find_Simulated_Annealing(main_function, 0.5, 2.6) # find result with Simulated annealing
    show_plot(x, y, z)


def show_plot(x1, y1, z):  # show polt 
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

    x = np.linspace(0.5, 2.5, 200)
    y = (np.sin(10 * np.pi * x) / (2 * x)) + np.power(x - 1, 4)
    plt.plot(x, y, 'k')

    plt.title('Hill climbing ', fontdict=font)
    plt.xlabel('x', fontdict=font)
    plt.ylabel('f(x)', fontdict=font)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)

    hill_climb = x1
    Iterative_Hill_Climbing = y1
    Simulated_Annealing = z

    plt.plot([hill_climb], [(np.sin(10 * np.pi * hill_climb) / (2 * hill_climb)) + np.power(hill_climb - 1, 4)],
             'ro', color='green')

    plt.plot([Iterative_Hill_Climbing], [
        (np.sin(10 * np.pi * Iterative_Hill_Climbing) / (2 * Iterative_Hill_Climbing)) + np.power(
            Iterative_Hill_Climbing - 1, 4)], 'bo', color='orange', markersize=12)

    plt.plot([Simulated_Annealing], [
        (np.sin(10 * np.pi * Simulated_Annealing) / (2 * Simulated_Annealing)) + np.power(
            Simulated_Annealing - 1, 4)], 'ro')

    plt.legend(('f(x)', 'Hill Climbing result', 'Iterative Hill Climbing result ', 'Simulated Annealing result'),
               loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
