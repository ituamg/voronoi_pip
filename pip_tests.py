# Source code to reproduce the results given in the paper:
# "A Novel Point Inclusion Test for Convex Polygons Based on Voronoi Tessellations"

# BSD 3-Clause License
#
# Copyright (c) 2020, Rahman Salim Zengin, Volkan Sezer
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import matplotlib.pyplot as plt
import time
from math import sin, cos, pi
import timeit
from numpy.random import default_rng
from tabulate import tabulate
import pickle

# seed the random generator
rg = default_rng(12345)

# Timeit constants
REPEAT = 1
NUMBER = 10

# Time scale of the results
TIME_SCALE = 1e-9 # 1ns

# Test polygon generation constants
center_of_poly = (3.0, 3.0)
radius_of_poly = 10.0
rotation_of_poly = pi/6

# Number of edges for testing
polygon_test_sizes = [5, 8, 11]
point_test_sizes = list(range(20480, 512000, 20480))

# For readibility
X = 0
Y = 1


def calculate_centroid(poly):
    """Calculates centroid of a polygon

    Args:
        poly (ndarray): Vertices of the polygon, (2,n)

    Returns:
        ndarray: centroid, (2,)
    """    
    
    v = poly # vertices
    v_r = np.roll(v, 1, axis=1) # rolled vertices

    a = v_r[X] * v[Y] - v[X] * v_r[Y]
    area = a.sum() / 2
    centroid = ((v + v_r) @ a) / (6 * area)

    return centroid


def calculate_generators(poly):
    """Calculates voronoi generator points as a centroidal voronoi polygon

    Args:
        poly (ndarray): Vertices of the polygon, (2,n)

    Returns:
        ndarray: Generator points, (2,(n+1))
    """    

    p_0 = calculate_centroid(poly)

    v = poly # vertices
    v_r = np.roll(v, 1, axis=1) # rolled vertices

    a = v[Y] - v_r[Y]
    b = v_r[X] - v[X]
    c = - (a * v[X] + b * v[Y])

    w = np.array([[b**2 - a**2, -2 * a * b],
                  [-2 * a * b, a**2 - b**2]])

    p_k = (np.einsum('ijk,j', w, p_0) - 2 * c * np.array([a, b])) / (a**2 + b**2)

    return np.hstack((p_0.reshape(-1,1), p_k))


def voronoi(points, poly):
    """Voronoi point inclusion test

    Args:
        points (ndarray): Test points, (2,m)
        poly (ndarray): Vertices of the polygon, (2,n)

    Returns:
        ndarray(dtype=bool): Result of the point inclusion test, (m,)
    """    
    
    generators = calculate_generators(poly)

    x_minus_p = points[:,np.newaxis,:] - generators[...,np.newaxis]
    metrics = np.einsum("ijk,ijk->jk", x_minus_p, x_minus_p)
    result = (metrics[0:1] < metrics[1:]).all(axis=0)
    
    return result


def crossing(points, poly):
    """Ray crossings point inclusion test

    Args:
        points (ndarray): Test points, (2,m)
        poly (ndarray): Vertices of the polygon, (2,n)

    Returns:
        ndarray(dtype=bool): Result of the point inclusion test, (m,)
    """    

    q = points[:,np.newaxis,:] # queried points
    v = poly[...,np.newaxis] # vertices
    vr = np.roll(v, 1, axis=1) # rolled vertices

    v_delta = v - vr # differences between successive vertices

    in_range = np.logical_xor(v[Y] > q[Y], vr[Y] > q[Y])

    going_up = v[Y] > vr[Y]

    lhs = q[Y] * v_delta[X] - q[X] * v_delta[Y] 
    rhs = vr[Y] * v_delta[X] - vr[X] * v_delta[Y]

    on_left = np.where(going_up, lhs > rhs, lhs < rhs)

    crossings = np.logical_and(in_range, on_left)

    result = (crossings.sum(axis=0) % 2) != 0

    return result


def sign_of_offset(points, poly):
    """Sign of offset point inclusion test

    Args:
        points (ndarray): Test points, (2,m)
        poly (ndarray): Vertices of the polygon, (2,n)

    Returns:
        ndarray(dtype=bool): Result of the point inclusion test, (m,)
    """    

    q = points[:,np.newaxis,:] # queried points
    v = poly[...,np.newaxis] # vertices
    vr = np.roll(v, 1, axis=1) # rolled vertices

    v_delta = v - vr # differences between successive vertices

    lhs = q[Y] * v_delta[X] - q[X] * v_delta[Y] 
    rhs = vr[Y] * v_delta[X] - vr[X] * v_delta[Y]

    # Check if all True or all False, no mix
    result = (lhs < rhs).sum(axis=0) % poly.shape[1] == 0

    return result


def transform(poly, xytheta):
    """Affine transformation of the polygon

    Args:
        poly (ndarray): Vertices of the polygon, (2,n)
        xytheta (tuple): x, y, theta

    Returns:
        (ndarray): Vertices of transformed polygon, (2,n)
    """    

    augmented = np.vstack((poly, np.ones(poly.shape[1])))
    x, y, t = xytheta
    transform = np.asarray([[cos(t), -sin(t), x],
                            [sin(t), cos(t), y],
                            [0, 0, 1]])
    tfed_poly = transform @ augmented

    return tfed_poly[0:2,:]


def visualize_test(points, poly, result, title, save=False, plot_generators=False):
    plt.figure(figsize=(5, 5))
    plt.title("{} edges {}".format(poly.shape[1], title))
    plt.axis('equal')
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92)
    plt.fill(poly[0,:], poly[1,:], "#75bbfd")
    xx = points[0,:]
    yy = points[1,:]
    plt.plot(xx[result], yy[result], 'ro', markersize=1)
    plt.plot(xx[np.logical_not(result)], yy[np.logical_not(result)], 'ko', markersize=1)
    if plot_generators:
        generators = calculate_generators(poly)
        plt.plot(generators[0], generators[1], 'go', markersize=7)

    if save:
        plt.savefig("{} edges {}.pdf".format(poly.shape[1], title))
    else:
        plt.show()


def create_convex_poly(n_vertices=7, radius=1.0, center=(0.0,0.0)):

    theta = np.linspace(0, 2*pi, n_vertices, False)
    vertices = radius * np.array([np.cos(theta), np.sin(theta)]) + np.array(center).reshape(2,1)

    return vertices


def setup_figure():
    fig = plt.figure(figsize=(5, 13))
    # Turn off axis lines and ticks of the big subplot
    big_ax = fig.add_subplot(111)
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # Set common labels
    big_ax.set_xlabel("Number of points")
    # big_ax.set_ylabel("Per point processing time (ns)")
    axes = fig.subplots(3, 1, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.98, bottom=0.04)
    
    return fig, axes


def generate_timing_plot(all_results):
    
    fig, axes = setup_figure()

    for idx, n_vertices in enumerate(all_results.keys()):
        timings = all_results[n_vertices]
        ax = axes[idx]

        linestyles = ['-.', '--', '-', ':']
        ax.set_title("Tests for {} edges".format(n_vertices))
        ax.set_yscale("log")
        ax.grid(True, 'both')
        # ax.set_xlabel("Number of points")
        ax.set_ylabel("Per point processing time (ns)")
        for i, method in enumerate(timings.keys()):
            x = []
            y = []
            for n_pts in timings[method].keys():
                x.append(n_pts)
                y.append(timings[method][n_pts])
            ax.plot(x, y, label=method, linestyle=linestyles[i], linewidth=3)
        ax.legend()
    
    fig.savefig("experimental.pdf")


def experimental_results():

    all_results = {}

    for idx, n_vertices in enumerate(polygon_test_sizes):

        timings = {"ray crossing" : {},
                   "sign of offset" : {},
                   "voronoi" : {}}

        poly = create_convex_poly(n_vertices=n_vertices, radius=radius_of_poly)

        tfed_poly = transform(poly, (*center_of_poly, rotation_of_poly))

        test_points = (rg.random((2,point_test_sizes[-1])) - 0.5) * 3 * radius_of_poly + np.array(center_of_poly).reshape(2,1)


        # ################ RAY CROSSING TEST ################
        for n_points in point_test_sizes:

            timer_crossing = timeit.Timer(lambda : crossing(test_points[:,:n_points], tfed_poly))
            crossing_time = timer_crossing.repeat(repeat=REPEAT, number=NUMBER)
            crossing_time = np.min(crossing_time) / NUMBER / n_points / TIME_SCALE
            print("Timing with ray crossing for {} vertices, {} points: {} ns/pt".format(n_vertices, n_points, crossing_time))
            
            timings["ray crossing"][n_points] = crossing_time            


        # ################ SIGN OF OFFSET TEST ################
        for n_points in point_test_sizes:

            timer_signoff = timeit.Timer(lambda : sign_of_offset(test_points[:,:n_points], tfed_poly))
            signoff_time = timer_signoff.repeat(repeat=REPEAT, number=NUMBER)
            signoff_time = np.min(signoff_time) / NUMBER / n_points / TIME_SCALE
            print("Timing with sign of offset for {} vertices, {} points: {} ns/pt".format(n_vertices, n_points, signoff_time))
            
            timings["sign of offset"][n_points] = signoff_time


        # ################ VORONOI TEST ################
        for n_points in point_test_sizes:
            
            timer_voronoi = timeit.Timer(lambda : voronoi(test_points[:,:n_points], tfed_poly))
            voronoi_time = timer_voronoi.repeat(repeat=REPEAT, number=NUMBER)
            voronoi_time = np.min(voronoi_time) / NUMBER / n_points / TIME_SCALE
            print("Timing with voronoi for {} vertices, {} points: {} ns/pt".format(n_vertices, n_points, voronoi_time))

            timings["voronoi"][n_points] = voronoi_time


        all_results[n_vertices] = timings

    return all_results


def near_poly_sample(poly, n_points, k_nearness=0.1):
    
    q_i = poly
    q_j = np.roll(q_i, 1, axis=1)

    a = q_i[1] - q_j[1]
    b = q_j[0] - q_i[0]
    c = - (a * q_i[0] + b * q_i[1])

    x = (rg.random((2,n_points)) - 0.5) * 2 * (1 + 2 * k_nearness) * radius_of_poly + np.array(center_of_poly).reshape(2,1)
    centroid = calculate_centroid(poly)
    in_radius = np.linalg.norm(x - centroid.reshape(2,1), axis=0) < ((1 + k_nearness) * radius_of_poly)
    x = x[:,in_radius]
    
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # d = (ax + by + c) / sqrt(a^2 + b^2)
    d = (np.outer(a, x[0]) + np.outer(b, x[1]) + c.reshape(-1,1)) / np.sqrt(a**2 + b**2).reshape(-1,1)
    
    x_near = x[:,(np.abs(d) < (k_nearness * radius_of_poly)).any(axis=0)]
    
    return x_near


def correctness_test():
    poly = create_convex_poly(n_vertices=5, radius=radius_of_poly)
    tfed_poly = transform(poly, (*center_of_poly, rotation_of_poly))
    test_points =  near_poly_sample(tfed_poly, 10000, 0.5)

    c_test = crossing(test_points, tfed_poly)
    v_test = voronoi(test_points, tfed_poly)

    if (c_test != v_test).sum() == 0:
        # Test passed
        visualize_test(test_points, tfed_poly, voronoi(test_points, tfed_poly), "voronoi", True, True)
    else:
        print("Correctness test failed")


def table_generation(data):

    table = []
    table.append(["", "", "\multicolumn{3}{|c|}{Algorithms}", "\\\\ %"])
    table.append(["\cline{3-5} %"])
    table.append(["",      "",       "Ray",      "Sign of","",        "Diff"])
    table.append(["Edges", "Points", "Crossing", "Offset", "Voronoi", "\%"])
    n_v_range = list(data.keys())
    for n_v in n_v_range:
        table.append(["\\hline %"])
        timings = data[n_v]
        methods = list(timings.keys())
        for method in methods:
            result = timings[method]
            n_p_range = list(result.keys())
            short_n_p_range = [n_p_range[0], 
                               n_p_range[1 * len(n_p_range) // 6],
                               n_p_range[2 * len(n_p_range) // 6],
                               n_p_range[3 * len(n_p_range) // 6],
                               n_p_range[4 * len(n_p_range) // 6],
                               n_p_range[5 * len(n_p_range) // 6],
                               n_p_range[-1]]
            for idx, n_p in enumerate(short_n_p_range):
                row = []
                row.append(n_v if idx == 3 else "")
                row.append(n_p)
                timez = np.array([timings[methods[0]][n_p], timings[methods[1]][n_p], timings[methods[2]][n_p]])
                for t in timez:
                    if t == timez.min():
                        row.append("\\textbf{{{:3.2f}}}".format(t))
                    else:
                        row.append("{:3.2f}".format(t))
                best_two = np.partition(timez, (0,1))
                diff_percent = (best_two[1] - best_two[0]) / best_two[1] * 100
                row.append("{:2.1f}".format(diff_percent))
                table.append(row)

            break # do not loop the loop

    tabular_latex = tabulate(table, tablefmt="latex_raw")
    print(tabular_latex)

    with open('timings.tex', "w") as f:
        f.write(tabular_latex)

    return table


def main():

    all_results = experimental_results()

    with open('exp_results.pickle', 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

    # with open('exp_results.pickle', 'rb') as f:
    #     all_results = pickle.load(f)

    generate_timing_plot(all_results)
    table_generation(all_results)
    correctness_test()

    # poly = create_convex_poly(n_vertices=5, radius=radius_of_poly)
    # tfed_poly = transform(poly, (*center_of_poly, rotation_of_poly))

    # test_points = near_poly_sample(tfed_poly, 10000, 0.2)

    # visualize_test(test_points, tfed_poly, crossing(test_points, tfed_poly), "ray crossing")
    # visualize_test(test_points, tfed_poly, sign_of_offset(test_points, tfed_poly), "sign of offset")
    # visualize_test(test_points, tfed_poly, voronoi(test_points, tfed_poly), "voronoi")


if __name__ == "__main__":
    main()