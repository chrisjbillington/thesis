from __future__ import division, print_function
import os
import sys
import time
import traceback

import numpy as np
from mpi4py import MPI
import h5py

from FEDVR import FiniteElements2D


# Some slice objects for conveniently slicing axes in multidimensional arrays:
FIRST = np.s_[:1]
LAST = np.s_[-1:]
ALL = np.s_[:]
INTERIOR = np.s_[1:-1]
ALL_BUT_FIRST = np.s_[1:]
ALL_BUT_LAST = np.s_[:-1]

# Each of the following is (x_elements, y_elements, x_points, y_points)
ALL_ELEMENTS_AND_POINTS = (ALL, ALL, ALL, ALL)
LEFT_BOUNDARY = (FIRST, ALL, FIRST, ALL) # The points on the left edge of the left boundary element
RIGHT_BOUNDARY = (LAST, ALL, LAST, ALL) # The points on the right edge of the right boundary element
BOTTOM_BOUNDARY = (ALL, FIRST, ALL, FIRST) # The points on the bottom edge of the bottom boundary element
TOP_BOUNDARY = ( ALL, LAST, ALL, LAST) # The points on the top edge of the top boundary element
INTERIOR_ELEMENTS = (INTERIOR, INTERIOR, ALL, ALL) # All points in all elements that are not boundary elements
INTERIOR_POINTS = (ALL, ALL, INTERIOR, INTERIOR) # Points that are not edgepoints in all elements
LEFT_INTERIOR_EDGEPOINTS = (ALL_BUT_FIRST, ALL, FIRST, ALL) # All points on the left edge of a non-border element
RIGHT_INTERIOR_EDGEPOINTS = (ALL_BUT_LAST, ALL, LAST, ALL) # All points on the right edge of a non-border element
BOTTOM_INTERIOR_EDGEPOINTS = (ALL, ALL_BUT_FIRST, ALL, FIRST) # All points on the bottom edge of a non-border element
TOP_INTERIOR_EDGEPOINTS = (ALL, ALL_BUT_LAST, ALL, LAST) # All points on the top edge of a non-border element
# All but the last points in the x and y directions, so that we don't double count them when summing.
DONT_DOUBLE_COUNT_EDGES = (ALL, ALL, ALL_BUT_LAST, ALL_BUT_LAST)


def get_factors(n):
    """return all the factors of n"""
    factors = set()
    for i in range(1, int(n**(0.5)) + 1):
        if not n % i:
            factors.update((i, n // i))
    return factors


def get_best_2D_segmentation(size_x, size_y, N_segments):
    """Returns (best_n_segments_x, best_n_segments_y), describing the optimal
    cartesian grid for splitting up a rectangle of size (size_x, size_y) into
    N_segments equal sized segments such as to minimise surface area between
    the segments."""
    lowest_surface_area = None
    for n_segments_x in get_factors(N_segments):
        n_segments_y = N_segments // n_segments_x
        surface_area = n_segments_x * size_y + n_segments_y * size_x
        if lowest_surface_area is None or surface_area < lowest_surface_area:
            lowest_surface_area = surface_area
            best_n_segments_x, best_n_segments_y = n_segments_x, n_segments_y
    return best_n_segments_x, best_n_segments_y


def format_float(x, sigfigs=4, units=''):
    """Returns a string of the float f with a limited number of sig figs and a metric prefix"""

    prefixes = {
        -24: u"y",
        -21: u"z",
        -18: u"a",
        -15: u"f",
        -12: u"p",
        -9: u"n",
        -6: u"u",
        -3: u"m",
        0: u"",
        3: u"k",
        6: u"M",
        9: u"G",
        12: u"T",
        15: u"P",
        18: u"E",
        21: u"Z",
        24: u"Y"
    }

    if np.isnan(x) or np.isinf(x):
        return str(x)

    if x != 0:
        exponent = int(np.floor(np.log10(np.abs(x))))
        # Only multiples of 10^3
        exponent = int(np.floor(exponent / 3) * 3)
    else:
        exponent = 0

    significand = x / 10 ** exponent
    pre_decimal, post_decimal = divmod(significand, 1)
    digits = sigfigs - len(str(int(pre_decimal)))
    significand = round(significand, digits)
    result = str(significand)
    if exponent:
        try:
            # If our number has an SI prefix then use it
            prefix = prefixes[exponent]
            result += ' ' + prefix
        except KeyError:
            # Otherwise display in scientific notation
            result += 'e' + str(exponent)
            if units:
                result += ' '
    elif units:
        result += ' '
    return result + units


class Simulator2D(object):
    def __init__(self, x_min_global, x_max_global, y_min_global, y_max_global,
                 n_elements_x_global, n_elements_y_global, Nx, Ny, n_components,
                 output_filepath=None, single_output_file=False, resume=False, natural_units=False):
        """A class for simulating a the nonlinear Schrodinger equation in two
        spatial dimensions with the finite element discrete variable
        representation, on multiple cores if using MPI"""
        if (n_elements_x_global % 2):
            raise ValueError("Odd-even split step method requires even n_elements_x_global")
        if (n_elements_y_global % 2):
            raise ValueError("Odd-even split step method requires even n_elements_y_global")
        self.x_min_global = x_min_global
        self.x_max_global = x_max_global
        self.y_min_global = y_min_global
        self.y_max_global = y_max_global
        self.n_elements_x_global = n_elements_x_global
        self.n_elements_y_global = n_elements_y_global
        self.Nx = Nx
        self.Ny = Ny
        self.n_components = n_components
        self.output_filepath = output_filepath
        self.single_output_file = single_output_file
        self.resume = resume
        self.element_width_x = (self.x_max_global - self.x_min_global)/self.n_elements_x_global
        self.element_width_y = (self.y_max_global - self.y_min_global)/self.n_elements_y_global

        self._setup_MPI_grid()

        self.elements = FiniteElements2D(self.n_elements_x, self.n_elements_y, Nx, Ny,
                                         n_components, self.x_min, self.x_max, self.y_min, self.y_max)

        self.shape = self.elements.shape
        self.global_shape = (self.n_elements_x_global, self.n_elements_y_global, self.Nx, self.Ny, self.n_components, 1)
        self.local_shape = (self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)

        # Derivative operators, shapes (Nx, 1, 1, Nx) and (Ny, 1, Ny):
        self.gradx, self.grady = self.elements.derivative_operators()
        self.grad2x, self.grad2y = self.elements.second_derivative_operators()

        # Density operator. Is diagonal and so is represented as an (Nx, Ny, 1, 1)
        # array containing its diagonals:
        self.density_operator = self.elements.density_operator()

        # The x spatial points of the DVR basis functions, an (n_elements_x, 1, Nx, 1, 1, 1) array:
        self.x = self.elements.points_x
        # The y spatial points of the DVR basis functions, an (n_elements_y, 1, Ny, 1, 1) array:
        self.y = self.elements.points_y

        self.natural_units = natural_units
        if natural_units:
            self.hbar = 1
            self.time_units = ''
        else:
            self.hbar = 1.054571726e-34
            self.time_units = 's'

        if self.output_filepath is not None and not (os.path.exists(self.output_filepath) and self.resume):
            if self.single_output_file:
                self.output_file = h5py.File(self.output_filepath, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            else:
                # Only rank 0 should create the output folder if it doesn't already exist
                if not self.MPI_rank and not os.path.isdir(self.output_filepath):
                    os.mkdir(self.output_filepath)
                output_file_basename = str(self.MPI_rank).zfill(len(str(self.MPI_size))) + '.h5'
                output_filepath = os.path.join(self.output_filepath, output_file_basename)
                # Ensure output folder exists before other processes continue:
                self.MPI_comm.Barrier()
                self.output_file = h5py.File(output_filepath, 'w')

            self.output_file.attrs['x_min_global'] = x_min_global
            self.output_file.attrs['x_max_global'] = x_max_global
            self.output_file.attrs['y_min_global'] = y_min_global
            self.output_file.attrs['y_max_global'] = y_max_global
            self.output_file.attrs['element_width_x'] = self.element_width_x
            self.output_file.attrs['element_width_y'] = self.element_width_y
            self.output_file.attrs['n_elements_x_global'] = self.n_elements_x_global
            self.output_file.attrs['n_elements_y_global'] = self.n_elements_y_global
            self.output_file.attrs['Nx'] = Nx
            self.output_file.attrs['Ny'] = Ny
            self.output_file.attrs['global_shape'] = self.global_shape
            geometry_dtype = [('rank', int),
                              ('processor_name', 'a256'),
                              ('x_cart_coord', int),
                              ('y_cart_coord', int),
                              ('first_element_x', int),
                              ('first_element_y', int),
                              ('n_elements_x', int),
                              ('n_elements_y', int)]
            MPI_geometry_dset = self.output_file.create_dataset('MPI_geometry',
                                                                shape=(self.MPI_size,), dtype=geometry_dtype)
            MPI_geometry_dset.attrs['MPI_size'] = 1 if self.single_output_file else self.MPI_size
            data = (self.MPI_rank, self.processor_name,self.MPI_x_coord, self.MPI_y_coord,
                    self.global_first_x_element, self.global_first_y_element, self.n_elements_x, self.n_elements_y)
            MPI_geometry_dset[0 if self.single_output_file else self.MPI_rank] = data
            self.output_file.create_group('output')

        # A dictionary for keeping track of what row we're up to in file
        # output in each group:
        self.output_row = {}

        # Slices for convenient indexing:
        EDGE_POINTS_X = np.s_[::self.Nx-1]
        EDGE_POINTS_Y = np.s_[::self.Ny-1]
        BOUNDARY_ELEMENTS_X = np.s_[::self.n_elements_x-1]
        BOUNDARY_ELEMENTS_Y = np.s_[::self.n_elements_y-1]

        # These are for indexing all edges of all boundary elements. The below
        # four sets of slices used in succession cover the edges of boundary
        # elements exactly once:
        self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_X = (BOUNDARY_ELEMENTS_X, ALL, EDGE_POINTS_X, ALL)
        self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_X = (INTERIOR, BOUNDARY_ELEMENTS_Y, EDGE_POINTS_X, ALL)
        self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_Y = (BOUNDARY_ELEMENTS_X, ALL, INTERIOR, EDGE_POINTS_Y)
        self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_Y = (INTERIOR, BOUNDARY_ELEMENTS_Y, INTERIOR, EDGE_POINTS_Y)

        # These are for indexing all edges of non-boundary elements. Used in
        # succession they cover the edges of these elements exactly once:
        self.INTERIOR_ELEMENTS_EDGE_POINTS_X = (INTERIOR, INTERIOR, EDGE_POINTS_X, ALL)
        self.INTERIOR_ELEMENTS_EDGE_POINTS_Y = (INTERIOR, INTERIOR, INTERIOR, EDGE_POINTS_Y)

        # These are for indexing all points of boundary elements. Used in
        # succession they cover the these elements exactly once:
        self.BOUNDARY_ELEMENTS_X_ALL_POINTS = (BOUNDARY_ELEMENTS_X, ALL, ALL, ALL)
        self.BOUNDARY_ELEMENTS_Y_ALL_POINTS = (INTERIOR, BOUNDARY_ELEMENTS_Y, ALL, ALL)

    def _setup_MPI_grid(self):
        """Split space up according to the number of MPI tasks. Set instance
        attributes for spatial extent and number of elements in this MPI task,
        and create buffers and persistent communication requests for sending
        data to adjacent processes"""

        self.MPI_size = MPI.COMM_WORLD.Get_size()
        self.MPI_size_x, self.MPI_size_y = get_best_2D_segmentation(
                                               self.n_elements_x_global, self.n_elements_y_global, self.MPI_size)
        self.MPI_comm = MPI.COMM_WORLD.Create_cart([self.MPI_size_x, self.MPI_size_y],
                                                   periods=[True, True], reorder=True)
        self.MPI_rank = self.MPI_comm.Get_rank()
        self.MPI_x_coord, self.MPI_y_coord = self.MPI_comm.Get_coords(self.MPI_rank)
        self.MPI_rank_left = self.MPI_comm.Get_cart_rank((self.MPI_x_coord - 1, self.MPI_y_coord))
        self.MPI_rank_right = self.MPI_comm.Get_cart_rank((self.MPI_x_coord + 1, self.MPI_y_coord))
        self.MPI_rank_down = self.MPI_comm.Get_cart_rank((self.MPI_x_coord, self.MPI_y_coord - 1))
        self.MPI_rank_up = self.MPI_comm.Get_cart_rank((self.MPI_x_coord, self.MPI_y_coord + 1))
        self.processor_name = MPI.Get_processor_name()

        # We need an even number of elements in each direction per process. So let's share them out.
        x_elements_per_process, remaining_x_elements = (int(2*n)
                                                        for n in divmod(self.n_elements_x_global / 2, self.MPI_size_x))
        self.n_elements_x = x_elements_per_process
        if self.MPI_x_coord < remaining_x_elements/2:
            # Give the remaining to the lowest ranked processes:
            self.n_elements_x += 2

        y_elements_per_process, remaining_y_elements = (int(2*n)
                                                        for n in divmod(self.n_elements_y_global / 2, self.MPI_size_y))
        self.n_elements_y = y_elements_per_process
        if self.MPI_y_coord < remaining_y_elements/2:
            # Give the remaining to the lowest ranked processes:
            self.n_elements_y += 2

        # Where in the global array of elements are we?
        self.global_first_x_element = x_elements_per_process * self.MPI_x_coord
        # Include the extra elements some tasks have:
        if self.MPI_x_coord < remaining_x_elements/2:
            self.global_first_x_element += 2*self.MPI_x_coord
        else:
            self.global_first_x_element += remaining_x_elements

        self.global_first_y_element = y_elements_per_process * self.MPI_y_coord
        # Include the extra elements some tasks have:
        if self.MPI_y_coord < remaining_y_elements/2:
            self.global_first_y_element += 2*self.MPI_y_coord
        else:
            self.global_first_y_element += remaining_y_elements

        self.x_min = self.x_min_global + self.element_width_x * self.global_first_x_element
        self.x_max = self.x_min + self.element_width_x * self.n_elements_x

        self.y_min = self.y_min_global + self.element_width_y * self.global_first_y_element
        self.y_max = self.y_min + self.element_width_y * self.n_elements_y


        # The data we want to send to adjacent processes isn't in contiguous
        # memory, so we need to copy it into and out of temporary buffers:

        # Buffers for operating on psi with operators that are non-diagonal in
        # the spatial basis, requiring summing contributions from adjacent
        # elements:
        self.MPI_left_kinetic_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_left_kinetic_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_kinetic_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_kinetic_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_top_kinetic_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_top_kinetic_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_bottom_kinetic_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_bottom_kinetic_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

        # Buffers for sending values of psi to adjacent processes. Values are
        # supposed to be identical on edges of adjacent elements, but due to
        # rounding error they may not stay perfectly identical. This can be a
        # problem, so we send the values across from one to the other once a
        # timestep to keep them agreeing.
        self.MPI_left_values_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_values_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_bottom_values_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_top_values_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

        # We need to tag our data to have a way other than rank to distinguish
        # between multiple messages the two tasks might be sending each other
        # at the same time:
        TAG_LEFT_TO_RIGHT_KINETIC = 0
        TAG_RIGHT_TO_LEFT_KINETIC = 1
        TAG_DOWN_TO_UP_KINETIC = 2
        TAG_UP_TO_DOWN_KINETIC = 3
        TAG_RIGHT_TO_LEFT_VALUES = 4
        TAG_UP_TO_DOWN_VALUES = 5

        # Create persistent requests for the data transfers we will regularly be doing:
        self.MPI_send_kinetic_left = self.MPI_comm.Send_init(self.MPI_left_kinetic_send_buffer,
                                                             self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT_KINETIC)
        self.MPI_send_kinetic_right = self.MPI_comm.Send_init(self.MPI_right_kinetic_send_buffer,
                                                              self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT_KINETIC)
        self.MPI_receive_kinetic_left = self.MPI_comm.Recv_init(self.MPI_left_kinetic_receive_buffer,
                                                                self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT_KINETIC)
        self.MPI_receive_kinetic_right = self.MPI_comm.Recv_init(self.MPI_right_kinetic_receive_buffer,
                                                                 self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT_KINETIC)
        self.MPI_send_kinetic_bottom = self.MPI_comm.Send_init(self.MPI_bottom_kinetic_send_buffer,
                                                               self.MPI_rank_down, tag=TAG_UP_TO_DOWN_KINETIC)
        self.MPI_send_kinetic_top = self.MPI_comm.Send_init(self.MPI_top_kinetic_send_buffer,
                                                            self.MPI_rank_up, tag=TAG_DOWN_TO_UP_KINETIC)
        self.MPI_receive_kinetic_bottom = self.MPI_comm.Recv_init(self.MPI_bottom_kinetic_receive_buffer,
                                                                  self.MPI_rank_down, tag=TAG_DOWN_TO_UP_KINETIC)
        self.MPI_receive_kinetic_top = self.MPI_comm.Recv_init(self.MPI_top_kinetic_receive_buffer,
                                                               self.MPI_rank_up, tag=TAG_UP_TO_DOWN_KINETIC)
        self.MPI_send_values_left = self.MPI_comm.Send_init(self.MPI_left_values_send_buffer,
                                                            self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT_VALUES)
        self.MPI_receive_values_right = self.MPI_comm.Recv_init(self.MPI_right_values_receive_buffer,
                                                                self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT_VALUES)
        self.MPI_send_values_down = self.MPI_comm.Send_init(self.MPI_bottom_values_send_buffer,
                                                            self.MPI_rank_down, tag=TAG_UP_TO_DOWN_VALUES)
        self.MPI_receive_values_up = self.MPI_comm.Recv_init(self.MPI_top_values_receive_buffer,
                                                             self.MPI_rank_up, tag=TAG_UP_TO_DOWN_VALUES)

        self.MPI_kinetic_requests = [self.MPI_send_kinetic_left, self.MPI_receive_kinetic_left,
                                     self.MPI_send_kinetic_right, self.MPI_receive_kinetic_right,
                                     self.MPI_send_kinetic_bottom, self.MPI_receive_kinetic_bottom,
                                     self.MPI_send_kinetic_top, self.MPI_receive_kinetic_top]

        self.MPI_values_requests = [self.MPI_send_values_left, self.MPI_receive_values_right,
                                    self.MPI_send_values_down, self.MPI_receive_values_up]

    def MPI_send_border_kinetic(self, Kx_psi, Ky_psi):
        """Start an asynchronous MPI send to all adjacent MPI processes,
        sending them the values of H_nondiag_psi on the borders"""
        self.MPI_left_kinetic_send_buffer[:] = Kx_psi[LEFT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_right_kinetic_send_buffer[:] = Kx_psi[RIGHT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_bottom_kinetic_send_buffer[:] = Ky_psi[BOTTOM_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        self.MPI_top_kinetic_send_buffer[:] = Ky_psi[TOP_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        MPI.Prequest.Startall(self.MPI_kinetic_requests)

    def MPI_receive_border_kinetic(self, Kx_psi, Ky_psi):
        """Finalise an asynchronous MPI transfer from all adjacent MPI processes,
        receiving values into H_nondiag_psi on the borders"""
        MPI.Prequest.Waitall(self.MPI_kinetic_requests)
        left_data = self.MPI_left_kinetic_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny, 1, 1))
        right_data = self.MPI_right_kinetic_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny, 1, 1))
        bottom_data = self.MPI_bottom_kinetic_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1, 1, 1))
        top_data = self.MPI_top_kinetic_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1, 1, 1))
        Kx_psi[LEFT_BOUNDARY] += left_data
        Kx_psi[RIGHT_BOUNDARY] += right_data
        Ky_psi[BOTTOM_BOUNDARY] += bottom_data
        Ky_psi[TOP_BOUNDARY] += top_data

    def MPI_send_border_values(self, psi):
        """Start an asynchronous send to the MPI processes left and down from
        us, sending them our values of psi on those borders. This to ensure
        that values of psi on shared edges are numerically identical on
        elements both sides of the edge. Mathematically they should be
        identical without us doing this, but due to rounding error they may
        not be."""
        self.MPI_left_values_send_buffer[:] = psi[LEFT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_bottom_values_send_buffer[:] = psi[BOTTOM_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        MPI.Prequest.Startall(self.MPI_values_requests)

    def MPI_receive_border_values(self, psi):
        """Finalise an asynchronous MPI transfer from the MPI processes right
        and up from us,, receiving values into psi on the borders"""
        MPI.Prequest.Waitall(self.MPI_values_requests)
        right_data = self.MPI_right_values_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny, 1, 1))
        top_data = self.MPI_top_values_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1, 1, 1))
        psi[RIGHT_BOUNDARY] = right_data
        psi[TOP_BOUNDARY] = top_data

    def global_dot(self, vec1, vec2):
        """"Dots two vectors and sums result over MPI processes"""
        # Don't double count edges
        if vec1.shape != self.shape or vec2.shape != self.shape:
            message = ('arguments must both have shape self.shape=%s, '%str(self.shape) +
                       'but they are %s and %s'%(str(vec1.shape), str(vec2.shape)))
            raise ValueError(message)
        local_dot = np.vdot(vec1[DONT_DOUBLE_COUNT_EDGES], vec2[DONT_DOUBLE_COUNT_EDGES]).real
        local_dot = np.asarray(local_dot).reshape(1)
        result = np.zeros(1)
        self.MPI_comm.Allreduce(local_dot, result, MPI.SUM)
        return result[0]

    def compute_number(self, psi):
        return self.global_dot(psi, psi)

    def normalise(self, psi, N_2D):
        """Normalise psi to the 2D normalisation constant N_2D, which has
        units of a linear density"""
        # imposing normalisation on the wavefunction:
        ncalc = self.global_dot(psi, psi)
        psi[:] *= np.sqrt(N_2D/ncalc)

    def compute_H(self, t, psi, H, boundary_element_slices=(), internal_element_slices=(),
                  sum_at_edges=True, outarrays=None):
        """Applies the Hamiltonian H to the wavefunction psi at time t, sums
        the kinetic terms at element edges and MPI task borders, and returns
        the resulting three terms K_psi, U_psi, andn U_nonlinear. We don't sum
        them together here because the caller may with to treat them
        separately in an energy calculation, such as dividing the nonlinear
        term by two before summing up the total energy. The nonlinear term is
        returned without having been multiplied by psi.

        boundary_element_slices and internal_element slices can be provided to
        specify which points should be evaluated. If both are empty, all
        points will be evaluated."""

        # optimisation, don't create arrays if the user has provided them:
        if outarrays is None:
            Kx_psi = np.empty(psi.shape, dtype=psi.dtype)
            Ky_psi = np.empty(psi.shape, dtype=psi.dtype)
            K_psi = np.empty(psi.shape, dtype=psi.dtype)
            U_psi = np.empty(psi.shape, dtype=psi.dtype)
            U_nonlinear = np.empty(psi.shape, dtype=psi.dtype)
        else:
            Kx_psi, Ky_psi, K_psi, U_psi, U_nonlinear = outarrays

        # Compute H_psi at the boundary elements first, before firing off data
        # to other MPI tasks. Then compute H_psi on the internal elements,
        # before adding in the contributions from adjacent processes at the
        # last moment. This lets us cater to as much latency in transport as
        # possible. If the caller has provided boundary_element_slices and
        # internal_element_slices, then don't evaluate all all the points, just at the
        # ones requested. Basically pre_MPI_slices must contain any boundary
        # points that the caller requires, and for maximum efficiency should
        # contain as few as possible other points - they should be be in
        # post_MPI_slices and be evaluated after the MPI send has been done.

        if not (boundary_element_slices or internal_element_slices):
            boundary_element_slices = (self.BOUNDARY_ELEMENTS_X_ALL_POINTS, self.BOUNDARY_ELEMENTS_Y_ALL_POINTS)
            internal_element_slices = (INTERIOR_ELEMENTS,)

        # Evaluate H_psi at the boundary element slices, if any:
        for slices in boundary_element_slices:
            x_elements, y_elements, x_points, y_points = slices
            Kx, Ky, U, U_nonlinear[slices] = H(t, psi, *slices)
            Kx_psi[slices] = np.einsum('...nmci,...imcq->...nmcq', Kx,  psi[x_elements, y_elements, :, y_points])
            Ky_psi[slices] = np.einsum('...mcj,...jcq->...mcq', Ky,  psi[x_elements, y_elements, x_points, :])
            U_psi[slices] = np.einsum('...cl,...lq->...cq', U,  psi[slices])

        if boundary_element_slices and sum_at_edges:
            # Send values on the border to adjacent MPI tasks:
            self.MPI_send_border_kinetic(Kx_psi, Ky_psi)

        # Now evaluate H_psi at the internal element slices:
        for slices in internal_element_slices:
            x_elements, y_elements, x_points, y_points = slices
            Kx, Ky, U, U_nonlinear[slices] = H(t, psi, *slices)
            Kx_psi[slices] = np.einsum('...nmci,...imcq->...nmcq', Kx,  psi[x_elements, y_elements, :, y_points])
            Ky_psi[slices] = np.einsum('...mcj,...jcq->...mcq', Ky,  psi[x_elements, y_elements, x_points, :])
            U_psi[slices] = np.einsum('...cl,...lq->...cq', U,  psi[slices])

        if sum_at_edges:
            # Add contributions to Kx_psi and Ky_psi at edges shared by interior elements.
            total_at_x_edges = Kx_psi[LEFT_INTERIOR_EDGEPOINTS] + Kx_psi[RIGHT_INTERIOR_EDGEPOINTS]
            Kx_psi[LEFT_INTERIOR_EDGEPOINTS] = Kx_psi[RIGHT_INTERIOR_EDGEPOINTS] = total_at_x_edges
            total_at_y_edges = Ky_psi[BOTTOM_INTERIOR_EDGEPOINTS] + Ky_psi[TOP_INTERIOR_EDGEPOINTS]
            Ky_psi[BOTTOM_INTERIOR_EDGEPOINTS] = Ky_psi[TOP_INTERIOR_EDGEPOINTS] = total_at_y_edges

        # Add contributions to K_psi from adjacent MPI tasks, if any were computed:
        if boundary_element_slices and sum_at_edges:
            self.MPI_receive_border_kinetic(Kx_psi, Ky_psi)

        for slices in boundary_element_slices:
            K_psi[slices] = Kx_psi[slices] + Ky_psi[slices]

        for slices in internal_element_slices:
            K_psi[slices] = Kx_psi[slices] + Ky_psi[slices]

        return K_psi, U_psi, U_nonlinear

    def compute_mu(self, t, psi, H, uncertainty=False):
        """Calculate chemical potential of DVR basis wavefunction psi with
        Hamiltonian H at time t. Optionally return its uncertainty."""

        # Total Hamiltonian operator operating on psi:
        K_psi, U_psi, U_nonlinear = self.compute_H(t, psi, H)
        H_psi = K_psi + U_psi + U_nonlinear * psi

        # Total norm:
        ncalc = self.compute_number(psi)

        # Expectation value and uncertainty of Hamiltonian gives the
        # expectation value and uncertainty of the chemical potential:
        mucalc = self.global_dot(psi, H_psi)/ncalc
        if uncertainty:
            mu2calc = self.global_dot(H_psi, H_psi)/ncalc
            var_mucalc = mu2calc - mucalc**2
            if var_mucalc < 0:
                u_mucalc = 0
            else:
                u_mucalc = np.sqrt(var_mucalc)
            return mucalc, u_mucalc
        else:
            return mucalc

    def compute_energy(self, t, psi, H, uncertainty=False):
        """Calculate the total energy of DVR basis wavefunction psi with
        Hamiltonian H at time t. Optionally return its uncertainty."""

        K_psi, U_psi, U_nonlinear = self.compute_H(t, psi, H)

        # Total energy operator. Differs from total Hamiltonian in that the
        # nonlinear term is halved in order to avoid double counting the
        # interaction energy:
        E_total_psi = K_psi + U_psi + 0.5 * U_nonlinear * psi
        Ecalc = self.global_dot(psi, E_total_psi)
        if uncertainty:
            E2calc = self.global_dot(E_total_psi, E_total_psi)
            var_Ecalc = E2calc - Ecalc**2
            if var_Ecalc < 0:
                u_Ecalc = 0
            else:
                u_Ecalc = np.sqrt(var_Ecalc)
            return Ecalc, u_Ecalc
        else:
            return Ecalc

    def find_groundstate(self, psi_guess, H, mu, t=0, convergence=1e-12, relaxation_parameter=1.7,
                         output_group=None, output_interval=100, output_callback=None, wavefunction_output=True):
        """Find the groundstate corresponding to a particular chemical
        potential using successive over-relaxation.

        H(t, psi, x_elements, y_elements, x_points, y_points) should return
        four arrays, Kx, Ky, U, and U_nonlinear, each corresponding to
        different terms in the Hamiltonian. The first two arrays, Kx and Ky,
        should be the kinetic energy operators. These should comprise only
        linear combinations of the derivative operators provided by this
        class. They should  include any terms that contain derivatives, such
        as rotation terms with first derivatives. The second array returned
        must be the sum of terms (except the nonlinear one) that are diagonal
        in the spatial basis, i.e, no derivative operators. This is typically
        the potential and and couplings between states. The third array,
        U_nonlinear, must be the nonlinear term, and can be constructed by
        multiplying the nonlinear constant by self.density_operator.

        mu should be the desired chemical potential.

        Data will be saved to a group output_group of the output file every
        output_interval steps."""
        if not self.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            print('Beginning successive over relaxation')
            print("Target chemical potential is: " + repr(mu))
            print('==========')

        psi = np.array(psi_guess, dtype=complex)

        Kx, Ky, U, U_nonlinear = H(t, psi, *ALL_ELEMENTS_AND_POINTS)
        # Get the diagonals of the Kinetic part of the Hamiltonian, shape
        # (n_elements_x, n_elements_y, Nx, Ny, n_components, 1):
        Kx_diags = np.einsum('...nmcn->...nmc', Kx).copy()
        # Broadcast Kx_diags to be the same shape as psi:
        Kx_diags = Kx_diags.reshape(Kx_diags.shape + (1,))
        Kx_diags = np.ones((self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)) * Kx_diags
        Ky_diags = np.einsum('...mcm->...mc', Ky).copy()
        # Broadcast Ky_diags to be the same shape as psi:
        Ky_diags = Ky_diags.reshape(Ky_diags.shape + (1,))
        Ky_diags = np.ones((self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)) * Ky_diags

        # Instead of summing across edges we can just multiply by two at the
        # edges to get the full values of the diagonals there:
        Kx_diags[:, :, 0] *= 2
        Kx_diags[:, :, -1] *= 2
        Ky_diags[:, :, :, 0] *= 2
        Ky_diags[:, :, :, -1] *= 2

        K_diags = Kx_diags + Ky_diags

        # The diagonal part of U, which is just the potential at each point in
        # space, shape (n_elements_x, n_elements_y, Nx, Ny, n_components, 1):
        U_diags = np.einsum('...cc->...c', U).copy()
        # Broadcast U_diags to be the same shape as psi:
        U_diags = U_diags.reshape(U_diags.shape + (1,))
        U_diags = np.ones((self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)) * U_diags

        # Empty arrays for re-using each step:
        Kx_psi = np.zeros(psi.shape, dtype=complex)
        Ky_psi = np.zeros(psi.shape, dtype=complex)
        K_psi = np.zeros(psi.shape, dtype=complex)
        U_psi = np.zeros(psi.shape, dtype=complex)
        H_diags = np.zeros(psi.shape, dtype=complex)
        H_hollow_psi = np.zeros(psi.shape, dtype=complex)
        psi_new_GS = np.zeros(psi.shape, dtype=complex)

        # All the slices for covering the edges of the boundary elements and internal elements:
        BOUNDARY_ELEMENT_EDGES = (self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_X, self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_X,
                                   self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_Y, self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_Y)
        INTERIOR_ELEMENT_EDGES = (self.INTERIOR_ELEMENTS_EDGE_POINTS_X, self.INTERIOR_ELEMENTS_EDGE_POINTS_Y)

        # Each loop we first update the edges of each element, then we loop
        # over the internal basis points. Here we just create the list of
        # slices for selecting which points we are operating on:
        point_selections = []
        # We want something we can index psi with to select edges of elements.
        # Slices can't do that, so we have to use a boolean array for the last
        # two dimensions:
        EDGE_POINTS = np.zeros((self.Nx, self.Ny), dtype=bool)
        EDGE_POINTS[FIRST] = EDGE_POINTS[LAST] = EDGE_POINTS[:, FIRST] = EDGE_POINTS[:, LAST] = True
        EDGE_POINT_SLICES = (ALL, ALL, EDGE_POINTS)
        point_selections.append(EDGE_POINT_SLICES)
        # These indicate to do internal points:
        for j in range(1, self.Nx-1):
            for k in range(1, self.Ny-1):
                # A set of slices selecting a single non-edge point in every
                # element.
                slices = (ALL, ALL, np.s_[j:j+1], np.s_[k:k+1])
                point_selections.append(slices)

        if output_group is not None:
            if self.output_filepath is None:
                msg = 'output group specified, but no output file specified for this Simulator2D object'
                raise ValueError(msg)
            if not output_group in self.output_file['output']:
                group = self.output_file['output'].create_group(output_group)
                group.attrs['start_time'] = time.time()
                if wavefunction_output:
                    wavefunction_shape = self.global_shape if self.single_output_file else self.local_shape
                    group.create_dataset('psi', (0,) + wavefunction_shape,
                                         maxshape=(None,) + wavefunction_shape,
                                         dtype=psi.dtype)
                output_log_dtype = [('step_number', int), ('mucalc', float),
                                    ('convergence', float), ('time_per_step', float)]
                group.create_dataset('output_log', (0,), maxshape=(None,), dtype=output_log_dtype)
            else:
                self.output_row[output_group] = len(self.output_file['output'][output_group]['output_log']) - 1

        def do_output():
            mucalc = self.compute_mu(t, psi, H)
            convergence_calc = abs((mucalc - mu)/mu)
            time_per_step = (time.time() - start_time)/i if i else np.nan
            message =  ('step: %d'%i +
                        '  mucalc: ' + repr(mucalc) +
                        '  convergence: %E'%convergence_calc +
                        '  time per step: {}'.format(format_float(time_per_step, units='s')))
            if not self.MPI_rank: # Only one process prints to stdout:
                sys.stdout.write(message + '\n')
            output_log = (i, mucalc, convergence_calc, time_per_step)
            if output_group is not None:
                self.output(output_group, psi, output_log, wavefunction_output)
            if output_callback is not None:
                try:
                    output_callback(psi, output_log)
                except Exception:
                    traceback.print_exc()
            return convergence_calc

        i = 0
        # Output the initial state, which is the zeroth timestep.
        do_output()
        i += 1

        # Start simulating:
        start_time = time.time()
        while True:
            # We operate on all elements at once, but only some DVR basis functions at a time.
            for slices in point_selections:
                if slices is EDGE_POINT_SLICES:
                    # Evaluating H_psi on all edges of all elements. We
                    # provide self.compute_H with lists of slices so that
                    # it can evaluate on the edges of the border
                    # elements first, before doing MPI transport, so that we
                    # can cater to high latency by doing useful work during
                    # the transport:
                    self.compute_H(t, psi, H,
                                   boundary_element_slices=BOUNDARY_ELEMENT_EDGES,
                                   internal_element_slices=INTERIOR_ELEMENT_EDGES,
                                   outarrays=(Kx_psi, Ky_psi, K_psi, U_psi, U_nonlinear))
                else:
                    # Evaluate H_psi at a single DVR point in all elements, requires no MPI communication:
                    self.compute_H(t, psi, H,
                                   internal_element_slices=(slices,),
                                   sum_at_edges=False,
                                   outarrays=(Kx_psi, Ky_psi, K_psi, U_psi, U_nonlinear))

                # Diagonals of the total Hamiltonian operator at the DVR point(s):
                H_diags[slices] = K_diags[slices] + U_diags[slices] + U_nonlinear[slices]

                # Hamiltonian with diagonals subtracted off, operating on psi at the DVR point(s):
                H_hollow_psi[slices] = K_psi[slices] - K_diags[slices] * psi[slices]

                # The Gauss-Seidel prediction for the new psi at the DVR point(s):
                psi_new_GS[slices] = (mu * psi[slices] - H_hollow_psi[slices])/H_diags[slices]

                # Update psi at the DVR point(s) with overrelaxation:
                psi[slices] += relaxation_parameter * (psi_new_GS[slices] - psi[slices])

                if slices is EDGE_POINT_SLICES:
                    # Send values on the border, to ensure border points are
                    # numerically identical despite rounding error:
                    self.MPI_send_border_values(psi)

            # Ensure edgepoints are numerically identical:
            psi[RIGHT_INTERIOR_EDGEPOINTS] = psi[LEFT_INTERIOR_EDGEPOINTS]
            psi[TOP_INTERIOR_EDGEPOINTS] = psi[BOTTOM_INTERIOR_EDGEPOINTS]
            self.MPI_receive_border_values(psi)

            if not i % output_interval:
                convergence_calc = do_output()
                if convergence_calc < convergence:
                    if not self.MPI_rank: # Only one process prints to stdout
                        print('Convergence reached')
                    break
            i += 1

        # Output the final state if we haven't already this timestep:
        if i % output_interval:
            do_output()
        if output_group is not None:
            group = self.output_file['output'][output_group]
            group.attrs['completion_time'] = time.time()
            group.attrs['run_time'] = group.attrs['completion_time'] - group.attrs['start_time']
        # Return complex array ready for time evolution:
        # psi = np.array(psi, dtype=complex)
        return psi


    def evolve(self, psi, H, dt, t_initial=0, t_final=np.inf, imaginary_time=False,
               output_group=None, output_interval=100, output_callback=None, wavefunction_output=True, method='rk4'):

        if not self.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            if self.natural_units:
                time_units = ' time units'
            else:
                time_units = 's'
            if imaginary_time:
                print("Beginning {}{} of imaginary time evolution".format(format_float(t_final), time_units))
            else:
                print("Beginning {}{} of time evolution".format(format_float(t_final), time_units))
            print('Using dt = {}{}'.format(format_float(dt), time_units))
            print('==========')

        n_initial = self.compute_number(psi)
        mu_initial = self.compute_mu(0, psi, H)
        E_initial = self.compute_energy(0, psi, H)

        if output_group is not None:
            if self.output_filepath is None:
                msg = 'output group specified, but no output file specified for this Simulator2D object'
                raise ValueError(msg)
            if not output_group in self.output_file['output']:
                group = self.output_file['output'].create_group(output_group)
                group.attrs['start_time'] = time.time()
                group.attrs['dt'] = dt
                group.attrs['t_final'] = t_final
                group.attrs['imaginary_time'] = imaginary_time
                if wavefunction_output:
                    wavefunction_shape = self.global_shape if self.single_output_file else self.local_shape
                    group.create_dataset('psi', (0,) + wavefunction_shape,
                                         maxshape=(None,) + wavefunction_shape, dtype=psi.dtype)
                output_log_dtype = [('step_number', int), ('time', float),
                                    ('number_err', float), ('energy_err', float), ('time_per_step', float)]
                group.create_dataset('output_log', (0,), maxshape=(None,), dtype=output_log_dtype)
            else:
                self.output_row[output_group] = len(self.output_file['output'][output_group]['output_log']) - 1

        def do_output():
            if imaginary_time:
                self.normalise(psi, n_initial)
            energy_err = self.compute_energy(t, psi, H) / E_initial - 1
            number_err = self.compute_number(psi) / n_initial - 1
            time_per_step = (time.time() - start_time) / i if i else np.nan
            outmessage = ('step: %d' % i +
                  '  t = {}'.format(format_float(t, units=self.time_units)) +
                  '  number_err: %+.02E' % number_err +
                  '  energy_err: %+.02E' % energy_err +
                  '  time per step: {}'.format(format_float(time_per_step, units='s')))
            if not self.MPI_rank: # Only one process prints to stdout:
                sys.stdout.write(outmessage + '\n')
            output_log = (i, t, number_err, energy_err, time_per_step)
            if output_group is not None:
                self.output(output_group, psi, output_log, wavefunction_output)
            if output_callback is not None:
                try:
                    output_callback(psi, output_log)
                except Exception:
                    traceback.print_exc()

        i = 0
        t = t_initial

        # Output the initial state, which is the zeroth timestep.
        do_output()
        i += 1

        # Start simulating:
        start_time = time.time()

        def dpsi_dt(t, psi):
            K_psi, U_psi, U_nonlinear = self.compute_H(t, psi, H)
            # H_psi = K_psi + U_psi + (U_nonlinear - mu_initial)*psi
            H_psi = K_psi + U_psi + U_nonlinear*psi
            if imaginary_time:
                return -1/self.hbar * H_psi
            else:
                return -1j/self.hbar * H_psi

        while t < t_final:
            # Send values on the border, to ensure border points are
            # numerically identical despite rounding error:
            self.MPI_send_border_values(psi)

            if method == 'rk4':
                k1 = dpsi_dt(t, psi)
                k2 = dpsi_dt(t + 0.5*dt, psi + 0.5*k1*dt)
                k3 = dpsi_dt(t + 0.5*dt, psi + 0.5*k2*dt)
                k4 = dpsi_dt(t + dt, psi + k3*dt)
            elif method == 'rk4ilip':
                # 'rk4ilip can't deal with wavefunction being zero. Make it
                # nonzero so we can divide by it, whilst keeping it small
                # enough to not affect results:'
                # very_small = 1e-100
                # psi_very_small = np.abs(psi) < very_small
                # psi[psi_very_small] = np.exp(1j*np.angle(psi[psi_very_small]))*very_small

                # Compute time derivatives at the start of the timestep:
                f1 = dpsi_dt(t, psi)

                # Use them to define an interacton picture, and compute the
                # unitary transformations for switching between pictures at
                # the various substeps of rk4:

                omega_imag = (1j*f1/psi).imag
                omega = (1j*f1/psi).real

                # We only want to use RK4ILIP when dynamics are actually
                # dominated by dynamical phase evolution:
                omega[np.abs(omega) < np.abs(omega_imag)] = 0

                # # density = np.abs(psi)**2
                # # omega[density < 1e-10*density.max()] = 0
                # RK4ILIP_THRESHHOLD = 100
                # over_threshold = np.abs(omega*dt) > RK4ILIP_THRESHHOLD
                # global max_so_far
                # try:
                #     max_so_far
                # except NameError:
                #     max_so_far = 0

                # max_this_time = np.abs(omega).max()*dt
                # max_so_far = max(max_so_far, max_this_time)
                # print(over_threshold.sum(), max_so_far)
                # omega[over_threshold] = np.sign(omega[over_threshold]) * RK4ILIP_THRESHHOLD/dt
                # if np.isinf(omega).any() or np.isnan(omega).any():
                #     import IPython
                #     IPython.embed()
                #     return

                # if not i % 10:
                #     import matplotlib.image
                #     import matplotlib.cm as cm

                #     omega_plot = omega.transpose(0, 2, 1, 3, 4, 5).reshape((self.n_elements_x*self.Nx, self.n_elements_y*self.Ny))
                #     matplotlib.image.imsave('evolution/omega_%04d.png' % i, np.abs(omega_plot), origin='lower', cmap = cm.gray)

                #     arg_plot = np.angle(psi).transpose(0, 2, 1, 3, 4, 5).reshape((self.n_elements_x*self.Nx, self.n_elements_y*self.Ny))
                #     matplotlib.image.imsave('evolution/angle_%04d.png' % i, arg_plot, origin='lower', cmap = cm.gist_rainbow)

                U_half = np.exp(1j*omega*0.5*dt)
                U_full = U_half**2
                U_dagger_half = 1/U_half
                U_dagger_full = 1/U_full

                # Now do rk4 in this interaction picture
                k1 = f1 + 1j*omega*psi

                # k1 = (1j*f1/psi).imag*psi

                phi_1 = psi + 0.5*k1*dt
                psi_1 = U_dagger_half*phi_1
                f2 = dpsi_dt(t + 0.5*dt, psi_1)
                k2 = U_half*f2 + 1j*omega*phi_1

                phi_2 = psi + 0.5*k2*dt
                psi_2 = U_dagger_half*phi_2

                f3 = dpsi_dt(t + 0.5*dt, psi_2)
                k3 = U_half*f3 + 1j*omega*phi_2

                phi_3 = psi + k3*dt
                psi_3 = U_dagger_full*phi_3
                f4 = dpsi_dt(t + dt, psi_3)
                k4 = U_full*f4 + 1j*omega*phi_3

                # where = (0,28,2,6,0,0)
                # print(k1[where])
                # print(k2[where])
                # print(k3[where])
                # print(k4[where])
                # print()
            else:
                msg = "Invalid method, must be one of 'rk4' or 'rk4ilip'"
                raise ValueError(msg)

            # Ensure edgepoints are numerically identical:
            psi[RIGHT_INTERIOR_EDGEPOINTS] = psi[LEFT_INTERIOR_EDGEPOINTS]
            psi[TOP_INTERIOR_EDGEPOINTS] = psi[BOTTOM_INTERIOR_EDGEPOINTS]
            self.MPI_receive_border_values(psi)

            if method == 'rk4':
                psi[:] += dt/6*(k1 + 2*k2 + 2*k3 + k4)
            elif method == 'rk4ilip':
                phi_4 = psi + dt/6*(k1 + 2*k2 + 2*k3 + k4)
                psi_rk4ilip = U_dagger_full*phi_4

                # Compare this step to RK4:
                k1_rk4 = dpsi_dt(t, psi)
                k2_rk4 = dpsi_dt(t + 0.5*dt, psi + 0.5*k1_rk4*dt)
                k3_rk4 = dpsi_dt(t + 0.5*dt, psi + 0.5*k2_rk4*dt)
                k4_rk4 = dpsi_dt(t + dt, psi + k3_rk4*dt)
                psi_rk4 = psi + dt/6*(k1_rk4 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4)

                ratio = psi_rk4ilip/psi_rk4
                phasediff = np.angle(ratio)
                fractional_mod = np.abs(1 - np.abs(ratio))

                phasediff = phasediff.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))
                fractional_mod = fractional_mod.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))

                import matplotlib.image
                import matplotlib.cm as cm

                global biggest_phasediff
                try:
                    biggest_phasediff
                except NameError:
                    biggest_phasediff = 0
                biggest_phasediff = max(biggest_phasediff, abs(phasediff.min()))
                biggest_phasediff = max(biggest_phasediff, abs(phasediff.max()))
                print(i, biggest_phasediff)
                if i == 1039:
                    import IPython
                    IPython.embed()
                    x = (28, 31, 6, 3)
                # matplotlib.image.imsave('evolution/phasediff_%04d.png' % i, phasediff,
                #                         origin='lower', vmin=-np.pi, vmax=np.pi, cmap = cm.hsv)
                # matplotlib.image.imsave('evolution/mod_diff_%04d.png' % i, fractional_mod,
                #                         0, vmax=1, origin='lower', cmap = cm.gray)
                psi[:] = psi_rk4ilip

            else:
                msg = "Invalid method, must be one of 'rk4' or 'rk4ilip'"
                raise ValueError(msg)
            if imaginary_time:
                self.normalise(psi, n_initial)

            t += dt
            if not i % output_interval:
                do_output()
            i += 1

        # t_final reached:
        if (i - 1) % output_interval:
            do_output()
        if output_group is not None:
            group = self.output_file['output'][output_group]
            group.attrs['completion_time'] = time.time()
            group.attrs['run_time'] = group.attrs['completion_time'] - group.attrs['start_time']
        return psi

    def output(self, output_group, psi, output_log, wavefunction_output):
        group = self.output_file['output'][output_group]
        output_log_dataset = group['output_log']
        output_row = self.output_row.setdefault(output_group, 0)
        output_log_dataset.resize((output_row + 1,))
        output_log_dataset[output_row] = output_log
        if wavefunction_output:
            psi_dataset = group['psi']
            psi_dataset.resize((output_row + 1,) + psi_dataset.shape[1:])
            if self.single_output_file:
                start_x = self.global_first_x_element
                end_x = start_x + self.n_elements_x
                start_y = self.global_first_y_element
                end_y = start_y + self.n_elements_y
                psi_dataset[output_row, start_x:end_x, start_y:end_y] = psi
            else:
                psi_dataset[output_row] = psi
        self.output_row[output_group] += 1
