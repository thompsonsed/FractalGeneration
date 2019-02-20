"""
Generates images of fractals.

Created with reference to
https://www.ibm.com/developerworks/community/blogs/jfp/entry/How_To_Compute_Mandelbrodt_Set_Quickly and
https://github.com/hzy46/tensorflow-fractal-playground
"""

import numpy as np
import tensorflow as tf
from PIL import Image


class FractalGenerator:
    """Contains the operations for generating a mandelbrot set."""

    def __init__(self):
        self.grid = None
        self.threshold = None
        self.end_z = None
        self.end_step = None
        self.background = (0, 0, 0)

    def get_colour(self, r1, r2, r3, b1, b2, b3):
        """
        Generate a function which converts a z and step value to an RGB colour, based on three pre-defined ratios.

        :param float r1: the first rgb ratio
        :param float r2: the second rgb ratio
        :param float r3: the third rgb ratio
        :param float b1: the first base colour
        :param float b2: the second base colour
        :param float b3: the third base colour

        :rtype: func
        :return: a function which returns a tuple of the desired rgb values
        """

        def colour(z, i):
            """
            Gets the colour of a z and step value.

            :param complex z: the z value from the mandelbrot set
            :param int i: the step value

            :rtype: tuple
            :return: the three RGB colours
            """
            if abs(z) < self.threshold:
                return self.background
            v = np.log2(i + self.threshold - np.log2(np.log2(abs(z)))) / self.threshold
            if v < 1.0:
                return v ** b1, v ** b2, v ** b3 # background
            else:
                v = max(0, 2 - v)
                return v ** r1, v ** r2, v ** r3  # main tones
        return colour

    def get_inner_colour(self, r1, r2, r3, b1, b2, b3):
        """
        Generate a function which converts a z and step value to an RGB colour, based on three pre-defined ratios.

        :param float r1: the first rgb ratio
        :param float r2: the second rgb ratio
        :param float r3: the third rgb ratio
        :param float b1: the first base colour
        :param float b2: the second base colour
        :param float b3: the third base colour

        :rtype: func
        :return: a function which returns a list of the desired rgb values
        """

        def colour(z, i):
            """
            Gets the colour of a z and step value.

            :param z: the z value from the mandelbrot set
            :param i: the step value

            :rtype: list
            :return: list containing the RGB colours
            """
            if abs(z) < self.threshold:
                return 0, 0, 0
            v = np.log2(i + self.threshold - np.log2(np.log2(abs(z)))) / self.threshold
            if v < 1.0:
                return v ** b1, v ** b2, v ** b3 # coloured tones
            else:
                v = max(0, 2 - v)
                return v ** r1, v ** r2, v ** r3  # sepia tones
        return colour

    def set_grid(self, start_x, end_x, start_y, end_y, resolution_x, resolution_y, threshold):
        """
        Defines the size of the grid on which to draw the mandelbrot set.

        :param start_x: the starting x position
        :param end_x: the end x position
        :param start_y: the start y position
        :param end_y: the end y position
        :param resolution_x: the resolution of each pixel in the x dimension
        :param resolution_y: the resolution of each pixel in the y dimension
        :param threshold: sets the threshold for colouring the fractal

        :rtype: None
        """
        step_x = (end_x - start_x) / resolution_x
        step_y = (end_y - start_y) / resolution_y
        real, complex = np.mgrid[start_y:end_y:step_y, start_x:end_x:step_x]
        self.grid = real + complex * 1j
        self.threshold = threshold

    def get_coloured_grid(self, r1, r2, r3, b1=4, b2=2.5, b3=1):
        """
        Colours the grid and returns the image.

        :param r1: the first colour ratio
        :param r2: the second colour ratio
        :param r3: the third colour ratio
        :param b1: the first base colour
        :param b2: the second base colour
        :param b2: the third base colour

        :rtype: PIL.Image
        :return: a coloured image of the set
        """
        r, g, b = np.frompyfunc(self.get_colour(r1, r2, r3, b1, b2, b3), 2, 3)(self.end_z, self.end_step)
        img_array = np.dstack((r, g, b))
        return Image.fromarray(np.uint8(img_array * 255))

    def generate_mandelbrot(self, iterations):
        """
        Generates the mandelbrot set from the set parameters for a particular number of iterations.

        :param int iterations: number of iterations to complete for

        :rtype: None
        """
        if self.grid is None:
            raise RuntimeError("Grid hasn't been setup - call set_grid first.")
        # Define the tensorflow variables
        c = tf.constant(self.grid.astype(np.complex64))
        z = tf.Variable(c)
        n = tf.Variable(tf.zeros_like(c, tf.float32))
        # Start the tensorflow session
        with tf.Session():
            tf.global_variables_initializer().run()
            # Define the main mandelbrot algorithm - either take the square plus x, or keep z
            z_out = tf.where(tf.abs(z) < self.threshold, z ** 2 + c, z)
            not_diverged = tf.abs(z_out) < self.threshold
            # Create a group of tensorflow operations
            step = tf.group(
                z.assign(z_out),
                n.assign_add(tf.cast(not_diverged, tf.float32))
            )
            # Run the operations for a set number of steps
            for i in range(iterations):
                step.run()
            self.end_step = n.eval()
            self.end_z = z_out.eval()

    def generate_julia(self, iterations, c):
        """
        Generates the julia set from the set parameters for a number of iterations and using a particular value of c.

        :param int iterations: the number of iterations to continue for
        :param complex c: the c parameter

        :rtype: None
        """
        if self.grid is None:
            raise RuntimeError("Grid hasn't been setup - call set_grid first.")
        # Define the tensorflow variables
        c_val = tf.constant(np.full(shape=self.grid.shape, fill_value=c, dtype=self.grid.dtype))
        z = tf.Variable(self.grid)
        n = tf.Variable(tf.zeros_like(c_val, tf.float32))
        # Start the tensorflow session
        with tf.Session():
            tf.global_variables_initializer().run()
            # Define the main julia algorithm - either take the square plus x, or keep z

            z_out = tf.where(tf.abs(z) < self.threshold, z ** 2 + c_val, z)
            not_diverged = tf.abs(z_out) < self.threshold
            step = tf.group(
                z.assign(z_out),
                n.assign_add(tf.cast(not_diverged, tf.float32))
            )

            for i in range(iterations):
                step.run()
            self.end_step = n.eval()
            self.end_z = z_out.eval()