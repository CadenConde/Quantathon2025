from manim import *

"""
use this command to run the manim file in terminal
manim -pql manim/manimations.py ThreePlanesScene
"""

class ParametricSurfaceExample(ThreeDScene):
    def construct(self):
        # Create 3D axes
        axes = ThreeDAxes()

        # Define the parametric function for the surface
        def parametric_function(u, v):
            x = u
            y = v
            z = np.sin(x) * np.cos(y)
            return np.array([x, y, z])

        # Create the surface
        surface = Surface(
            parametric_function,
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(20, 20),
        )
        surface.set_style(fill_opacity=0.5, fill_color=BLUE)

        # Set the camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Add the axes and surface to the scene
        self.add(axes, surface)
        self.wait(2)