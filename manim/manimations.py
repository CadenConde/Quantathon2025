from manim import *

class ThreePlanesScene(ThreeDScene):
    def construct(self):
        # Set up the 3D axes
        axes = ThreeDAxes()

        # Define three planes using ParametricSurface
        plane1 = ParametricSurface(
            lambda u, v: np.array([u, v, 1]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(20, 20),
            checkerboard_colors=[BLUE, BLUE_E],
            fill_opacity=0.5
        )

        plane2 = ParametricSurface(
            lambda u, v: np.array([u, v, -u - v]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(20, 20),
            checkerboard_colors=[RED, RED_E],
            fill_opacity=0.5
        )

        plane3 = ParametricSurface(
            lambda u, v: np.array([u, v, 0.5 * u - v]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(20, 20),
            checkerboard_colors=[GREEN, GREEN_E],
            fill_opacity=0.5
        )

        # Set the camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Add objects to scene
        self.add(axes, plane1, plane2, plane3)

        # Hold the scene
        self.wait(3)

