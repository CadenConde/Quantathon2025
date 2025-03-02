# from manim import *

# """
# use this command to run the manim file in terminal
# manim -pql manim/manimations.py ThreePlanesScene
# """

# class ParametricSurfaceExample(ThreeDScene):
#     def construct(self):
#         # Create 3D axes
#         axes = ThreeDAxes()

#         # Define the parametric function for the surface
#         def parametric_function(u, v):
#             x = u
#             y = v
#             z = np.sin(x) * np.cos(y)
#             return np.array([x, y, z])

#         # Create the surface
#         surface = Surface(
#             parametric_function,
#             u_range=[-3, 3],
#             v_range=[-3, 3],
#             resolution=(20, 20),
#         )
#         surface.set_style(fill_opacity=0.5, fill_color=BLUE)

#         # Set the camera orientation
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

#         # Add the axes and surface to the scene
#         self.add(axes, surface)
#         self.wait(2)
        
from manim import *
import pandas as pd
import numpy as np

# Load and preprocess data
df = pd.read_csv("../data/DailyStockBondFloats.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Filter data to only include the first trading day of each month
df["Month"] = df["Date"].dt.to_period("M")
df = df.groupby("Month").first().reset_index()

# Split data into training (2007-2018) and testing (2019-2022)
train_data = df[(df["Date"] >= "2007-01-01") & (df["Date"] <= "2018-12-31")]
test_data = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2022-12-31")]

class SP500Animation(Scene):
    def construct(self):
        # Create title robot
        robot = SVGMobject("robot.svg").scale(0.8).to_edge(UP)
        self.play(FadeIn(robot))

        # Generate scatter plot points
        train_points = VGroup(*[Dot(point=[x*0.05 - 5, y*0.002 - 3, 0], color=BLUE) 
                                 for x, y in enumerate(train_data["S&P500"])])
        test_points = VGroup(*[Dot(point=[x*0.05 - 5, y*0.002 - 3, 0], color=RED) 
                                for x, y in enumerate(test_data["S&P500"])])
        
        self.play(FadeIn(train_points))
        self.wait(1)
        
        # Duplicate and move training points to robot
        train_points_copy = train_points.copy()
        self.play(train_points_copy.animate.move_to(robot), FadeOut(train_points_copy))
        self.wait(1)
        
        # Generate test points from robot
        test_points.move_to(robot)
        self.play(test_points.animate.move_to(ORIGIN))
        self.wait(2)
        
        self.play(FadeIn(test_points))
        self.wait(2)
        
class Test(Scene):
    def construct(self):
        # Load dataset (only first 100 rows)
        df = pd.read_csv("../data/DailyStockBondFloats.csv", parse_dates=["Date"])
        df = df.sort_values("Date")
        df = df[(df["Date"] <= "2022-12-31")]
        df = df.iloc[::14]  # Sample one row per 30 days to reduce points

        # Convert dates to numeric indices
        df["DateIndex"] = 2007 + round(((df["Date"] - df["Date"].min()).dt.days/365),1)

        # Extract numeric indices and S&P500 values
        dates = df["DateIndex"].to_numpy()
        stocks = df["S&P500"].to_numpy()

        # Create a scatter plot using Manim
        scatter = Axes(
            x_range=[min(dates)-1, 2022, 2],  # Use days instead of datetime
            y_range=[0, max(stocks) + 50, 1000],
            axis_config={
                "include_tip": False,
                "decimal_number_config": {"num_decimal_places": 0}  # Remove commas
            },
            x_length=7,
            y_length=5,
        ).add_coordinates()

        # Labels
        x_label = scatter.get_x_axis_label("Time (Days from Start)")
        y_label = scatter.get_y_axis_label("S and P500")

        # Convert (DateIndex, S&P500) to Manim coordinates and plot dots
        scatter_dots = VGroup(*[
            Dot(scatter.c2p(date, stock), color=RED, radius=0.05) for date, stock in zip(dates, stocks)
        ])

        # Add elements to scene
        self.play(Create(scatter), FadeIn(x_label, y_label))
        self.play(LaggedStartMap(FadeIn, scatter_dots, lag_ratio=0.1))
        self.wait(2)