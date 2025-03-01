from manim import *;

#To Run (in terminal):
# "manim -pql manimations.py demo"
class demo(Scene):
    def construct(self):
        t = Text("Hello!").shift(UP)
        t2 = Tex("Hi!").shift(DOWN)
        self.play(Write(t), Write(t2))
        self.wait(3)
        self.play(FadeOut(t), FadeOut(t2))
        self.wait(1)