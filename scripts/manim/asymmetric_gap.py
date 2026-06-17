"""Manim hero animation: the asymmetric composition gap.

Merging redistributes capacity from majority taxa (passerines, -11.8%) to
under-represented taxa (marine mammals +3.9%, amphibians +1.9%).

Render:
    manim -qh --format=mp4 scripts/manim/asymmetric_gap.py AsymmetricGap
Copy output to docs/static/videos/asymmetric_gap.mp4
"""
from manim import *

GREEN = "#2e8b57"
RED = "#c0392b"


class AsymmetricGap(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        title = Text("Composition redistributes capacity to minority taxa",
                     font="sans-serif", color=BLACK).scale(0.5).to_edge(UP)
        self.play(FadeIn(title))

        groups = [("Passerines", -11.8, "majority"),
                  ("Non-passerines", -3.2, "majority"),
                  ("Raptors/water", -1.0, "mid"),
                  ("Amphibians", +1.9, "minority"),
                  ("Marine mammals", +3.9, "minority")]

        axis = Line(LEFT * 5, RIGHT * 5, color=GREY_D).shift(DOWN * 0.2)
        zero_lbl = Text("joint baseline", font="sans-serif", color=GREY_D).scale(0.3)
        zero_lbl.next_to(axis, RIGHT)
        self.play(Create(axis), FadeIn(zero_lbl))

        bars = VGroup()
        n = len(groups)
        for i, (name, delta, _) in enumerate(groups):
            x = -4.2 + i * 2.1
            h = delta * 0.18
            color = GREEN if delta >= 0 else RED
            bar = Rectangle(width=0.9, height=abs(h), color=color, fill_color=color,
                            fill_opacity=0.85, stroke_opacity=0)
            bar.move_to([x, -0.2 + h / 2, 0])
            val = Text(f"{delta:+.1f}%", font="sans-serif", color=color).scale(0.34)
            val.next_to(bar, UP if delta >= 0 else DOWN, buff=0.1)
            lbl = Text(name, font="sans-serif", color=BLACK).scale(0.26)
            lbl.move_to([x, -2.4, 0])
            bars.add(VGroup(bar, val, lbl))

        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN if g[1] >= 0 else UP)
                                for b, g in zip(bars, groups)], lag_ratio=0.18))
        self.play(*[FadeIn(b[1]) for b in bars], *[FadeIn(b[2]) for b in bars])
        self.wait(1.5)
