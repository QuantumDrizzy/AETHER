"""Reaction-diffusion — morphogenesis: pattern self-organising in a chemical field.

Gray-Scott: two diffusing, reacting species whose feed/kill rates decide whether a
seed grows into spots, stripes or mazes. Pattern from local rules, the continuous-
chemistry member of the daemons / self-organisation line.
"""

from research.reaction_diffusion.gray_scott import PRESETS, GrayScott

__all__ = ["GrayScott", "PRESETS"]
