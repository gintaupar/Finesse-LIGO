"""
Simple substrate
"""
from finesse.element import ModelElement
from finesse.components import Nothing, Space
from finesse.parameter import float_parameter
from finesse.freeze import Freezable


@float_parameter("L", "Length", units="m")
@float_parameter("nr", "Index of refraction")
class Substrate(ModelElement):
    def __init__(self, name, L=0, nr=1):
        super().__init__(name)
        self.L = L
        self.nr = nr
        self.components = Freezable()

    @property
    def p1(self):
        return getattr(self.components, f"{self.name}_fr").p1

    @property
    def p2(self):
        return getattr(self.components, f"{self.name}_bk").p2

    def _on_add(self, model):
        fr = Nothing(f"{self.name}_fr")
        bk = Nothing(f"{self.name}_bk")
        fr._namespace = (f".{self.name}.components", )
        bk._namespace = (f".{self.name}.components", )
        model.add(fr, unremovable=True)
        model.add(bk, unremovable=True)
        substrate = Space(
            f"{self.name}_substrate", fr.p2, bk.p1, L=self.L.ref, nr=self.nr.ref,
        )
        substrate._namespace = (".spaces", f".{self.name}.components")
        model.add(substrate, unremovable=True)
