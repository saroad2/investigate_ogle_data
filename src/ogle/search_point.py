from dataclasses import asdict, dataclass, replace


@dataclass
class SearchPoint:

    u_min: float = 0
    t0: float = 0
    tau: float = 0
    f_bl: float = 0

    def __iadd__(self, other: "SearchPoint"):
        self.u_min += other.u_min
        self.t0 += other.t0
        self.tau += other.tau
        self.f_bl += other.f_bl
        return self

    def __imul__(self, other: float):
        self.u_min *= other
        self.t0 *= other
        self.tau *= other
        self.f_bl *= other
        return self

    def __add__(self, other: "SearchPoint"):
        new_point = self.copy()
        new_point += other
        return new_point

    def __mul__(self, other: float):
        new_point = self.copy()
        new_point *= other
        return new_point

    def __rmul__(self, other: float):
        return self * other

    def move(self, **kwargs) -> "SearchPoint":
        return self + SearchPoint(**kwargs)

    def replace(self, **kwargs) -> "SearchPoint":
        return replace(self, **kwargs)

    def copy(self) -> "SearchPoint":
        return SearchPoint(u_min=self.u_min, t0=self.t0, tau=self.tau, f_bl=self.f_bl)

    def asdict(self):
        return asdict(self)
