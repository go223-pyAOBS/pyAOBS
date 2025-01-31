"""
Basic model classes for seismic velocity modeling.

Classes:
    Point2d: A 2D point representation
    ZNode2d: A velocity node in the ZELT model
    TrapezoidCell2d: A trapezoid cell with velocity interpolation
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Point2d:
    """A 2D point representation.
    
    Attributes:
        x (float): X-coordinate
        z (float): Z-coordinate (depth)
    """
    x: float
    z: float

@dataclass
class ZNode2d:
    """A velocity node in the ZELT model.
    
    This class represents a node in the velocity model, containing x-coordinates,
    corresponding values (velocity or depth), and flags for inversion control.
    
    Attributes:
        x (List[float]): X-coordinates of the node points
        val (List[float]): Values (velocity or depth) at each x-coordinate
        flags (List[int]): Inversion flags for each point (0: fixed, 1: free)
    """
    x: List[float] = field(default_factory=list)
    val: List[float] = field(default_factory=list)
    flags: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate node data after initialization."""
        if len(self.x) != len(self.val) or len(self.x) != len(self.flags):
            raise ValueError("Mismatched lengths in node data")
            
    def add_point(self, x: float, val: float, flag: int = 0) -> None:
        """Add a new point to the node.
        
        Args:
            x (float): X-coordinate
            val (float): Value at the point (velocity or depth)
            flag (int, optional): Inversion flag. Defaults to 0 (fixed).
        """
        self.x.append(x)
        self.val.append(val)
        self.flags.append(flag)
        
    def get_value_at(self, x: float) -> float:
        """Get interpolated value at given x-coordinate.
        
        Args:
            x (float): X-coordinate to interpolate at
            
        Returns:
            float: Interpolated value
            
        Raises:
            ValueError: If x is outside the node's range
        """
        if not self.x:
            raise ValueError("Node has no points")
            
        if x <= self.x[0]:
            return self.val[0]
        if x >= self.x[-1]:
            return self.val[-1]
            
        # Linear interpolation
        for i in range(len(self.x) - 1):
            if self.x[i] <= x <= self.x[i + 1]:
                ratio = (x - self.x[i]) / (self.x[i + 1] - self.x[i])
                return self.val[i] + ratio * (self.val[i + 1] - self.val[i])
                
        raise ValueError(f"Failed to interpolate at x={x}")
        
    def get_x_coords(self) -> List[float]:
        """Get list of x-coordinates.
        
        Returns:
            List[float]: List of x-coordinates
        """
        return self.x.copy()
        
    def get_values(self) -> List[float]:
        """Get list of values.
        
        Returns:
            List[float]: List of values (velocity or depth)
        """
        return self.val.copy()

class TrapezoidCell2d:
    """A trapezoid cell in the velocity model.
    
    This class represents a trapezoid-shaped cell in the velocity model,
    defined by four corners with their respective velocities. The cell
    provides methods for point-in-cell testing and velocity interpolation.
    
    Attributes:
        x1 (float): Left x-coordinate
        x2 (float): Right x-coordinate
        s1 (float): Upper boundary slope
        b1 (float): Upper boundary intercept
        s2 (float): Lower boundary slope
        b2 (float): Lower boundary intercept
    """
    
    def __init__(self, x1: float, x2: float, 
                 z1: float, z2: float, z3: float, z4: float,
                 v1: float, v2: float, v3: float, v4: float):
        """Initialize a trapezoid cell.
        
        Args:
            x1 (float): Left x-coordinate
            x2 (float): Right x-coordinate
            z1 (float): Upper left z-coordinate
            z2 (float): Upper right z-coordinate
            z3 (float): Lower left z-coordinate
            z4 (float): Lower right z-coordinate
            v1 (float): Upper left velocity
            v2 (float): Upper right velocity
            v3 (float): Lower left velocity
            v4 (float): Lower right velocity
            
        Raises:
            ValueError: If x1 equals x2 (zero width cell)
        """
        if abs(x1 - x2) < 1e-10:
            raise ValueError("Cell width too small")
            
        self.x1, self.x2 = x1, x2
        self.v1, self.v2 = v1, v2
        self.v3, self.v4 = v3, v4

        rdx = 1.0/(x1-x2)
        # Calculate boundary line parameters
        self.s1 = (z1-z2)*rdx  # Upper boundary slope
        self.b1 = z1-self.s1*x1  # Upper boundary intercept
        self.s2 = (z3-z4)*rdx  # Lower boundary slope
        self.b2 = z3-self.s2*x1  # Lower boundary intercept

        # Check if boundaries are parallel (within numerical precision)
        self.is_parallel = abs(self.s2 - self.s1) < 1e-10 and abs(self.b2 - self.b1) < 1e-10

        if not self.is_parallel:
            # Precompute interpolation coefficients for non-parallel case
            self.c1 = self.s2*(x2*v1-x1*v2)+self.b2*(v2-v1)-self.s1*(x2*v3-x1*v4)-self.b1*(v4-v3)
            self.c2 = self.s2*(v2-v1)-self.s1*(v4-v3)
            self.c3 = x1*v2-x2*v1+x2*v3-x1*v4
            self.c4 = v1-v2+v4-v3
            self.c5 = self.b2*(x2*v1-x1*v2)-self.b1*(x2*v3-x1*v4)
            self.c6 = (self.s2-self.s1)*(x2-x1)
            self.c7 = (self.b2-self.b1)*(x2-x1)

    def is_in(self, p: Point2d) -> bool:
        """Check if a point is inside the cell.
        
        Args:
            p (Point2d): Point to test
            
        Returns:
            bool: True if point is inside the cell, False otherwise
        """
        if p.x < self.x1 or p.x > self.x2:
            return False
        
        z_up = self.s1 * p.x + self.b1
        z_down = self.s2 * p.x + self.b2
        if p.z < z_up or p.z > z_down:
            return False
            
        return True

    def at(self, p: Point2d) -> float:
        """Calculate velocity at a point within the cell.
        
        Args:
            p (Point2d): Point to calculate velocity at
            
        Returns:
            float: Interpolated velocity value
            
        Raises:
            ValueError: If point is outside the cell
        """
        if not self.is_in(p):
            raise ValueError(f"Point ({p.x}, {p.z}) is outside the cell")
            
        if self.is_parallel:
            # For parallel boundaries, use simple bilinear interpolation
            rx = (p.x - self.x1) / (self.x2 - self.x1)
            # Calculate z position relative to boundaries
            z_up = self.s1 * p.x + self.b1
            z_down = self.s2 * p.x + self.b2
            rz = (p.z - z_up) / (z_down - z_up) if abs(z_down - z_up) > 1e-10 else 0.5
            
            # Interpolate velocities
            v_top = self.v1 + rx * (self.v2 - self.v1)
            v_bottom = self.v3 + rx * (self.v4 - self.v3)
            return v_top + rz * (v_bottom - v_top)
        else:
            # Use original formula for non-parallel case
            val1 = (self.c1 + self.c2*p.x)*p.x + (self.c3 + self.c4*p.x)*p.z + self.c5
            val2 = self.c6*p.x + self.c7
            if abs(val2) < 1e-10:
                # If val2 is still close to zero, use average of corner velocities
                return (self.v1 + self.v2 + self.v3 + self.v4) / 4.0
            return val1/val2
        
    def __repr__(self) -> str:
        """Return string representation of the cell."""
        return (f"TrapezoidCell2d(x=[{self.x1}, {self.x2}], "
                f"v=[{self.v1}, {self.v2}, {self.v3}, {self.v4}])")

__all__ = ['Point2d', 'ZNode2d', 'TrapezoidCell2d'] 