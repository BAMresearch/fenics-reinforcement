"""Dof-index mapping between a mesh and a lower-dimensional submesh of it.

Adapted from the SubSpaceMap in BAMresearch/fenics-constitutive
(src/fenics_constitutive/solver/maps.py). That implementation maps between a
parent mesh and a same-dimension (co-dimension 0) submesh, with the
correspondence built from per-cell dofmap entries -- e.g. splitting a
material subdomain out of a larger mesh. Here the submesh has a *lower*
topological dimension than its parent (reinforcement lines embedded in a 3D
concrete mesh), so the correspondence instead comes directly from the vertex
map returned by dolfinx.mesh.create_submesh, which is exact (no
floating-point coordinate matching needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import dolfinx as dfx
import numpy as np

__all__ = ["SubSpaceMap", "build_vertex_subspace_map"]


@dataclass
class SubSpaceMap:
    """
    Maps dofs between a function space on a submesh and a function space on
    its parent mesh.

    Args:
        parent: Parent-space dof indices (scalar/unblocked numbering).
        sub: Sub-space dof indices (scalar/unblocked numbering),
            corresponding entry-by-entry to `parent`.

    """

    parent: np.ndarray
    sub: np.ndarray

    def map_to_parent(self, sub: dfx.fem.Function, parent: dfx.fem.Function) -> None:
        """
        Copy values from a Function on the submesh into a Function on the
        parent mesh.

        Args:
            sub: The function on the submesh.
            parent: The function on the parent mesh.
        """
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"
        bs = parent.function_space.dofmap.index_map_bs
        parent_array = parent.x.array.reshape(-1, bs)
        sub_array = sub.x.array.reshape(-1, bs)
        parent_array[self.parent] = sub_array[self.sub]
        parent.x.scatter_forward()

    def map_to_sub(self, parent: dfx.fem.Function, sub: dfx.fem.Function) -> None:
        """
        Copy values from a Function on the parent mesh into a Function on
        the submesh.

        Args:
            parent: The function on the parent mesh.
            sub: The function on the submesh.
        """
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"
        bs = parent.function_space.dofmap.index_map_bs
        parent_array = parent.x.array.reshape(-1, bs)
        sub_array = sub.x.array.reshape(-1, bs)
        sub_array[self.sub] = parent_array[self.parent]
        sub.x.scatter_forward()


def build_vertex_subspace_map(
    vertex_map: np.ndarray,
    parent_space: dfx.fem.FunctionSpace,
    sub_space: dfx.fem.FunctionSpace,
) -> SubSpaceMap:
    """
    Build a SubSpaceMap from the vertex map returned by
    dolfinx.mesh.create_submesh (submesh vertex index -> parent mesh vertex
    index).

    Both `parent_space` and `sub_space` must be first-order (vertex-based)
    continuous Lagrange spaces, built on the parent mesh and the submesh
    respectively. Dof indices are resolved per vertex via
    dolfinx.fem.locate_dofs_topological rather than assumed to equal the
    vertex index: dolfinx may internally reorder dofs relative to the
    mesh's own vertex numbering (e.g. for cache locality), so that
    equivalence does not hold in general, even though it may appear to for
    small/simple meshes.

    Args:
        vertex_map: As returned by dolfinx.mesh.create_submesh (3rd return
            value): submesh vertex index -> parent mesh vertex index.
        parent_space: P1 (vector) Lagrange space on the parent mesh.
        sub_space: P1 (vector) Lagrange space on the submesh.

    Returns:
        The SubSpaceMap between sub_space and parent_space.

    """
    parent_mesh = parent_space.mesh
    sub_mesh = sub_space.mesh
    parent_mesh.topology.create_connectivity(0, parent_mesh.topology.dim)
    sub_mesh.topology.create_connectivity(0, sub_mesh.topology.dim)

    num_sub_vertices = len(vertex_map)
    parent = np.empty(num_sub_vertices, dtype=np.int32)
    sub = np.empty(num_sub_vertices, dtype=np.int32)
    for sub_vertex, parent_vertex in enumerate(vertex_map):
        parent_dof = dfx.fem.locate_dofs_topological(
            parent_space, 0, np.array([parent_vertex], dtype=np.int32)
        )
        sub_dof = dfx.fem.locate_dofs_topological(
            sub_space, 0, np.array([sub_vertex], dtype=np.int32)
        )
        assert len(parent_dof) == 1 and len(sub_dof) == 1, (
            f"expected exactly 1 dof per vertex, found {len(parent_dof)} (parent) "
            f"and {len(sub_dof)} (sub) for vertex {sub_vertex}"
        )
        parent[sub_vertex] = parent_dof[0]
        sub[sub_vertex] = sub_dof[0]

    parent_coords = parent_space.tabulate_dof_coordinates()[parent]
    sub_coords = sub_space.tabulate_dof_coordinates()[sub]
    if not np.allclose(parent_coords, sub_coords):
        raise ValueError(
            "resolved parent/sub dofs do not have matching coordinates; "
            "both spaces must be first-order Lagrange spaces built directly "
            "on the parent mesh and the submesh."
        )

    return SubSpaceMap(parent=parent, sub=sub)
