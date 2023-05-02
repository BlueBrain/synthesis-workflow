from neurocollage.mesh_helper import MeshHelper
from voxcell.cell_collection import CellCollection
import numpy as np
import trimesh


def get_distances_to_mesh(mesh_helper, mesh, ray_origins, ray_directions):
    """Compute distances from point/directions to a mesh."""
    vox_ray_origins = mesh_helper.positions_to_indices(ray_origins)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=vox_ray_origins, ray_directions=ray_directions
    )
    return np.linalg.norm(
        mesh_helper.indices_to_positions(locations[index_ray]) - ray_origins, axis=1
    )


if __name__ == "__main__":
    atlas = {"atlas": "atlas", "structure": "out/synthesis_input/region_structure.yaml"}
    mesh_helper = MeshHelper(atlas, None)
    pia_mesh = mesh_helper.get_pia_mesh()
    pia_mesh = mesh_helper.get_boundary_mesh()
    #pia_mesh = mesh_helper.get_layer_meshes()[1]
    print(pia_mesh)
    pia_mesh.export("mesh.obj")

    morph_df = CellCollection().load("out/synthesis/circuit.h5").as_dataframe()

    ray_origins = morph_df[["x", "y", "z"]].to_numpy()
    ray_directions = np.array(len(morph_df) * [[0.0, 1, 0.0]])
    dists = get_distances_to_mesh(mesh_helper, pia_mesh, ray_origins, ray_directions)
    print(dists)
    ray_visualize = trimesh.load_path(
        np.hstack((ray_origins, ray_origins + ray_directions * 500.0)).reshape(-1, 2, 3)
    )

    scene = trimesh.Scene([pia_mesh, ray_visualize])
    scene.show()
    mesh_helper.show()
