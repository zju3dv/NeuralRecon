import pyrender
import trimesh

viewer_flags = {
    "lighting_intensity": 4.0,
    "window_title": "NeuralRecon reconstructions",
}
class Visualizer:
    def __init__(self):
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 0.], bg_color=[1.0, 1.0, 1.0, 0.])
        self.viewer = pyrender.Viewer(self.scene, viewport_size=(1400, 900), 
                                      run_in_thread=True, 
                                      viewer_flags = viewer_flags,
                                      use_raymond_lighting=True)
        self.mesh_node = None
        self.material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.5,
                        roughnessFactor=0.8,
                        alphaMode='OPAQUE',
                        baseColorFactor=(.6, .6, .6, 1.))

    def vis_mesh(self, mesh: trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=self.material)

        self.viewer.render_lock.acquire()
        self.scene.add(mesh)
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.viewer.render_lock.release()
        self.mesh_node = list(self.scene.mesh_nodes)[0]
    
    def close(self):
        self.viewer.close_external()