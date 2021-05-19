import pyrender
import trimesh

class Viser:
    def __init__(self):
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 0.], bg_color=[1.0, 1.0, 1.0, 0.])
        self.Viewer = pyrender.Viewer(self.scene, viewport_size=(1024, 1024), run_in_thread=True, use_raymond_lighting=True)
        self.mesh_Node = None

    def vis_mesh(self, mesh: trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        self.Viewer.render_lock.acquire()
        self.scene.add(mesh)
        if self.mesh_Node is not None:
            self.scene.remove_node(self.mesh_Node)
        self.Viewer.render_lock.release()
        self.mesh_Node = list(self.scene.mesh_nodes)[0]


