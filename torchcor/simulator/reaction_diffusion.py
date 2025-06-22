import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torchcor as tc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
import time
from torchcor.core import *
from pathlib import Path
import pandas as pd
from torchcor.signalanalysis.signalanalysis.ecg_QS import Ecg
from torchcor.tools.igbwriter import IGBWriter


class reaction_diffusion:
    def __init__(self, reaction_term, T, dt, dm, diff_f, diff_t, device=tc.get_device(), dtype=None):
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.T = T  # ms
        self.dt = dt  # ms
        self.nt = int(T / dt)
        self.dm = dm # 2D or 3D

        self.reaction_term = reaction_term

        self.pcd = None
        self.cg = None

        self.n_nodes = None
        self.nodes = None
        self.elems = None
        self.fibres = None
        self.region_ids = None

        self.K = None
        self.M = None
        self.A = None
        self.theta = 0.5

        self.diff_t = diff_t
        self.diff_f = diff_f
        self.sigma_m = None

        nvmlInit()
        device_id = torch.cuda.current_device()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(device_id)

        self.mesh_path = None
        self.result_path = None


    def load_mesh(self, path="Data/ventricle/Case_1", unit_conversion=1):
        self.mesh_path = Path(path)
        
        reader = MeshReader(path)
        nodes, elems, regions, fibres = reader.read(unit_conversion=unit_conversion)
        
        self.n_nodes = nodes.shape[0]
        self.nodes = torch.from_numpy(nodes).to(dtype=self.dtype, device=self.device)
        self.elems = torch.from_numpy(elems).to(dtype=torch.int, device=self.device)
        # self.regions = torch.from_numpy(regions).to(dtype=torch.int, device=self.device)
        self.fibres = torch.from_numpy(fibres).to(dtype=self.dtype, device=self.device)


    def generate_diffusivity_tensors(self):
        sigma_m = self.diff_f * torch.eye(self.dm, device=self.fibres.device, dtype=self.dtype).unsqueeze(0).expand(self.fibres.shape[0], self.dm, self.dm)
        sigma_m += (self.diff_f - self.diff_t) * self.fibres.unsqueeze(2) @ self.fibres.unsqueeze(1)
        return sigma_m

    def assemble(self):
        self.sigma_m = self.generate_diffusivity_tensors()

        if self.elems.shape[1] == 3 and self.dm == 2:
            matrices = Matrices2D(vertices=self.nodes, triangles=self.elems, device=self.device, dtype=self.dtype)
        elif self.elems.shape[1] == 3:
            matrices = Matrices3DSurface(vertices=self.nodes, triangles=self.elems, device=self.device, dtype=self.dtype)
        else:
            matrices = Matrices3D(vertices=self.nodes, tetrahedrons=self.elems, device=self.device, dtype=self.dtype)

        raise  Exception(self.elems.shape[1] == 3, self.dm == 2, type(matrices))
        K, M = matrices.assemble_matrices(self.sigma_m)
        
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M + self.K * self.dt * self.theta
        A = A.coalesce()

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)

        self.M = self.M.to_sparse_csr()
        self.K = self.K.to_sparse_csr()
        self.A = A.to_sparse_csr()

        self.cg = ConjugateGradient(self.pcd, self.A, dtype=torch.float64)
        

    def step(self, u, t, a_tol, r_tol, max_iter, verbose=False):
        if verbose:
            torch.cuda.synchronize()

        ### reaction term ###
        du = self.reaction_term(u) / 100
        b = u + self.dt * du
        #############

        if verbose:
            torch.cuda.synchronize()

        b = self.M @ b
        b -= (1 - self.theta) * self.dt * self.K @ u

        u, n_iter = self.cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        ################
        if verbose:
            torch.cuda.synchronize()

        return u, n_iter

    def solve(self, a_tol, r_tol, max_iter, linear_guess=True, snapshot_interval=1, verbose=True, result_path=None):
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.assemble()
        
        u = self.reaction_term.initialize(self.n_nodes)
        u_initial = u.clone()
        self.cg.initialize(x=u, linear_guess=linear_guess)
        ts_per_frame = int(snapshot_interval / self.dt)

        t = 0
        solving_time = time.time()
        n_total_iter = 0
        gpu_utilisation_list = []
        gpu_memory_list = []
        solution_list = [u_initial]
        for n in range(1, self.nt + 1):
            t += self.dt
            
            ### CG step ###
            u, n_iter = self.step(u, t, a_tol, r_tol, max_iter, verbose)
            n_total_iter += n_iter

            ### keep track of GPU usage ###
            if n % ts_per_frame == 0:
                solution_list.append(u.clone())

                gpu_utilisation_list.append(nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu)
                gpu_memory_list.append(nvmlDeviceGetMemoryInfo(self.gpu_handle).used / 1e9)
                
                if verbose and snapshot_interval != self.T:
                    print(f"t: {round(t, 1)}/{self.T}", 
                          f"Time elapsed:", round(time.time() - solving_time, 2),
                          f"Total CG iter:", n_total_iter,
                          flush=True)

        if snapshot_interval < self.T:
            torch.save(torch.stack(solution_list, dim=0).cpu(), self.result_path / "Vm.pt")

        ### print log info to console ###
        if verbose:
            print(self.reaction_term.name,
                  self.n_nodes, 
                  round(time.time() - solving_time, 2),
                  n_total_iter,
                  f"{round(sum(gpu_utilisation_list)/len(gpu_utilisation_list), 2)}",
                  f"{round(sum(gpu_memory_list)/len(gpu_memory_list), 2)}",
                  flush=True)

        return n_total_iter


    def pt_to_vtk(self):
        start_time = time.time()

        if self.elems.shape[1] == 3:
            visualization = VTK3DSurface(self.nodes, self.elems)
        else:
            visualization = VTK3D(self.nodes, self.elems)
        
        solutions = torch.load(self.result_path / "Vm.pt", map_location=torch.device('cpu'))
        n_solutions = solutions.shape[0]
        for i in range(n_solutions):
            visualization.save_frame(color_values=solutions[i],
                                     frame_path=self.result_path / f"Vm_vtk/frame_{i}.vtk")
        

        ATs = torch.load(self.result_path / "ATs.pt")
        visualization.save_frame(color_values=ATs,
                                 frame_path=self.result_path / "ATs.vtk")
        RTs = torch.load(self.result_path / "RTs.pt")
        visualization.save_frame(color_values=RTs,
                                 frame_path=self.result_path / "RTs.vtk")

        print(f"Saved vtk files in {round(time.time() - start_time, 2)}", flush=True)




