# Applications Available on FastX/DAV Nodes

> **Note:** This section should be added to the FastX documentation page after the "Choosing a GPU on Kestrel" section.

The following applications can be run on Kestrel DAV nodes through FastX sessions. These applications typically require graphical user interfaces (GUI) and benefit from the hardware-accelerated OpenGL rendering provided by VirtualGL.

## Visualization and Analysis Tools

| Application | Module Command | Launch Command | Quick Start Example |
|------------|---------------|----------------|-------------------|
| **ParaView** | `module load paraview/5.11.0-gui` | `vglrun -d :0.1 paraview` | Launch GUI for interactive visualization |
| **VisIt** | `module load visit` | `visit` | Launch GUI for data visualization |

## Engineering and Simulation Software

| Application | Module Command | Launch Command | Notes |
|------------|---------------|----------------|-------|
| **MATLAB** | `module load matlab` | `vglrun matlab` | For interactive GUI usage |
| **Ansys Workbench** | `module load ansys/<version>` | `vglrun runwb2` | For building models and meshes |
| **COMSOL** | `module load comsol` | `vglrun comsol` | For building and testing models |
| **Chemkin (Ansys)** | `module load ansys` | `run_rdworkbench.sh` | Chemkin Reaction Workbench GUI |
| **M-Star CFD** | `module load mstar` | N/A | Requires compute node with X-forwarding. See [M-Star documentation](../Applications/LBMcfd.md) for setup instructions |

## Development and Profiling Tools

| Application | Module Command | Launch Command | Notes |
|------------|---------------|----------------|-------|
| **Linaro Forge (MAP)** | Follow [Linaro Forge instructions](../Development/Performance_Tools/Linaro-Forge/map.md) | `map` | Performance profiling tool |

## Important Notes

* **Resource Limits**: DAV nodes are shared resources. CPU and RAM usage is monitored, and intensive operations should be moved to dedicated compute nodes.
* **VirtualGL Usage**: Applications requiring hardware-accelerated OpenGL rendering should be launched with `vglrun` to utilize GPU acceleration.
* **GPU Selection**: Use the `-d :0.0` or `-d :0.1` flag with `vglrun` to select between the two available NVIDIA A40 GPUs.
* **Compute-Intensive Work**: For resource-intensive operations, request a compute node allocation and connect via `ssh -X <nodename>` from within your FastX session. See the [interactive jobs documentation](../Slurm/interactive_jobs.md) for more information.
* **Module Availability**: Use `module avail` to see all available modules on the DAV nodes. Module versions may vary; use `module avail <application>` to see available versions.

For detailed usage instructions for each application, please refer to their respective documentation pages linked in the table above or in the [Applications section](../Applications/) of the documentation.
