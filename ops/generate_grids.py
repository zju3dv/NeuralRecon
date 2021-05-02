import torch


def generate_grid(n_vox, interval):
    with torch.no_grad():
        # Create voxel grid
        grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
        grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
        grid = grid.unsqueeze(0).cuda().float()  # 1 3 dx dy dz
        grid = grid.view(1, 3, -1)
    return grid
