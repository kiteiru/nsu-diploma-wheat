from nn_tools.metrics.conf import SeedDetectionConfusions

def get_spikelet_centers(mask):
    """
        NOTE:
            assumed that 
                1. mask is binary
    """

    coords = SeedDetectionConfusions._get_central_points(mask)

    return coords



