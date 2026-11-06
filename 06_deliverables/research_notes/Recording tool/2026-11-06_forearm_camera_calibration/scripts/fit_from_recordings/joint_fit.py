"""
Fit the forearm camera to two kinds of recorded evidence at once: the paper sheet's straight outer edges and the puck
while it does not move.

The sheet (a rectangle of unknown pose in each episode, and of one unknown size shared by all) lies on the table and
does not move while an episode is recorded. Every edge pixel found in a picture must therefore lie on the picture of
one of the rectangle's four sides; the residual is that pixel's distance, measured across the side's picture, in
pixels. The sides are seen from arm positions up to 30 cm apart, which fixes the camera's scale and pose far better
than the short runs of a still puck. Each still-puck run adds one unknown point on the puck's axis at an unknown
height above the table, the same height for all runs.

Which side an edge pixel belongs to is decided on the table with the current estimate (nearest side, outward
direction agreeing, away from the corners, within a gate); the fit is repeated with the gate narrowing. The solver
is scipy's robust least squares with a sparse finite-difference Jacobian.
"""

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy
from scipy import optimize
from scipy import sparse

import calibration_data
from forearm_camera_model import (FIT_PARAMETER_NAMES, FOREARM_CAMERA_PARAMETERS, forearm_frames_in_base, merged_camera_parameters,
                                  project_with_forearm_frames, rotation_matrix_from_vector, rotation_vector_from_matrix,
                                  undistort_normalised)

JOINT_FIT_PARAMETERS = {
    # names from FIT_PARAMETER_NAMES that the fit may change; the others keep the starting camera's value
    "free_camera_parameters": list(FIT_PARAMETER_NAMES),
    # soft pulls towards the starting camera, name -> sigma (rotations in radians, towards zero correction)
    "camera_prior_sigmas": {},
    "is_puck_height_free": True,
    "puck_point_height_m": 0.0145,
    "sheet_size_m": [0.28, 0.19],
    "is_sheet_size_free": True,
    # the residual scales: an edge pixel's distance across the side, and a puck pixel's error per axis
    "edge_pixel_sigma_px": 1.5,
    "puck_pixel_sigma_px": 2.0,
    # robust loss of scipy.optimize.least_squares and where it starts to bend, in sigmas
    "loss": "soft_l1",
    "loss_scale_sigma": 2.0,
    # the outer rounds: a side is assigned to an edge pixel within this distance on the table (one value per round)
    "edge_gates_m": [0.03, 0.015, 0.008, 0.008],
    # edge pixels this close to a corner of the rectangle are not used (the corner's own picture is rounded)
    "corner_exclusion_m": 0.01,
    # the edge pixel's outward direction, carried to the table, must agree with the side's outward normal
    "normal_agreement_deg": 40.0,
    # one picture's points on one side are weighted as if there were at most this many
    "effective_points_per_side": 15,
    "maximum_function_evaluations": 60,
    "finite_difference_steps": {
        "camera_tilt_correction_rad": 1e-6, "camera_pan_correction_rad": 1e-6, "camera_roll_correction_rad": 1e-6,
        "translation_x_m": 1e-6, "translation_y_m": 1e-6, "translation_z_m": 1e-6,
        "focal_length_px": 1e-3, "principal_point_u_px": 1e-3, "principal_point_v_px": 1e-3, "radial_distortion_k1": 1e-6
    }
}

SIDE_NAMES = ["far", "near", "left", "right"]



def merged_joint_parameters(joint_parameters: Optional[Dict] = None) -> Dict:
    """
    Return ``JOINT_FIT_PARAMETERS`` with ``joint_parameters`` merged over them.

    Parameters
    ----------
    joint_parameters : dict, optional
        Overrides.

    Returns
    -------
    dict
        A new dictionary.
    """

    parameters = copy.deepcopy(JOINT_FIT_PARAMETERS)
    if joint_parameters:
        parameters.update(copy.deepcopy(joint_parameters))
    return parameters



class EdgeObservations:

    def __init__(
            self,
            records: List[Dict],
            edge_points: Dict[Tuple[str, int], Dict],
            frame_keys: Optional[Sequence[Tuple[str, int]]] = None
    ):
        """
        Collect the edge pixels of the chosen pictures into flat arrays.

        Parameters
        ----------
        records : list of dict
            Frame records from ``calibration_data.load_frame_records``.
        edge_points : dict
            (episode, frame index) -> {"points_px": (N, 2), "outward_px": (N, 2)}.
        frame_keys : sequence of tuple, optional
            The pictures to use; all by default.
        """

        records_by_key = {(record["episode"], record["frame_index"]): record for record in records}
        keys = list(frame_keys) if frame_keys is not None else list(records_by_key)
        pixels, outwards, frame_numbers, episodes = [], [], [], []
        self.frame_keys = []
        joint_sets = []
        for key in keys:
            entry = edge_points.get(key)
            if entry is None or len(entry["points_px"]) == 0:
                continue
            frame_number = len(self.frame_keys)
            self.frame_keys.append(key)
            joint_sets.append(records_by_key[key]["joints"])
            pixels.append(entry["points_px"])
            outwards.append(entry["outward_px"])
            frame_numbers.append(numpy.full(len(entry["points_px"]), frame_number))
            episodes.extend([key[0]] * len(entry["points_px"]))
        self.pixels = numpy.vstack(pixels) if pixels else numpy.zeros((0, 2))
        self.outwards = numpy.vstack(outwards) if outwards else numpy.zeros((0, 2))
        self.frame_of_point = numpy.concatenate(frame_numbers) if frame_numbers else numpy.zeros(0, dtype = int)
        self.episode_of_point = numpy.array(episodes)
        self.frame_positions, self.frame_rotations = forearm_frames_in_base(joint_sets)
        self.point_positions = self.frame_positions[self.frame_of_point]
        self.point_rotations = self.frame_rotations[self.frame_of_point]



def camera_from_values(
        values: Dict,
        initial_rotation: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray, float, numpy.ndarray, float]:
    """
    Return the camera's rotation and translation in the forearm frame and its lens from the fit's named values.

    Parameters
    ----------
    values : dict
        Every name of ``FIT_PARAMETER_NAMES`` -> value.
    initial_rotation : numpy.ndarray
        The starting camera's rotation; the three corrections turn it about its own axes.

    Returns
    -------
    tuple
        Rotation (3, 3), translation (3,), focal length, principal point (2,) and k1.
    """

    correction = numpy.array([values["camera_tilt_correction_rad"], values["camera_pan_correction_rad"], values["camera_roll_correction_rad"]])
    rotation = initial_rotation @ rotation_matrix_from_vector(correction)
    translation = numpy.array([values["translation_x_m"], values["translation_y_m"], values["translation_z_m"]])
    principal_point = numpy.array([values["principal_point_u_px"], values["principal_point_v_px"]])
    return rotation, translation, float(values["focal_length_px"]), principal_point, float(values["radial_distortion_k1"])



def map_pixels_to_table(
        pixels: numpy.ndarray,
        forearm_positions: numpy.ndarray,
        forearm_rotations: numpy.ndarray,
        camera: Tuple
) -> numpy.ndarray:
    """
    Return where the rays through pixels meet the table (config's tilted plane), each through its own forearm frame.

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (N, 2).
    forearm_positions, forearm_rotations : numpy.ndarray
        The forearm frame of each pixel's picture, (N, 3) and (N, 3, 3).
    camera : tuple
        The output of ``camera_from_values``.

    Returns
    -------
    numpy.ndarray
        Table points (N, 3); NaN where the pixel lies beyond the lens model or the ray does not reach the table.
    """

    rotation, translation, focal, principal_point, k1 = camera
    normalised, is_valid = undistort_normalised((pixels - principal_point[None, :]) / focal, k1)
    rays_camera = numpy.column_stack([normalised, numpy.ones(len(normalised))])
    camera_rotations = numpy.einsum("nij,jk->nik", forearm_rotations, rotation)
    rays = numpy.einsum("nij,nj->ni", camera_rotations, rays_camera)
    origins = forearm_positions + numpy.einsum("nij,j->ni", forearm_rotations, translation)

    plane = calibration_data.CALIBRATION_TABLE_PLANE
    # the plane z = a x + b y + c: solve origin + s ray on it exactly
    normal = numpy.array([-plane["a"], -plane["b"], 1.0])
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distances = (plane["c"] - origins @ normal) / (rays @ normal)
    points = origins + distances[:, None] * rays
    is_hit = is_valid & numpy.isfinite(distances) & (distances > 0.0)
    points[~is_hit] = numpy.nan
    return points



def rectangle_side_geometry(
        points_xy: numpy.ndarray,
        sides: numpy.ndarray,
        centre: numpy.ndarray,
        angle: numpy.ndarray,
        half_size: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Return, for points assigned to sides of their rectangles, the foot of the perpendicular on the side, the side's
    direction, its outward normal and the signed distance (positive outside).

    Parameters
    ----------
    points_xy : numpy.ndarray
        Table points (N, 2).
    sides : numpy.ndarray
        Side of each point: 0 far (+x of the rectangle), 1 near (-x), 2 left (+y), 3 right (-y).
    centre : numpy.ndarray
        Each point's rectangle centre (N, 2).
    angle : numpy.ndarray
        Each point's rectangle angle (N,), radians from the base's x axis.
    half_size : numpy.ndarray
        Half the rectangle's length (along its x) and width (along its y), (2,).

    Returns
    -------
    tuple of numpy.ndarray
        Foot points (N, 2), side directions (N, 2), outward normals (N, 2), signed distances (N,).
    """

    cosine, sine = numpy.cos(angle), numpy.sin(angle)
    offset = points_xy - centre
    local_x = cosine * offset[:, 0] + sine * offset[:, 1]
    local_y = -sine * offset[:, 0] + cosine * offset[:, 1]

    is_x_side = sides <= 1
    sign = numpy.where((sides == 0) | (sides == 2), 1.0, -1.0)
    foot_local_x = numpy.where(is_x_side, sign * half_size[0], local_x)
    foot_local_y = numpy.where(is_x_side, local_y, sign * half_size[1])
    distance = numpy.where(is_x_side, sign * local_x - half_size[0], sign * local_y - half_size[1])
    normal_local = numpy.column_stack([numpy.where(is_x_side, sign, 0.0), numpy.where(is_x_side, 0.0, sign)])
    direction_local = numpy.column_stack([numpy.where(is_x_side, 0.0, 1.0), numpy.where(is_x_side, 1.0, 0.0)])

    def to_world(local: numpy.ndarray) -> numpy.ndarray:
        return numpy.column_stack([cosine * local[:, 0] - sine * local[:, 1], sine * local[:, 0] + cosine * local[:, 1]])

    feet = centre + to_world(numpy.column_stack([foot_local_x, foot_local_y]))
    return feet, to_world(direction_local), to_world(normal_local), distance



class JointCameraFit:

    def __init__(
            self,
            edge_observations: EdgeObservations,
            puck_groups: List[Dict],
            start_camera: Dict,
            start_sheets: Dict[str, List[float]],
            joint_parameters: Optional[Dict] = None
    ):
        """
        Prepare the joint fit.

        Parameters
        ----------
        edge_observations : EdgeObservations
            The sheet edge pixels.
        puck_groups : list of dict
            Still-puck runs: {"run_id", "joint_sets", "pixels", "initial_xy_m"}.
        start_camera : dict
            The camera to start from (camera-model parameters).
        start_sheets : dict
            Episode -> [centre x, centre y, angle] of its sheet to start from.
        joint_parameters : dict, optional
            Overrides for ``JOINT_FIT_PARAMETERS``.
        """

        self.edges = edge_observations
        self.puck_groups = list(puck_groups)
        self.start_camera = merged_camera_parameters(start_camera)
        self.episodes = sorted(start_sheets)
        self.start_sheets = {episode: [float(value) for value in start_sheets[episode]] for episode in self.episodes}

        self.__initialize_variables()
        self.update_joint_parameters(joint_parameters)
        self.__prepare()


    def __initialize_variables(self):

        self.joint_parameters = copy.deepcopy(JOINT_FIT_PARAMETERS)
        self.sides = None
        self.edge_weights = None
        self.is_edge_used = None


    def update_joint_parameters(self, joint_parameters: Optional[Dict]):

        if joint_parameters:
            self.joint_parameters.update(copy.deepcopy(joint_parameters))


    def __prepare(self):

        parameters = self.joint_parameters
        translation = self.start_camera["translation_in_forearm_m"]
        principal_point = self.start_camera["principal_point_px"]
        self.start_values = {
            "camera_tilt_correction_rad": 0.0, "camera_pan_correction_rad": 0.0, "camera_roll_correction_rad": 0.0,
            "translation_x_m": float(translation[0]), "translation_y_m": float(translation[1]), "translation_z_m": float(translation[2]),
            "focal_length_px": float(self.start_camera["focal_length_px"]),
            "principal_point_u_px": float(principal_point[0]), "principal_point_v_px": float(principal_point[1]),
            "radial_distortion_k1": float(self.start_camera["radial_distortion_k1"])
        }
        self.initial_rotation = rotation_matrix_from_vector(self.start_camera["rotation_vector_in_forearm_rad"])
        self.free_names = [name for name in FIT_PARAMETER_NAMES if name in parameters["free_camera_parameters"]]

        # the parameter vector: free camera values, puck height, sheet size, sheet poses, puck points
        self.layout = {}
        position = len(self.free_names)
        self.layout["camera"] = slice(0, position)
        self.layout["height"] = slice(position, position + (1 if parameters["is_puck_height_free"] else 0))
        position = self.layout["height"].stop
        self.layout["size"] = slice(position, position + (2 if parameters["is_sheet_size_free"] else 0))
        position = self.layout["size"].stop
        self.layout["sheets"] = slice(position, position + 3 * len(self.episodes))
        position = self.layout["sheets"].stop
        self.layout["groups"] = slice(position, position + 2 * len(self.puck_groups))
        self.parameter_count = self.layout["groups"].stop
        self.episode_index = {episode: index for index, episode in enumerate(self.episodes)}
        self.point_episode_index = numpy.array([self.episode_index.get(episode, -1) for episode in self.edges.episode_of_point], dtype = int)

        observation_joints, observation_pixels, observation_groups = [], [], []
        for group_index, group in enumerate(self.puck_groups):
            for joints, pixel in zip(group["joint_sets"], group["pixels"]):
                observation_joints.append(joints)
                observation_pixels.append(pixel)
                observation_groups.append(group_index)
        self.puck_pixels = numpy.array(observation_pixels, dtype = float).reshape(-1, 2)
        self.puck_group_of_observation = numpy.array(observation_groups, dtype = int)
        self.puck_positions, self.puck_rotations = forearm_frames_in_base(observation_joints)


    def initial_vector(self) -> numpy.ndarray:
        """Return the starting parameter vector."""

        vector = numpy.zeros(self.parameter_count)
        vector[self.layout["camera"]] = [self.start_values[name] for name in self.free_names]
        if self.joint_parameters["is_puck_height_free"]:
            vector[self.layout["height"]] = self.joint_parameters["puck_point_height_m"]
        if self.joint_parameters["is_sheet_size_free"]:
            vector[self.layout["size"]] = self.joint_parameters["sheet_size_m"]
        vector[self.layout["sheets"]] = numpy.array([self.start_sheets[episode] for episode in self.episodes]).ravel()
        if self.puck_groups:
            vector[self.layout["groups"]] = numpy.array([group["initial_xy_m"] for group in self.puck_groups]).ravel()
        return vector


    def unpack(self, vector: numpy.ndarray) -> Dict:
        """Return the named quantities of a parameter vector."""

        values = dict(self.start_values)
        for index, name in enumerate(self.free_names):
            values[name] = float(vector[index])
        height = float(vector[self.layout["height"]][0]) if self.joint_parameters["is_puck_height_free"] else self.joint_parameters["puck_point_height_m"]
        size = numpy.array(vector[self.layout["size"]]) if self.joint_parameters["is_sheet_size_free"] else numpy.array(self.joint_parameters["sheet_size_m"], dtype = float)
        sheets = numpy.array(vector[self.layout["sheets"]]).reshape(-1, 3)
        groups = numpy.array(vector[self.layout["groups"]]).reshape(-1, 2)
        return {"values": values, "puck_height_m": height, "sheet_size_m": size, "sheets": sheets, "groups": groups}


    def camera_parameters(self, vector: numpy.ndarray) -> Dict:
        """Return the camera-model parameters of a parameter vector."""

        unpacked = self.unpack(vector)
        rotation, translation, focal, principal_point, k1 = camera_from_values(unpacked["values"], self.initial_rotation)
        parameters = copy.deepcopy(self.start_camera)
        parameters.update({
            "rotation_vector_in_forearm_rad": rotation_vector_from_matrix(rotation).tolist(),
            "translation_in_forearm_m": translation.tolist(),
            "focal_length_px": focal,
            "principal_point_px": principal_point.tolist(),
            "radial_distortion_k1": k1
        })
        return parameters


    def edge_geometry(self, vector: numpy.ndarray) -> Dict:
        """
        Map every edge pixel to the table and find its nearest side, the distance to it and whether the outward
        directions agree, for assigning sides.
        """

        unpacked = self.unpack(vector)
        camera = camera_from_values(unpacked["values"], self.initial_rotation)
        points = map_pixels_to_table(self.edges.pixels, self.edges.point_positions, self.edges.point_rotations, camera)
        ahead = map_pixels_to_table(self.edges.pixels + 3.0 * self.edges.outwards, self.edges.point_positions, self.edges.point_rotations, camera)
        outward_table = ahead[:, :2] - points[:, :2]
        outward_table = outward_table / numpy.maximum(numpy.linalg.norm(outward_table, axis = 1), 1e-12)[:, None]

        valid_episode = self.point_episode_index >= 0
        sheets = unpacked["sheets"][numpy.maximum(self.point_episode_index, 0)]
        half_size = unpacked["sheet_size_m"] / 2.0
        best_distance = numpy.full(len(points), numpy.inf)
        best_side = numpy.full(len(points), -1, dtype = int)
        best_agreement = numpy.zeros(len(points))
        best_along = numpy.zeros(len(points))
        for side in range(4):
            sides = numpy.full(len(points), side)
            feet, directions, normals, distances = rectangle_side_geometry(points[:, :2], sides, sheets[:, :2], sheets[:, 2], half_size)
            is_better = numpy.abs(distances) < best_distance
            best_distance = numpy.where(is_better, numpy.abs(distances), best_distance)
            best_side = numpy.where(is_better, side, best_side)
            agreement = numpy.sum(outward_table * normals, axis = 1)
            best_agreement = numpy.where(is_better, agreement, best_agreement)
            # how far along the side from its middle, to keep away from the corners
            offset = feet - sheets[:, :2]
            along = numpy.abs(numpy.sum(offset * directions, axis = 1))
            best_along = numpy.where(is_better, along, best_along)
        side_half_lengths = numpy.where(best_side <= 1, half_size[1], half_size[0])
        is_finite = numpy.all(numpy.isfinite(points), axis = 1) & valid_episode
        return {"points": points, "side": best_side, "distance": best_distance, "agreement": best_agreement,
                "corner_clearance": side_half_lengths - best_along, "is_finite": is_finite}


    def assign_sides(
            self,
            vector: numpy.ndarray,
            gate_m: float
    ) -> None:
        """Assign each edge pixel to a side (or to none) and weight it, for the next solve."""

        geometry = self.edge_geometry(vector)
        parameters = self.joint_parameters
        self.is_edge_used = (geometry["is_finite"] & (geometry["distance"] <= gate_m)
                             & (geometry["agreement"] >= math.cos(math.radians(parameters["normal_agreement_deg"])))
                             & (geometry["corner_clearance"] >= parameters["corner_exclusion_m"]))
        self.sides = geometry["side"]

        # one picture's points on one side count as at most "effective_points_per_side" independent points
        keys = self.edges.frame_of_point * 4 + numpy.maximum(self.sides, 0)
        counts = numpy.bincount(keys[self.is_edge_used], minlength = keys.max() + 1 if len(keys) else 1)
        per_point_count = counts[keys]
        weights = numpy.ones(len(keys))
        is_crowded = per_point_count > parameters["effective_points_per_side"]
        weights[is_crowded] = numpy.sqrt(parameters["effective_points_per_side"] / per_point_count[is_crowded])
        self.edge_weights = weights


    def edge_residuals_px(self, vector: numpy.ndarray) -> numpy.ndarray:
        """
        Signed pixel distance of every used edge pixel from the picture of its side (not divided by sigma).
        """

        unpacked = self.unpack(vector)
        camera = camera_from_values(unpacked["values"], self.initial_rotation)
        used = numpy.nonzero(self.is_edge_used)[0]
        pixels = self.edges.pixels[used]
        positions = self.edges.point_positions[used]
        rotations = self.edges.point_rotations[used]
        points = map_pixels_to_table(pixels, positions, rotations, camera)
        sheets = unpacked["sheets"][self.point_episode_index[used]]
        feet, directions, _, _ = rectangle_side_geometry(numpy.nan_to_num(points[:, :2]), self.sides[used], sheets[:, :2], sheets[:, 2],
                                                         unpacked["sheet_size_m"] / 2.0)
        plane = calibration_data.CALIBRATION_TABLE_PLANE
        second = feet + 0.01 * directions
        feet_3d = numpy.column_stack([feet, plane["a"] * feet[:, 0] + plane["b"] * feet[:, 1] + plane["c"]])
        second_3d = numpy.column_stack([second, plane["a"] * second[:, 0] + plane["b"] * second[:, 1] + plane["c"]])
        rotation, translation, focal, principal_point, k1 = camera
        first_pixels, _, is_first_valid = project_with_forearm_frames(feet_3d, positions, rotations, rotation, translation, focal, principal_point, k1)
        second_pixels, _, is_second_valid = project_with_forearm_frames(second_3d, positions, rotations, rotation, translation, focal, principal_point, k1)
        along = second_pixels - first_pixels
        along_length = numpy.maximum(numpy.linalg.norm(along, axis = 1), 1e-9)
        offset = pixels - first_pixels
        distances = (along[:, 0] * offset[:, 1] - along[:, 1] * offset[:, 0]) / along_length
        is_valid = is_first_valid & is_second_valid & numpy.all(numpy.isfinite(points), axis = 1)
        distances[~is_valid] = 50.0
        return distances


    def puck_residuals_px(self, vector: numpy.ndarray) -> numpy.ndarray:
        """Predicted minus observed pixels of the still-puck observations, (M, 2)."""

        if len(self.puck_pixels) == 0:
            return numpy.zeros((0, 2))
        unpacked = self.unpack(vector)
        rotation, translation, focal, principal_point, k1 = camera_from_values(unpacked["values"], self.initial_rotation)
        plane = calibration_data.CALIBRATION_TABLE_PLANE
        group_xy = unpacked["groups"][self.puck_group_of_observation]
        heights = plane["a"] * group_xy[:, 0] + plane["b"] * group_xy[:, 1] + plane["c"] + unpacked["puck_height_m"]
        points = numpy.column_stack([group_xy, heights])
        pixels, _, is_valid = project_with_forearm_frames(points, self.puck_positions, self.puck_rotations, rotation, translation, focal, principal_point, k1)
        errors = pixels - self.puck_pixels
        errors[~is_valid] = 50.0
        return errors


    def residuals(self, vector: numpy.ndarray) -> numpy.ndarray:
        """All residuals in units of their sigma: edge pixels (weighted), puck pixels, priors."""

        parameters = self.joint_parameters
        edge = self.edge_residuals_px(vector) * self.edge_weights[self.is_edge_used] / parameters["edge_pixel_sigma_px"]
        puck = (self.puck_residuals_px(vector) / parameters["puck_pixel_sigma_px"]).ravel()
        values = self.unpack(vector)["values"]
        priors = [(values[name] - self.start_values[name]) / sigma for name, sigma in parameters["camera_prior_sigmas"].items() if name in self.free_names]
        return numpy.concatenate([edge, puck, numpy.array(priors, dtype = float)])


    def sparsity(self) -> sparse.csr_matrix:
        """Which parameters each residual depends on, built from index arrays (one row per residual)."""

        edge_count = int(self.is_edge_used.sum())
        puck_count = 2 * len(self.puck_pixels)
        prior_names = [name for name in self.joint_parameters["camera_prior_sigmas"] if name in self.free_names]
        row_count = edge_count + puck_count + len(prior_names)
        camera_columns = numpy.arange(self.layout["camera"].start, self.layout["camera"].stop)
        size_columns = numpy.arange(self.layout["size"].start, self.layout["size"].stop)
        height_columns = numpy.arange(self.layout["height"].start, self.layout["height"].stop)
        rows, columns = [], []

        edge_rows = numpy.arange(edge_count)
        shared = numpy.concatenate([camera_columns, size_columns])
        if len(shared):
            rows.append(numpy.repeat(edge_rows, len(shared)))
            columns.append(numpy.tile(shared, edge_count))
        sheet_starts = self.layout["sheets"].start + 3 * self.point_episode_index[self.is_edge_used]
        for offset in range(3):
            rows.append(edge_rows)
            columns.append(sheet_starts + offset)

        if puck_count:
            puck_rows = edge_count + numpy.arange(puck_count)
            group_starts = self.layout["groups"].start + 2 * numpy.repeat(self.puck_group_of_observation, 2)
            shared = numpy.concatenate([camera_columns, height_columns])
            if len(shared):
                rows.append(numpy.repeat(puck_rows, len(shared)))
                columns.append(numpy.tile(shared, puck_count))
            for offset in range(2):
                rows.append(puck_rows)
                columns.append(group_starts + offset)

        for prior_row, name in enumerate(prior_names):
            rows.append(numpy.array([edge_count + puck_count + prior_row]))
            columns.append(numpy.array([self.free_names.index(name)]))

        rows = numpy.concatenate(rows) if rows else numpy.zeros(0, dtype = int)
        columns = numpy.concatenate(columns) if columns else numpy.zeros(0, dtype = int)
        pattern = sparse.coo_matrix((numpy.ones(len(rows), dtype = int), (rows, columns)), shape = (row_count, self.parameter_count))
        return pattern.tocsr()


    def difference_steps(self) -> numpy.ndarray:
        """Relative finite-difference steps for every parameter (scipy multiplies them by max(1, |x|))."""

        steps = numpy.full(self.parameter_count, 1e-6)
        for index, name in enumerate(self.free_names):
            steps[index] = self.joint_parameters["finite_difference_steps"][name]
        return steps


    def solve(self, start: numpy.ndarray) -> Tuple[numpy.ndarray, Dict]:
        """Run every outer round: assign sides with the round's gate, then solve with sides fixed."""

        vector = start.copy()
        history = []
        for gate in self.joint_parameters["edge_gates_m"]:
            self.assign_sides(vector, gate)
            result = optimize.least_squares(
                self.residuals,
                vector,
                jac = "2-point",
                jac_sparsity = self.sparsity(),
                diff_step = self.difference_steps(),
                loss = self.joint_parameters["loss"],
                f_scale = self.joint_parameters["loss_scale_sigma"],
                x_scale = "jac",
                method = "trf",
                tr_solver = "lsmr",
                max_nfev = self.joint_parameters["maximum_function_evaluations"]
            )
            vector = result.x
            edge = self.edge_residuals_px(vector)
            history.append({"gate_m": gate, "edge_points_used": int(self.is_edge_used.sum()), "cost": float(result.cost),
                            "status": int(result.status), "message": str(result.message), "evaluations": int(result.nfev),
                            "edge_median_px": float(numpy.median(numpy.abs(edge))) if len(edge) else None})
        return vector, {"rounds": history, "jacobian": result.jac}



def initial_sheet_poses(
        edge_observations: EdgeObservations,
        camera_parameters: Dict,
        histogram_bin_m: float = 0.004
) -> Dict[str, Dict]:
    """
    Estimate each episode's sheet rectangle from its edge pixels mapped to the table with a camera.

    The rectangle's angle is the four-fold mean of the mapped outward normals; each side's position is the mode of
    the points whose normal points that way. A side that is not seen keeps a nominal size.

    Parameters
    ----------
    edge_observations : EdgeObservations
        The edge pixels.
    camera_parameters : dict
        The camera used for the mapping.
    histogram_bin_m : float, optional
        Bin width of the histograms that find each side.

    Returns
    -------
    dict
        Episode -> {"centre_angle": [x, y, angle], "size_m": [length, width], "sides_seen": [...]}.
    """

    parameters = merged_camera_parameters(camera_parameters)
    rotation = rotation_matrix_from_vector(parameters["rotation_vector_in_forearm_rad"])
    camera = (rotation, numpy.asarray(parameters["translation_in_forearm_m"], dtype = float), float(parameters["focal_length_px"]),
              numpy.asarray(parameters["principal_point_px"], dtype = float), float(parameters["radial_distortion_k1"]))
    points = map_pixels_to_table(edge_observations.pixels, edge_observations.point_positions, edge_observations.point_rotations, camera)
    ahead = map_pixels_to_table(edge_observations.pixels + 3.0 * edge_observations.outwards, edge_observations.point_positions,
                                edge_observations.point_rotations, camera)
    normals = ahead[:, :2] - points[:, :2]
    normals = normals / numpy.maximum(numpy.linalg.norm(normals, axis = 1), 1e-12)[:, None]
    is_finite = numpy.all(numpy.isfinite(points), axis = 1) & numpy.all(numpy.isfinite(normals), axis = 1)

    def mode(values: numpy.ndarray) -> Optional[float]:
        if len(values) < 20:
            return None
        edges = numpy.arange(values.min() - histogram_bin_m, values.max() + 2 * histogram_bin_m, histogram_bin_m)
        counts, edges = numpy.histogram(values, bins = edges)
        best = int(numpy.argmax(counts))
        near = values[(values >= edges[max(best - 1, 0)]) & (values <= edges[min(best + 2, len(edges) - 1)])]
        return float(numpy.median(near))

    poses = {}
    for episode in sorted(set(edge_observations.episode_of_point)):
        selected = is_finite & (edge_observations.episode_of_point == episode)
        if selected.sum() < 50:
            continue
        angles = numpy.arctan2(normals[selected, 1], normals[selected, 0])
        angle = 0.25 * math.atan2(numpy.sin(4 * angles).sum(), numpy.cos(4 * angles).sum())
        cosine, sine = math.cos(angle), math.sin(angle)
        local_x = cosine * points[selected, 0] + sine * points[selected, 1]
        local_y = -sine * points[selected, 0] + cosine * points[selected, 1]
        local_normal_x = cosine * normals[selected, 0] + sine * normals[selected, 1]
        local_normal_y = -sine * normals[selected, 0] + cosine * normals[selected, 1]
        sides = {
            "far": mode(local_x[local_normal_x > 0.9]),
            "near": mode(local_x[local_normal_x < -0.9]),
            "left": mode(local_y[local_normal_y > 0.9]),
            "right": mode(local_y[local_normal_y < -0.9])
        }
        support = {
            "far": int((local_normal_x > 0.9).sum()), "near": int((local_normal_x < -0.9).sum()),
            "left": int((local_normal_y > 0.9).sum()), "right": int((local_normal_y < -0.9).sum())
        }
        # a pair of opposite sides whose distance is far from any sheet size has one wrong side: drop the one with
        # fewer points (it is filled from the other episodes below)
        for first, second, low, high in (("far", "near", 0.24, 0.36), ("left", "right", 0.16, 0.27)):
            if sides[first] is not None and sides[second] is not None and not (low <= sides[first] - sides[second] <= high):
                weaker = first if support[first] < support[second] else second
                sides[weaker] = None
        poses[episode] = {"angle": angle, "sides": sides}

    # sides not seen take the median size of the episodes where both opposite sides were seen
    lengths = [pose["sides"]["far"] - pose["sides"]["near"] for pose in poses.values() if pose["sides"]["far"] is not None and pose["sides"]["near"] is not None]
    widths = [pose["sides"]["left"] - pose["sides"]["right"] for pose in poses.values() if pose["sides"]["left"] is not None and pose["sides"]["right"] is not None]
    length = float(numpy.median(lengths)) if lengths else 0.297
    width = float(numpy.median(widths)) if widths else 0.21
    results = {}
    for episode, pose in poses.items():
        sides = dict(pose["sides"])
        if sides["far"] is None and sides["near"] is not None:
            sides["far"] = sides["near"] + length
        if sides["near"] is None and sides["far"] is not None:
            sides["near"] = sides["far"] - length
        if sides["left"] is None and sides["right"] is not None:
            sides["left"] = sides["right"] + width
        if sides["right"] is None and sides["left"] is not None:
            sides["right"] = sides["left"] - width
        if any(value is None for value in sides.values()):
            continue
        centre_local = numpy.array([(sides["far"] + sides["near"]) / 2.0, (sides["left"] + sides["right"]) / 2.0])
        cosine, sine = math.cos(pose["angle"]), math.sin(pose["angle"])
        centre = numpy.array([cosine * centre_local[0] - sine * centre_local[1], sine * centre_local[0] + cosine * centre_local[1]])
        results[episode] = {"centre_angle": [float(centre[0]), float(centre[1]), float(pose["angle"])],
                            "size_m": [float(sides["far"] - sides["near"]), float(sides["left"] - sides["right"])],
                            "sides_seen": [name for name, value in pose["sides"].items() if value is not None]}
    return results
