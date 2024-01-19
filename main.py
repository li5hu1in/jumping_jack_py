# Reference: https://www.bilibili.com/video/BV1dL4y1h7Q6/

from matplotlib import pyplot as plt
import os
import csv
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


# helper function
def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


# 编码身体姿态
class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky_1",
            "right_pinky_1",
            "left_index_1",
            "right_index_1",
            "left_thumb_2",
            "right_thumb_2",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it easier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index("left_shoulder")]
        right_shoulder = landmarks[self._landmark_names.index("right_shoulder")]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We different types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array(
            [
                # One joint.
                self._get_distance(
                    self._get_average_by_names(landmarks, "left_hip", "right_hip"),
                    self._get_average_by_names(
                        landmarks, "left_shoulder", "right_shoulder"
                    ),
                ),
                self._get_distance_by_names(landmarks, "left_shoulder", "left_elbow"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_elbow"),
                self._get_distance_by_names(landmarks, "left_elbow", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_elbow", "right_wrist"),
                self._get_distance_by_names(landmarks, "left_hip", "left_knee"),
                self._get_distance_by_names(landmarks, "right_hip", "right_knee"),
                self._get_distance_by_names(landmarks, "left_knee", "left_ankle"),
                self._get_distance_by_names(landmarks, "right_knee", "right_ankle"),
                # Two joints.
                self._get_distance_by_names(landmarks, "left_shoulder", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_wrist"),
                self._get_distance_by_names(landmarks, "left_hip", "left_ankle"),
                self._get_distance_by_names(landmarks, "right_hip", "right_ankle"),
                # Four joints.
                self._get_distance_by_names(landmarks, "left_hip", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_hip", "right_wrist"),
                # Five joints.
                self._get_distance_by_names(landmarks, "left_shoulder", "left_ankle"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_ankle"),
                self._get_distance_by_names(landmarks, "left_hip", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_hip", "right_wrist"),
                # Cross body.
                self._get_distance_by_names(landmarks, "left_elbow", "right_elbow"),
                self._get_distance_by_names(landmarks, "left_knee", "right_knee"),
                self._get_distance_by_names(landmarks, "left_wrist", "right_wrist"),
                self._get_distance_by_names(landmarks, "left_ankle", "right_ankle"),
            ]
        )

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


# 每张图片特征
class PoseSample(object):
    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


# 异常值
class PoseSampleOutlier(object):
    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


# K-means聚类
class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(
        self,
        pose_samples_folder,
        pose_embedder,
        file_extension="csv",
        file_separator=",",
        n_landmarks=33,
        n_dimensions=3,
        top_n_by_max_distance=30, # 按最大距离选择的样本数量
        top_n_by_mean_distance=10, # 按平均距离选择的样本数量
        axes_weights=(1.0, 1.0, 0.2),
    ):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        # 加载姿势样本
        self._pose_samples = self._load_pose_samples(
            pose_samples_folder,
            file_extension,
            file_separator,
            n_landmarks,
            n_dimensions,
            pose_embedder,
        )

    def _load_pose_samples(
        self,
        pose_samples_folder,
        file_extension,
        file_separator,
        n_landmarks,
        n_dimensions,
        pose_embedder,
    ):
        """Loads pose samples from a given folder.

        Required folder structure:
          neutral_standing.csv
          pushups_down.csv
          pushups_up.csv
          squats_down.csv
          ...

        Required CSV structure:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
          ...
        """
        # 每个文件代表一个姿势类别
        file_names = [
            name
            for name in os.listdir(pose_samples_folder)
            if name.endswith(file_extension)
        ]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[: -(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert (
                        len(row) == n_landmarks * n_dimensions + 1
                    ), "Wrong number of values: {}".format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape(
                        [n_landmarks, n_dimensions]
                    )
                    pose_samples.append(
                        PoseSample(
                            name=row[0],
                            landmarks=landmarks,
                            class_name=class_name,
                            embedding=pose_embedder(landmarks),
                        )
                    )

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database."""
        # 寻找异常值
        outliers = []
        for sample in self._pose_samples:
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [
                class_name
                for class_name, count in pose_classification.items()
                if count == max(pose_classification.values())
            ]

            # 最近的姿势具有不同类别/被检测为最近的姿势的类别不止一个 -> 异常值
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(
                    PoseSampleOutlier(sample, class_names, pose_classification)
                )

        return outliers

    def __call__(self, pose_landmarks):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (
            self._n_landmarks,
            self._n_dimensions,
        ), "Unexpected shape: {}".format(pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(
            pose_landmarks * np.array([-1, 1, 1])
        )

        # 按最大距离过滤
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[: self._top_n_by_max_distance]

        # 按平均距离过滤
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[: self._top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [
            self._pose_samples[sample_idx].class_name
            for _, sample_idx in mean_dist_heap
        ]
        result = {
            class_name: class_names.count(class_name) for class_name in set(class_names)
        }

        return result


# 对姿势分类结果平滑处理
# 计算给定时间窗口内每个姿势类别的指数移动平均值
class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size # 时间窗口大小
        self._alpha = alpha # 指数移动平均的平滑因子

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes are replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[: self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= 1.0 - self._alpha

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data


# 目标姿势类别重复次数
class RepetitionCounter(object):
    """Counts number of repetitions of given target pose class."""

    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        self._class_name = class_name

        # If pose counter passes given threshold, then we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # 标记是否处于给定姿势中
        self._pose_entered = False

        # Number of times we exited the pose.
        self._n_repeats = 0

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification):
        """Counts number of repetitions happened until given frame.
        直到给定帧的重复次数

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'pushups_down': 8.3,
                'pushups_up': 1.7,
              }

        Returns:
          Integer counter of repetitions.
        """
        # 获取姿势置信度
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # 第一帧或不在姿势中的情况下，只需检查是否在此帧进入姿势并更新状态
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats

        # 如果在姿势中并正在退出，增加计数器，更新状态
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False

        return self._n_repeats


# 在帧上绘制计数器
class PoseClassificationVisualizer(object):
    """Keeps track of classifications for every frame and renders them."""

    def __init__(
        self,
        class_name,
        counter_location_x=0.75,
        counter_location_y=0.15,
        counter_font_path="fonts/Roboto-Regular.ttf",
        counter_font_color="cyan",
        counter_font_size=0.15,
    ):
        self._class_name = class_name
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None

        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(
        self,
        frame,
        pose_classification,
        pose_classification_filtered,
        repetitions_count,
    ):
        """Renders pose classification and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Output frame with counter.
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
            self._counter_font = ImageFont.truetype(self._counter_font_path, size=font_size)
        output_img_draw.text(
            (
                output_width * self._counter_location_x,
                output_height * self._counter_location_y,
            ),
            str(repetitions_count),
            font=self._counter_font,
            fill=self._counter_font_color,
        )

        return output_img


# 提取训练集关键点坐标
# 处理图像，姿势分类
class BootstrapHelper(object):
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # 获取姿势类别列表
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if not n.startswith('.')]
        )

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...

        Produced CSVs out folder:
          pushups_up.csv
          pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # 创建输出文件夹
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print("Bootstrapping ", pose_class_name, file=sys.stderr)

            # 姿势类别的路径
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + ".csv")
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
                )
                # 获取图像列表
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder) if not n.startswith(".")]
                )
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                # 打印进度
                for image_name in tqdm.tqdm(image_names):
                    # Load images.
                    input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                    # 初始化新追踪器
                    with mp_pose.Pose() as pose_tracker:
                        # upper_body_only
                        result = pose_tracker.process(image=input_frame)
                        pose_landmarks = result.pose_landmarks

                    # 若检测到姿势，保存带有姿势预测的图像
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS,
                        )
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(images_out_folder, image_name), output_frame
                    )

                    # 检测到姿势保存landmarks
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height, frame_width = (
                            output_frame.shape[0],
                            output_frame.shape[1],
                        )
                        pose_landmarks = np.array(
                            [
                                [
                                    lmk.x * frame_width,
                                    lmk.y * frame_height,
                                    lmk.z * frame_width,
                                ]
                                for lmk in pose_landmarks.landmark
                            ],
                            dtype=np.float32,
                        )
                        assert pose_landmarks.shape == (
                            33,
                            3,
                        ), "Unexpected landmarks shape: {}".format(pose_landmarks.shape)
                        csv_out_writer.writerow(
                            [image_name]
                            + pose_landmarks.flatten().astype(np.str_).tolist()
                        )

                    # Draw XZ projection and concatenate with the image.
                    projection_xz = self._draw_xz_projection(
                        output_frame=output_frame, pose_landmarks=pose_landmarks
                    )
                    output_frame = np.concatenate((output_frame, projection_xz), axis=1)

    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color="red"):
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        img = Image.new("RGB", (frame_width, frame_height), color="white")

        if pose_landmarks is None:
            return np.asarray(img)

        # 根据图片宽度缩放半径
        r *= frame_width * 0.01

        draw = ImageDraw.Draw(img)
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
            # Flip Z and move hips center to the center of the image.
            x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
            x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)

        return np.asarray(img)

    def align_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.

        Leaves only intersection of samples in both image folders and CSVs.
        仅保留图像文件夹和csv中的样本交际
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + ".csv")

            # Read CSV into memory.
            rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=",")
                for row in csv_out_reader:
                    rows.append(row)

            # Image names left in CSV.
            image_names_in_csv = []

            # 删除没有对应图像的行，重新写入csv
            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
                )
                for row in rows:
                    image_name = row[0]
                    image_path = os.path.join(images_out_folder, image_name)
                    if os.path.exists(image_path):
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        print("Removed image from CSV: ", image_path)

            # 删除没有对应csv行的图像
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print("Removed image from folder: ", image_path)

    def analyze_outliers(self, outliers):
        """Classifies each sample against all other to find outliers.
        每个样本再次分类以查找异常值

        If sample is classified differrrently than the original class - it should
        either be deleted or more similar samples should be added.
        如果样本的分类与原始类别不同，删除或添加更相似的样本
        """
        for outlier in outliers:
            image_path = os.path.join(
                self._images_out_folder, outlier.sample.class_name, outlier.sample.name
            )

            print("Outlier")
            print("  sample path =    ", image_path)
            print("  sample class =   ", outlier.sample.class_name)
            print("  detected class = ", outlier.detected_class)
            print("  all classes =    ", outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))

    def remove_outliers(self, outliers):
        """Removes outliers from the image folders."""
        for outlier in outliers:
            image_path = os.path.join(
                self._images_out_folder, outlier.sample.class_name, outlier.sample.name
            )
            os.remove(image_path)

    def print_images_in_statistics(self):
        """Prints statistics from the input image folder."""
        self._print_images_statistics(self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """Prints statistics from the output image folder."""
        self._print_images_statistics(self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print("Number of images per pose class:")
        for pose_class_name in pose_class_names:
            n_images = len(
                [
                    n
                    for n in os.listdir(os.path.join(images_folder, pose_class_name))
                    if not n.startswith(".")
                ]
            )
            print("  {}: {}".format(pose_class_name, n_images))


# 训练

# 指定训练集路径
# 若已有csv文件则不需要训练
if not os.path.exists("jumping_jack_csvs_out"):
    bootstrap_images_in_folder = "jumping_jack_dataset"

    # Output folders for bootstrapped images and CSVs.
    bootstrap_images_out_folder = "jumping_jack_images_out"
    bootstrap_csvs_out_folder = "jumping_jack_csvs_out"

    # 初始化helper用于预处理
    bootstrap_helper = BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )

    # 检查每个动作有多少张图像
    bootstrap_helper.print_images_in_statistics()

    # 提取特征
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # 检查每个动作有多少张图像提取了特征
    bootstrap_helper.print_images_out_statistics()

    # 删除没有检测到姿势的图像
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()


    # detecting outliers
    # Align CSVs with filtered images.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # Transforms pose landmarks into embedding.
    pose_embedder = FullBodyPoseEmbedder()

    # 通过分类器查找异常值
    pose_classifier = PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
    )

    outliers = pose_classifier.find_pose_sample_outliers()
    print("Number of outliers: ", len(outliers))

    # 查看所有异常数据点
    bootstrap_helper.analyze_outliers(outliers)

    # 移除异常数据点
    bootstrap_helper.remove_outliers(outliers)

    # 重新整理二分类数据
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()
