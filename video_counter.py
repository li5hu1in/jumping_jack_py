import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from main import (
    RepetitionCounter,
    PoseClassificationVisualizer,
    FullBodyPoseEmbedder,
    PoseClassifier,
    EMADictSmoothing,
)


def process_video(video_filename):
    # 指定视频路径和输出名称
    video_path = f"uploads/{video_filename}"
    class_name = "up" # 分类动作类别为up，表明up在开合跳动作中更重要
    out_video_path = f"static/output.mp4"

    video_cap = cv2.VideoCapture(video_path)

    # 初始化相关变量
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化姿势追踪器、嵌入器、分类器、EMA平滑器、重复计数器、可视化工具
    pose_samples_folder = "jumping_jack_csvs_out"

    pose_tracker = mp_pose.Pose()

    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
    )

    pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)

    # 指定动作的两个阈值
    repetition_counter = RepetitionCounter(
        class_name=class_name, enter_threshold=8, exit_threshold=2
    )

    pose_classification_visualizer = PoseClassificationVisualizer(class_name=class_name)

    # 打开输出视频文件
    out_video = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (video_width, video_height),
    )

    frame_idx = 0
    output_frame = None
    repetitions_count = 0
    all_body_parts_visible = True
    # 处理视频帧
    while True:
        # 循环读取视频帧直到结束
        success, input_frame = video_cap.read()
        if not success:
            break

        # 运行mediapipe姿势追踪器
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        output_frame = input_frame.copy()
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]

        # 如果检测到关键点，在帧上绘制姿势预测
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
            )

            # 检测全身入画
            num_visible_landmarks = sum(
                1 for lmk in pose_landmarks.landmark if lmk.visibility > 0.5
            ) # 可见关键点数量
            # 处理未全身入画的情况
            if num_visible_landmarks != 33:
                output_frame = Image.fromarray(output_frame)
                warning_draw = ImageDraw.Draw(output_frame)
                relative_position = (0.1 * frame_width, 0.1 * frame_height)
                relative_font_size = int(0.1 * min(frame_width, frame_height))
                warning_text = "请全身入画"
                warning_font = ImageFont.truetype("fonts/msyh.ttc", relative_font_size)
                warning_draw.text(
                    relative_position,
                    warning_text,
                    font=warning_font,
                    fill=(0, 235, 255),
                )
                output_frame = np.array(output_frame)

                all_body_parts_visible = False
            else:
                all_body_parts_visible = True

        if pose_landmarks is not None:
            # Get landmarks.
            pose_landmarks = np.array(
                [
                    [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                    for lmk in pose_landmarks.landmark
                ],
                dtype=np.float32,
            )
            assert pose_landmarks.shape == (
                33,
                3,
            ), "Unexpected landmarks shape: {}".format(pose_landmarks.shape)

            # 对当前帧进行姿势分类
            pose_classification = pose_classifier(pose_landmarks)

            # EMA平滑处理
            pose_classification_filtered = pose_classification_filter(
                pose_classification
            )

            # 如果全身入画，计数
            if all_body_parts_visible:
                repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # 未检测到关键点情况
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to maintaining correct smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # 保持计数不变
            repetitions_count = repetition_counter.n_repeats

        # 绘制计数器
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count,
        )

        # 保存输出帧至输出视频文件
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        frame_idx += 1

    # Close output video.
    out_video.release()

    # Release MediaPipe resources.
    pose_tracker.close()
