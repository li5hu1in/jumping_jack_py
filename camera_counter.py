import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from main import (
    FullBodyPoseEmbedder,
    PoseClassifier,
    EMADictSmoothing,
    RepetitionCounter,
    PoseClassificationVisualizer,
)


# 将计数返回给HTML
repetitions_count_global = {"count": 0}
def return_count():
    return repetitions_count_global["count"]


def realtime_counter():
    class_name = "up"

    video_cap = cv2.VideoCapture(0)
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化
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
        class_name=class_name, enter_threshold=7, exit_threshold=3
    )

    pose_classification_visualizer = PoseClassificationVisualizer(class_name=class_name)

    output_frame = None
    repetitions_count = 0
    all_body_parts_visible = {"flag": True}
    
    # 循环读取来自摄像头的视频帧
    while True:
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
            break

        # 水平翻转摄像头
        input_frame = cv2.flip(input_frame, 1)

        # 运行mediapipe姿势追踪器
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        output_frame = input_frame.copy()
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]

        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
            )

            # 检测全身入画
            num_visible_landmarks = sum(
                1 for lmk in pose_landmarks.landmark if lmk.visibility > 0.5
            )
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

                all_body_parts_visible["flag"] = False
            else:
                all_body_parts_visible["flag"] = True

        if pose_landmarks is not None:
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

            pose_classification = pose_classifier(pose_landmarks)

            pose_classification_filtered = pose_classification_filter(
                pose_classification
            )

            if all_body_parts_visible["flag"]:
                repetitions_count = repetition_counter(pose_classification_filtered)
        # 未检测到关键点 -> 当作未分类处理，计数器保持不变
        else:
            pose_classification = None

            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            repetitions_count = repetition_counter.n_repeats

        # 绘制输出帧
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count,
        )

        # 更新全局计数变量
        repetitions_count_global["count"] = repetitions_count

        # 画面按帧传给HTML
        _, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
        )
        frame = jpeg.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    video_cap.release()
