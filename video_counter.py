import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from main import RepetitionCounter, PoseClassificationVisualizer, FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing

def process_video(video_filename):
    # predicting from video
    # 指定视频路径和输出名称
    video_path = f"uploads/{video_filename}"
    class_name = "up"
    out_video_path = f"static/output.mp4"

    video_cap = cv2.VideoCapture(video_path)

    # Get some video parameters to generate output video with classification.
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize tracker, classifier and counter.
    # Do that before every video as all of them have state.

    # Folder with pose class CSVs. That should be the same folder you using while
    # building classifier to output CSVs.
    pose_samples_folder = "jumping_jack_csvs_out"

    # Initialize tracker.
    pose_tracker = mp_pose.Pose()

    # Initialize embedder.
    pose_embedder = FullBodyPoseEmbedder()

    # Initialize classifier.
    # Check that you are using the same parameters as during bootstrapping.
    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
    )

    # # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.
    pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)


    # 指定动作的两个阈值
    repetition_counter = RepetitionCounter(
        class_name=class_name, enter_threshold=8, exit_threshold=2
    )

    # Initialize renderer.
    pose_classification_visualizer = PoseClassificationVisualizer(
        class_name=class_name,
        # plot_x_max=video_n_frames,
        # # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        # plot_y_max=10,
    )

    # Open output video.
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
    while True:
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
            break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
            )

            # check if all body parts available
            num_visible_landmarks = sum(1 for lmk in pose_landmarks.landmark if lmk.visibility > 0.5)
            if num_visible_landmarks != 33:
                output_frame = Image.fromarray(output_frame)
                warning_draw = ImageDraw.Draw(output_frame)
                warning_text = "请全身入画"
                warning_font = ImageFont.truetype("fonts/msyh.ttc", 100)
                warning_draw.text((50, 50), warning_text, font=warning_font, fill=(0, 235, 255))
                output_frame = np.array(output_frame)

                all_body_parts_visible = False
            else:
                all_body_parts_visible = True

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
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

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(
                pose_classification
            )

            # Count repetitions.
            if all_body_parts_visible:
                repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to maintaining correct
            # smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetitions count.
            if all_body_parts_visible:
                repetitions_count = repetition_counter.n_repeats

        # Draw classification plot and repetition counter.
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count,
        )

        # Save the output frame.
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        frame_idx += 1

    # Close output video.
    out_video.release()

    # Release MediaPipe resources.
    pose_tracker.close()
