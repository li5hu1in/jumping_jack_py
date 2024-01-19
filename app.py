from flask import (
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    send_from_directory,
    Response,
    jsonify,
)
from werkzeug.utils import secure_filename
from video_counter import process_video
from camera_counter import realtime_counter, return_count
import os

app = Flask(__name__)
app.secret_key = "xqlsl"


@app.route("/")
def index():
    return render_template("index.html")


# 方式一 用户上传视频
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        flash("No video file provided", "error")
        return redirect(url_for("index"))

    video_file = request.files["video"]

    if video_file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("index"))

    try:
        video_file.save("uploads/" + secure_filename(video_file.filename))

        # 处理视频
        process_video(f"{video_file.filename}")

        return render_template(
            "index.html",
            result=os.path.exists("static/output.mp4"),
            scroll_to_button=True,
        )

    except Exception as e:
        flash(f"Error when processing video: {str(e)}", "error")
        return redirect(url_for("index"))


# 方式一 下载结果
@app.route("/download_output")
def download_output():
    try:
        output_file_path = "static/output.mp4"
        return send_from_directory(
            os.path.dirname(output_file_path),
            os.path.basename(output_file_path),
            as_attachment=True,
        )
    except Exception as e:
        flash(f"Error when downloading output: {str(e)}", "error")
        return redirect(url_for("index"))


# 下载完成后删除相关视频
@app.route("/remove_files")
def remove_files():
    try:
        os.remove("static/output.mp4")
        for filename in os.listdir("uploads"):
            os.remove(f"uploads/{filename}")
        flash("Files removed successfully", "info")
    except Exception as e:
        flash(f"Error when removing files: {str(e)}", "error")
    return redirect(url_for("index"))


# 按帧显示摄像头画面
@app.route("/video_stream")
def video_stream():
    return Response(
        realtime_counter(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# 获取摄像头实时计数变量
@app.route("/video_res")
def video_res():
    count_number = return_count()
    return jsonify({"count": count_number})


if __name__ == "__main__":
    app.run(debug=True)
