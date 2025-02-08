import gradio as gr
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import os
import sys
from utils.utils import validate_color


def extract_audio(video_path):
    """Extract audio from video file"""
    video = VideoFileClip(video_path, fps_source="fps")
    audio = video.audio
    audio_path = "temp_audio.mp3"
    audio.write_audiofile(audio_path)
    audio.close()  # Ensure the audio is properly closed
    video.close()
    return audio_path


def create_srt_content(segments):
    """Convert Whisper segments to SRT format"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt_content


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_font_path(font_file):
    """Get the correct font path"""
    if font_file and os.path.exists(font_file):
        print(f"Using uploaded font: {font_file}")  # Debug log
        return font_file  # Use the uploaded font file
    else:
        # Default fallback font (ensure it's a valid .ttf/.otf file)
        return (
            "/System/Library/Fonts/Supplemental/Arial.ttf"
            if sys.platform == "darwin"
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        )


def add_subtitles_to_video(
    video_path,
    segments,
    font_file,
    font_size,
    color,
    stroke_width,
    stroke_color,
):
    """Add subtitles to video with chosen font"""
    try:
        video = VideoFileClip(video_path, fps_source="fps")

        subtitle_clips = []

        font_path = get_font_path(font_file)  # Ensure correct font path is used

        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time

            text_clip = (
                TextClip(
                    text=segment["text"].strip(),
                    font=font_path,
                    font_size=font_size,
                    color=validate_color(color),
                    stroke_color=validate_color(stroke_color),
                    stroke_width=stroke_width,
                    text_align="center",
                    method="caption",
                    size=(int(video.w * 0.8), int(video.h * 0.3)),
                )
                .with_position(("center", "bottom"))
                .with_duration(duration)
                .with_start(start_time)
            )

            subtitle_clips.append(text_clip)

        final_video = CompositeVideoClip([video] + subtitle_clips)

        output_path = (
            "output_" + os.path.splitext(os.path.basename(video_path))[0] + ".mp4"
        )
        # Match the input video duration with output path video (CompositeVideoClip)
        final_video.duration = video.duration  # Ensure duration matches source video

        final_video.write_videofile(
            output_path,
            audio_codec="aac",
            threads=4,
            preset="medium",
        )

    finally:
        video.close()
        if "final_video" in locals():
            final_video.close()
        for clip in subtitle_clips:
            clip.close()

    return output_path


def transcribe_and_subtitle(
    video_path,
    font_path,
    font_size,
    color,
    stroke_width,
    stroke_color,
):
    """Main function to handle video transcription and subtitling"""
    try:
        audio_path = extract_audio(video_path)

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)

        srt_content = create_srt_content(result["segments"])

        srt_path = video_path.rsplit(".", 1)[0] + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        output_video = add_subtitles_to_video(
            video_path,
            result["segments"],
            font_path,
            font_size,
            color,
            stroke_width,
            stroke_color,
        )

        os.remove(audio_path)

        return srt_path, output_video
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return str(e), None


def create_subtitle_tab():
    """Create Gradio tab for video transcription with font upload"""
    with gr.Tab("Subtitle"):
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            with gr.Column():
                font_input = gr.File(label="Upload Font (.ttf or .otf)")
                with gr.Accordion("Subtitle Appearance Settings", open=False):
                    with gr.Row():
                        font_size = gr.Number(label="Font Size", value=24, precision=0)
                        color = gr.ColorPicker(label="Text Color", value="#FFFFFF")
                    with gr.Row():
                        stroke_width = gr.Number(
                            label="Stroke Width", value=1, precision=0
                        )
                        stroke_color = gr.ColorPicker(
                            label="Stroke Color", value="#000000"
                        )

        with gr.Row():
            transcribe_button = gr.Button("Transcribe and Add Subtitles")

        with gr.Row():
            srt_output = gr.File(label="Generated SRT File")
            video_output = gr.Video(label="Video with Subtitles")

        transcribe_button.click(
            fn=transcribe_and_subtitle,
            inputs=[
                video_input,
                font_input,
                font_size,
                color,
                stroke_width,
                stroke_color,
            ],
            outputs=[srt_output, video_output],
        )
