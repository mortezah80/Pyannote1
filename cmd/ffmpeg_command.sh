# $1 voice_path
# $2 folder_path
# ffmpeg -i $1 -vn -c:a libvorbis -f segment -segment_time 60 -reset_timestamps 1 $2/segments/%05d.mp3
#ffmpeg -i $1 -vn -ar 44100 -ac 1 -b:a 128k -acodec libmp3lame -f segment -segment_time 60 -reset_timestamps 1 $2/segments/%05d.mp3
ffmpeg -i $1 -f segment -segment_time 60 -c copy $2/segments/%05d.mp3