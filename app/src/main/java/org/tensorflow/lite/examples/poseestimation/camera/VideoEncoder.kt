// VideoEncoder.kt
package org.tensorflow.lite.examples.poseestimation.camera

import android.graphics.Bitmap
import android.media.*
import android.os.Environment
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ArrayBlockingQueue

class VideoEncoder(private val width: Int, private val height: Int, private val bitRate: Int) {
    private val TAG = "VideoEncoder"
    private var mediaCodec: MediaCodec? = null
    private var mediaMuxer: MediaMuxer? = null
    private var trackIndex = -1
    private var isRecordingStarted = false
    private var frameQueue = ArrayBlockingQueue<Bitmap>(10)
    private var isEncoderThreadRunning = false
    private lateinit var encoderThread: Thread
    private var presentationTimeUs: Long = 0

    fun prepare() {
        val mimeType = MediaFormat.MIMETYPE_VIDEO_AVC
        val format = MediaFormat.createVideoFormat(mimeType, width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        mediaCodec = MediaCodec.createEncoderByType(mimeType)
        mediaCodec?.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)

        val inputSurface = mediaCodec?.createInputSurface()

        // 비디오 파일 생성
        val outputFile = createVideoFile()
        mediaMuxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

        mediaCodec?.start()
        isRecordingStarted = true

        // 인코더 스레드 시작
        isEncoderThreadRunning = true
        encoderThread = Thread(EncoderRunnable())
        encoderThread.start()
    }

    fun stop() {
        if (!isRecordingStarted) return

        isEncoderThreadRunning = false
        encoderThread.join()

        mediaCodec?.signalEndOfInputStream()
        drainEncoder()
        mediaCodec?.stop()
        mediaCodec?.release()
        mediaCodec = null

        mediaMuxer?.stop()
        mediaMuxer?.release()
        mediaMuxer = null

        isRecordingStarted = false
        presentationTimeUs = 0
    }

    fun encodeFrame(bitmap: Bitmap) {
        if (!isRecordingStarted) return

        // 프레임을 큐에 추가
        frameQueue.offer(bitmap.copy(Bitmap.Config.ARGB_8888, false))
    }

    private inner class EncoderRunnable : Runnable {
        override fun run() {
            val canvas = mediaCodec?.createInputSurface()?.lockHardwareCanvas()
            while (isEncoderThreadRunning) {
                val bitmap = frameQueue.poll()
                if (bitmap != null && canvas != null) {
                    synchronized(this) {
                        canvas.drawBitmap(bitmap, 0f, 0f, null)
                        mediaCodec?.createInputSurface()?.unlockCanvasAndPost(canvas)
                    }
                    bitmap.recycle()
                }
            }
        }
    }

    private fun drainEncoder() {
        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferId = mediaCodec?.dequeueOutputBuffer(bufferInfo, 0) ?: break
            if (outputBufferId >= 0) {
                val encodedData = mediaCodec?.getOutputBuffer(outputBufferId) ?: continue
                if (bufferInfo.size != 0) {
                    encodedData.position(bufferInfo.offset)
                    encodedData.limit(bufferInfo.offset + bufferInfo.size)
                    mediaMuxer?.writeSampleData(trackIndex, encodedData, bufferInfo)
                }
                mediaCodec?.releaseOutputBuffer(outputBufferId, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    break
                }
            } else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val newFormat = mediaCodec?.outputFormat
                trackIndex = mediaMuxer?.addTrack(newFormat!!) ?: -1
                mediaMuxer?.start()
            }
        }
    }

    private fun createVideoFile(): File {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val videoFileName = "VIDEO_${sdf.format(Date())}.mp4"
        val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
        return File(storageDir, videoFileName)
    }
}
