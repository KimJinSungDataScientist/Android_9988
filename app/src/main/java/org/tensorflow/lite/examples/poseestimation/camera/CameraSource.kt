// CameraSource.kt
package org.tensorflow.lite.examples.poseestimation.camera

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.graphics.*
import android.hardware.camera2.*
import android.media.ImageReader
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import android.view.SurfaceView
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.tensorflow.lite.examples.poseestimation.VisualizationUtils
import org.tensorflow.lite.examples.poseestimation.YuvToRgbConverter
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.examples.poseestimation.ml.*
import java.io.*
import java.text.SimpleDateFormat
import java.util.*
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max
import kotlin.math.min

class CameraSource(
    private val surfaceView: SurfaceView,
    private val listener: CameraSourceListener? = null
) {

    interface CameraSourceListener {
        fun onFPSListener(fps: Int)
        fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?)
    }

    companion object {
        private const val PREVIEW_WIDTH = 640
        private const val PREVIEW_HEIGHT = 480

        /** Threshold for confidence score. */
        private const val MIN_CONFIDENCE = .2f
        private const val TAG = "Camera Source"
    }

    private val lock = Any()
    private var detector: PoseDetector? = null
    private var classifier: PoseClassifier? = null
    private var isTrackerEnabled = false
    private var yuvConverter: YuvToRgbConverter = YuvToRgbConverter(surfaceView.context)
    private lateinit var imageBitmap: Bitmap

    /** Frame count that have been processed so far in an one second interval to calculate FPS. */
    private var fpsTimer: Timer? = null
    private var frameProcessedInOneSecondInterval = 0
    private var framesPerSecond = 0

    /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
    private val cameraManager: CameraManager by lazy {
        val context = surfaceView.context
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** Readers used as buffers for camera still shots */
    private var imageReader: ImageReader? = null

    /** The [CameraDevice] that will be opened in this fragment */
    private var camera: CameraDevice? = null

    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private var session: CameraCaptureSession? = null

    /** [HandlerThread] where all buffer reading operations run */
    private var imageReaderThread: HandlerThread? = null

    /** [Handler] corresponding to [imageReaderThread] */
    private var imageReaderHandler: Handler? = null
    private var cameraId: String = ""

    // MediaRecorder variables
    private var mediaRecorder: MediaRecorder? = null
    private var isVideoRecording = false
    private var videoFile: File? = null

    // Variable to control coordinate data saving
    private var isRecording = false
    private var coordinateFile: File? = null
    private var coordinateFileWriter: FileWriter? = null

    // List to store recorded coordinate data for similarity computation
    private val recordedData = mutableListOf<List<Pair<Float, Float>>>()

    // Face detection variables
    private var faceDetector: FaceDetector? = null
    private var isFaceDetectionEnabled = false

    // Method to enable or disable face detection
    fun setFaceDetectionEnabled(enabled: Boolean) {
        isFaceDetectionEnabled = enabled
        if (enabled) {
            val highAccuracyOpts = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .build()
            faceDetector = FaceDetection.getClient(highAccuracyOpts)
        } else {
            faceDetector?.close()
            faceDetector = null
        }
    }

    suspend fun initCamera() {
        camera = openCamera(cameraManager, cameraId)
        imageReader =
            ImageReader.newInstance(PREVIEW_WIDTH, PREVIEW_HEIGHT, ImageFormat.YUV_420_888, 3)
        imageReader?.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage()
            if (image != null) {
                if (!::imageBitmap.isInitialized) {
                    imageBitmap =
                        Bitmap.createBitmap(
                            PREVIEW_WIDTH,
                            PREVIEW_HEIGHT,
                            Bitmap.Config.ARGB_8888
                        )
                }
                yuvConverter.yuvToRgb(image, imageBitmap)
                // Create rotated version for portrait display
                val rotateMatrix = Matrix()
                rotateMatrix.postRotate(90.0f)

                val rotatedBitmap = Bitmap.createBitmap(
                    imageBitmap, 0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    rotateMatrix, false
                )
                processImage(rotatedBitmap)
                image.close()
            }
        }, imageReaderHandler)

        imageReader?.surface?.let { surface ->
            session = createSession(listOf(surface))
            val cameraRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )?.apply {
                addTarget(surface)
            }
            cameraRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }
        }
    }

    private suspend fun createSession(targets: List<Surface>): CameraCaptureSession =
        suspendCancellableCoroutine { cont ->
            camera?.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) =
                    cont.resume(captureSession)

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    cont.resumeWithException(Exception("Session error"))
                }
            }, null)
        }

    @SuppressLint("MissingPermission")
    private suspend fun openCamera(manager: CameraManager, cameraId: String): CameraDevice =
        suspendCancellableCoroutine { cont ->
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) = cont.resume(camera)

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    if (cont.isActive) cont.resumeWithException(Exception("Camera error"))
                }
            }, imageReaderHandler)
        }

    fun prepareCamera() {
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // We don't use a front facing camera in this sample.
            val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (cameraDirection != null &&
                cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
            ) {
                continue
            }
            this.cameraId = cameraId
        }
    }

    fun setDetector(detector: PoseDetector) {
        synchronized(lock) {
            if (this.detector != null) {
                this.detector?.close()
                this.detector = null
            }
            this.detector = detector
        }
    }

    fun setClassifier(classifier: PoseClassifier?) {
        synchronized(lock) {
            if (this.classifier != null) {
                this.classifier?.close()
                this.classifier = null
            }
            this.classifier = classifier
        }
    }

    /**
     * Set Tracker for Movenet MuiltiPose model.
     */
    fun setTracker(trackerType: TrackerType) {
        isTrackerEnabled = trackerType != TrackerType.OFF
        (this.detector as? MoveNetMultiPose)?.setTracker(trackerType)
    }

    fun resume() {
        imageReaderThread = HandlerThread("imageReaderThread").apply { start() }
        imageReaderHandler = Handler(imageReaderThread!!.looper)
        fpsTimer = Timer()
        fpsTimer?.scheduleAtFixedRate(
            object : TimerTask() {
                override fun run() {
                    framesPerSecond = frameProcessedInOneSecondInterval
                    frameProcessedInOneSecondInterval = 0
                }
            },
            0,
            1000
        )
    }

    fun close() {
        session?.close()
        session = null
        camera?.close()
        camera = null
        imageReader?.close()
        imageReader = null
        stopImageReaderThread()
        detector?.close()
        detector = null
        classifier?.close()
        classifier = null
        fpsTimer?.cancel()
        fpsTimer = null
        frameProcessedInOneSecondInterval = 0
        framesPerSecond = 0

        faceDetector?.close()
        faceDetector = null
    }

    // process image
    private fun processImage(bitmap: Bitmap) {
        val persons = mutableListOf<Person>()
        var classificationResult: List<Pair<String, Float>>? = null

        synchronized(lock) {
            detector?.estimatePoses(bitmap)?.let {
                persons.addAll(it)

                // Save coordinate data when recording
                if (isRecording) {
                    saveCoordinateData(persons)
                }

                // if the model only returns one item, allow running the Pose classifier.
                if (persons.isNotEmpty()) {
                    classifier?.run {
                        classificationResult = classify(persons[0])
                    }
                }
            }
        }
        frameProcessedInOneSecondInterval++
        if (frameProcessedInOneSecondInterval == 1) {
            // send fps to view
            listener?.onFPSListener(framesPerSecond)
        }

        // if the model returns only one item, show that item's score.
        if (persons.isNotEmpty()) {
            listener?.onDetectedInfo(persons[0].score, classificationResult)
        }

        // 얼굴 모자이크 적용을 위해 비트맵을 복사
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        if (isFaceDetectionEnabled && faceDetector != null) {
            val inputImage = InputImage.fromBitmap(mutableBitmap, 0)
            faceDetector?.process(inputImage)
                ?.addOnSuccessListener { faces ->
                    for (face in faces) {
                        val bounds = face.boundingBox
                        applyMosaicEffect(mutableBitmap, bounds)
                    }
                    visualize(persons, mutableBitmap)
                }
                ?.addOnFailureListener { e ->
                    Log.e(TAG, "Face detection failed", e)
                    visualize(persons, mutableBitmap)
                }
        } else {
            visualize(persons, mutableBitmap)
        }
    }

    private fun applyMosaicEffect(bitmap: Bitmap, bounds: Rect) {
        val canvas = Canvas(bitmap)

        // 얼굴 영역 좌표 보정 (이미지 경계 내에 있도록)
        val left = max(0, bounds.left)
        val top = max(0, bounds.top)
        val right = min(bitmap.width, bounds.right)
        val bottom = min(bitmap.height, bounds.bottom)
        val width = right - left
        val height = bottom - top

        if (width <= 0 || height <= 0) return

        // 얼굴 영역을 크롭
        val faceBitmap = Bitmap.createBitmap(
            bitmap,
            left,
            top,
            width,
            height
        )

        // 모자이크 효과 적용 (이미지 크기를 줄였다가 다시 확대)
        val mosaicSize = 16 // 모자이크 타일 크기 조절
        val mosaicBitmap = Bitmap.createScaledBitmap(
            faceBitmap,
            width / mosaicSize,
            height / mosaicSize,
            false
        )
        val scaledBitmap = Bitmap.createScaledBitmap(
            mosaicBitmap,
            width,
            height,
            false
        )

        // 모자이크된 얼굴을 원본 이미지에 덮어쓰기
        canvas.drawBitmap(scaledBitmap, left.toFloat(), top.toFloat(), null)
    }

    private fun visualize(persons: List<Person>, bitmap: Bitmap) {

        val outputBitmap = VisualizationUtils.drawBodyKeypoints(
            bitmap,
            persons.filter { it.score > MIN_CONFIDENCE }, isTrackerEnabled
        )

        val holder = surfaceView.holder
        val surfaceCanvas = holder.lockCanvas()
        surfaceCanvas?.let { canvas ->
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (canvas.height > canvas.width) {
                val ratio = outputBitmap.height.toFloat() / outputBitmap.width
                screenWidth = canvas.width
                left = 0
                screenHeight = (canvas.width * ratio).toInt()
                top = (canvas.height - screenHeight) / 2
            } else {
                val ratio = outputBitmap.width.toFloat() / outputBitmap.height
                screenHeight = canvas.height
                top = 0
                screenWidth = (canvas.height * ratio).toInt()
                left = (canvas.width - screenWidth) / 2
            }
            val right: Int = left + screenWidth
            val bottom: Int = top + screenHeight

            canvas.drawBitmap(
                outputBitmap, Rect(0, 0, outputBitmap.width, outputBitmap.height),
                Rect(left, top, right, bottom), null
            )
            surfaceView.holder.unlockCanvasAndPost(canvas)
        }
    }

    private fun stopImageReaderThread() {
        imageReaderThread?.quitSafely()
        try {
            imageReaderThread?.join()
            imageReaderThread = null
            imageReaderHandler = null
        } catch (e: InterruptedException) {
            Log.d(TAG, e.message.toString())
        }
    }

    // Start video recording
    suspend fun startVideoRecording() {
        if (isVideoRecording) return

        isRecording = true  // Start saving coordinate data

        // Prepare coordinate data file
        coordinateFile = createCoordinateFile()
        coordinateFileWriter = FileWriter(coordinateFile)

        // Clear recordedData list
        recordedData.clear()

        // Initialize MediaRecorder
        mediaRecorder = MediaRecorder().apply {
            // Set up MediaRecorder configurations
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            // Set the output file path
            videoFile = createVideoFile()
            setOutputFile(videoFile?.absolutePath)
            setVideoEncodingBitRate(10000000)
            setVideoFrameRate(30)
            setVideoSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            prepare()
        }

        isVideoRecording = true

        // Re-initialize camera session with MediaRecorder surface
        val surfaces = mutableListOf<Surface>()
        imageReader?.surface?.let { surface ->
            surfaces.add(surface)
        }
        mediaRecorder?.surface?.let { recorderSurface ->
            surfaces.add(recorderSurface)
        }

        if (surfaces.size < 2) {
            // Handle error: mediaRecorder surface is null
            Log.e(TAG, "Error starting video recording: MediaRecorder surface is null")
            return
        }

        session?.close()
        session = null

        session = createSession(surfaces)
        val cameraRequest = camera?.createCaptureRequest(
            CameraDevice.TEMPLATE_RECORD
        )?.apply {
            for (surface in surfaces) {
                addTarget(surface)
            }
        }
        cameraRequest?.build()?.let {
            session?.setRepeatingRequest(it, null, null)
        }
        mediaRecorder?.start()
    }

    // Stop video recording
    suspend fun stopVideoRecording() {
        if (!isVideoRecording) return

        isRecording = false  // Stop saving coordinate data

        // Stop and release MediaRecorder
        mediaRecorder?.apply {
            stop()
            reset()
            release()
        }
        mediaRecorder = null
        isVideoRecording = false

        // Close coordinate data file
        coordinateFileWriter?.close()
        coordinateFileWriter = null

        // Copy the video file to MediaStore
        videoFile?.let { file ->
            val contentValues = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis() / 1000)
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/PoseEstimation")
            }

            val resolver = surfaceView.context.contentResolver
            val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                try {
                    val outputStream = resolver.openOutputStream(uri)
                    val inputStream = file.inputStream()

                    outputStream?.use { out ->
                        inputStream.use { input ->
                            input.copyTo(out)
                        }
                    }

                    // Optionally, delete the original file
                    file.delete()
                } catch (e: Exception) {
                    Log.e(TAG, "Error saving video to gallery", e)
                }
            } else {
                Log.e(TAG, "Failed to create new MediaStore record.")
            }
        }
        videoFile = null

        // Copy the coordinate file to Downloads
        coordinateFile?.let { file ->
            val contentValues = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, file.name)
                put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                put(MediaStore.Downloads.DATE_ADDED, System.currentTimeMillis() / 1000)
                put(MediaStore.Downloads.RELATIVE_PATH, "Download/PoseEstimation")
            }

            val resolver = surfaceView.context.contentResolver
            val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                try {
                    val outputStream = resolver.openOutputStream(uri)
                    val inputStream = file.inputStream()

                    outputStream?.use { out ->
                        inputStream.use { input ->
                            input.copyTo(out)
                        }
                    }

                    // Optionally, delete the original file
                    file.delete()
                } catch (e: Exception) {
                    Log.e(TAG, "Error saving coordinate data to Downloads", e)
                }
            } else {
                Log.e(TAG, "Failed to create new MediaStore record for coordinates.")
            }
        }
        coordinateFile = null

        // Re-initialize camera session without MediaRecorder surface
        imageReader?.surface?.let { surface ->
            session?.close()
            session = null

            session = createSession(listOf(surface))
            val cameraRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )?.apply {
                addTarget(surface)
            }
            cameraRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }
        }
    }

    // Create video file
    private fun createVideoFile(): File {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val videoFileName = "VIDEO_${sdf.format(Date())}.mp4"
        val storageDir = surfaceView.context.cacheDir
        return File(storageDir, videoFileName)
    }

    // Create coordinate data file
    private fun createCoordinateFile(): File {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val dataFileName = "COORDINATES_${sdf.format(Date())}.csv"
        val storageDir = surfaceView.context.filesDir
        return File(storageDir, dataFileName)
    }

    // Save coordinate data to file and list
    private fun saveCoordinateData(persons: List<Person>) {
        // Save the coordinate data to a CSV file
        coordinateFileWriter?.let { writer ->
            for (person in persons) {
                val sb = StringBuilder()
                sb.append(System.currentTimeMillis())
                sb.append(",")
                sb.append(person.score)
                sb.append(",")
                val frameData = mutableListOf<Pair<Float, Float>>()
                for (keyPoint in person.keyPoints) {
                    sb.append(keyPoint.bodyPart)
                    sb.append(",")
                    sb.append(keyPoint.coordinate.x)
                    sb.append(",")
                    sb.append(keyPoint.coordinate.y)
                    sb.append(",")
                    sb.append(keyPoint.score)
                    sb.append(",")
                    frameData.add(Pair(keyPoint.coordinate.x, keyPoint.coordinate.y))
                }
                sb.append("\n")
                writer.write(sb.toString())
                // Also store the frame data in the list
                recordedData.add(frameData)
            }
        }
    }

    // Function to get recorded coordinate data for similarity computation
    suspend fun readRecordedCoordinateData(): List<List<Pair<Float, Float>>> {
        return withContext(Dispatchers.IO) {
            // Return a copy of the recorded data
            recordedData.toList()
        }
    }
}