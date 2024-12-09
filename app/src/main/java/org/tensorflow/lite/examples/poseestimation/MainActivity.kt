// MainActivity.kt (기존 제공된 코드 그대로)
package org.tensorflow.lite.examples.poseestimation

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Process
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.examples.poseestimation.camera.CameraSource
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.*
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.min
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    companion object {
        private const val FRAGMENT_DIALOG = "dialog"
    }

    private lateinit var surfaceView: SurfaceView
    private var modelPos = 1
    private var device = Device.CPU
    private lateinit var tvScore: TextView
    private lateinit var tvFPS: TextView
    private lateinit var spnDevice: Spinner
    private lateinit var spnModel: Spinner
    private lateinit var spnTracker: Spinner
    private lateinit var vTrackerOption: View
    private lateinit var tvClassificationValue1: TextView
    private lateinit var tvClassificationValue2: TextView
    private lateinit var tvClassificationValue3: TextView
    private lateinit var swClassification: SwitchCompat
    private lateinit var vClassificationOption: View
    private lateinit var tvSimilarityScore: TextView
    private var cameraSource: CameraSource? = null
    private var isClassifyPose = false
    private var isRecording = false

    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                openCamera()
            } else {
                ErrorDialog.newInstance(getString(R.string.tfe_pe_request_permission))
                    .show(supportFragmentManager, FRAGMENT_DIALOG)
            }
        }

    private var changeModelListener = object : AdapterView.OnItemSelectedListener {
        override fun onNothingSelected(parent: AdapterView<*>?) {}
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            changeModel(position)
        }
    }

    private var changeDeviceListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            changeDevice(position)
        }
        override fun onNothingSelected(parent: AdapterView<*>?) {}
    }

    private var changeTrackerListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            changeTracker(position)
        }
        override fun onNothingSelected(parent: AdapterView<*>?) {}
    }

    private var setClassificationListener =
        CompoundButton.OnCheckedChangeListener { _, isChecked ->
            showClassificationResult(isChecked)
            isClassifyPose = isChecked
            isPoseClassifier()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        tvScore = findViewById(R.id.tvScore)
        tvFPS = findViewById(R.id.tvFps)
        spnModel = findViewById(R.id.spnModel)
        spnDevice = findViewById(R.id.spnDevice)
        spnTracker = findViewById(R.id.spnTracker)
        vTrackerOption = findViewById(R.id.vTrackerOption)
        surfaceView = findViewById(R.id.surfaceView)
        tvClassificationValue1 = findViewById(R.id.tvClassificationValue1)
        tvClassificationValue2 = findViewById(R.id.tvClassificationValue2)
        tvClassificationValue3 = findViewById(R.id.tvClassificationValue3)
        swClassification = findViewById(R.id.swPoseClassification)
        vClassificationOption = findViewById(R.id.vClassificationOption)
        tvSimilarityScore = findViewById(R.id.tvSimilarityScore)
        initSpinner()
        spnModel.setSelection(modelPos)
        swClassification.setOnCheckedChangeListener(setClassificationListener)
        if (!isCameraPermissionGranted()) {
            requestPermission()
        }

        val recordButton = findViewById<Button>(R.id.recordButton)
        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
                recordButton.text = "Record"
            } else {
                startRecording()
                recordButton.text = "Stop"
            }
            isRecording = !isRecording
        }
    }

    override fun onStart() {
        super.onStart()
        openCamera()
    }

    override fun onResume() {
        cameraSource?.resume()
        super.onResume()
    }

    override fun onPause() {
        cameraSource?.close()
        cameraSource = null
        super.onPause()
    }

    private fun isCameraPermissionGranted(): Boolean {
        return checkPermission(
            Manifest.permission.CAMERA,
            Process.myPid(),
            Process.myUid()
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun openCamera() {
        if (isCameraPermissionGranted()) {
            if (cameraSource == null) {
                cameraSource = CameraSource(surfaceView, object : CameraSource.CameraSourceListener {
                    override fun onFPSListener(fps: Int) {
                        tvFPS.text = getString(R.string.tfe_pe_tv_fps, fps)
                    }
                    override fun onDetectedInfo(
                        personScore: Float?,
                        poseLabels: List<Pair<String, Float>>?
                    ) {
                        tvScore.text = getString(R.string.tfe_pe_tv_score, personScore ?: 0f)
                        poseLabels?.sortedByDescending { it.second }?.let {
                            tvClassificationValue1.text = getString(
                                R.string.tfe_pe_tv_classification_value,
                                convertPoseLabels(if (it.isNotEmpty()) it[0] else null)
                            )
                            tvClassificationValue2.text = getString(
                                R.string.tfe_pe_tv_classification_value,
                                convertPoseLabels(if (it.size >= 2) it[1] else null)
                            )
                            tvClassificationValue3.text = getString(
                                R.string.tfe_pe_tv_classification_value,
                                convertPoseLabels(if (it.size >= 3) it[2] else null)
                            )
                        }
                    }
                }).apply {
                    prepareCamera()
                    setFaceDetectionEnabled(true)
                }
                isPoseClassifier()
                lifecycleScope.launch(Dispatchers.Main) {
                    cameraSource?.initCamera()
                }
            }
            createPoseEstimator()
        }
    }

    private fun convertPoseLabels(pair: Pair<String, Float>?): String {
        if (pair == null) return "empty"
        return "${pair.first} (${String.format("%.2f", pair.second)})"
    }

    private fun isPoseClassifier() {
        cameraSource?.setClassifier(if (isClassifyPose) PoseClassifier.create(this) else null)
    }

    private fun initSpinner() {
        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_models_array,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spnModel.adapter = adapter
            spnModel.onItemSelectedListener = changeModelListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_device_name, android.R.layout.simple_spinner_item
        ).also { adaper ->
            adaper.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spnDevice.adapter = adaper
            spnDevice.onItemSelectedListener = changeDeviceListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_tracker_array, android.R.layout.simple_spinner_item
        ).also { adaper ->
            adaper.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spnTracker.adapter = adaper
            spnTracker.onItemSelectedListener = changeTrackerListener
        }
    }

    private fun changeModel(position: Int) {
        if (modelPos == position) return
        modelPos = position
        createPoseEstimator()
    }

    private fun changeDevice(position: Int) {
        val targetDevice = when (position) {
            0 -> Device.CPU
            1 -> Device.GPU
            else -> Device.NNAPI
        }
        if (device == targetDevice) return
        device = targetDevice
        createPoseEstimator()
    }

    private fun changeTracker(position: Int) {
        cameraSource?.setTracker(
            when (position) {
                1 -> TrackerType.BOUNDING_BOX
                2 -> TrackerType.KEYPOINTS
                else -> TrackerType.OFF
            }
        )
    }

    private fun createPoseEstimator() {
        val poseDetector = when (modelPos) {
            0 -> {
                showPoseClassifier(true)
                showDetectionScore(true)
                showTracker(false)
                MoveNet.create(this, device, ModelType.Lightning)
            }
            1 -> {
                showPoseClassifier(true)
                showDetectionScore(true)
                showTracker(false)
                MoveNet.create(this, device, ModelType.Thunder)
            }
            2 -> {
                showPoseClassifier(false)
                showDetectionScore(false)
                if (device == Device.GPU) {
                    showToast(getString(R.string.tfe_pe_gpu_error))
                }
                showTracker(true)
                MoveNetMultiPose.create(
                    this,
                    device,
                    Type.Dynamic
                )
            }
            3 -> {
                showPoseClassifier(true)
                showDetectionScore(true)
                showTracker(false)
                PoseNet.create(this, device)
            }
            else -> {
                null
            }
        }
        poseDetector?.let { detector ->
            cameraSource?.setDetector(detector)
        }
    }

    private fun showPoseClassifier(isVisible: Boolean) {
        vClassificationOption.visibility = if (isVisible) View.VISIBLE else View.GONE
        if (!isVisible) {
            swClassification.isChecked = false
        }
    }

    private fun showDetectionScore(isVisible: Boolean) {
        tvScore.visibility = if (isVisible) View.VISIBLE else View.GONE
    }

    private fun showClassificationResult(isVisible: Boolean) {
        val visibility = if (isVisible) View.VISIBLE else View.GONE
        tvClassificationValue1.visibility = visibility
        tvClassificationValue2.visibility = visibility
        tvClassificationValue3.visibility = visibility
    }

    private fun showTracker(isVisible: Boolean) {
        if (isVisible) {
            vTrackerOption.visibility = View.VISIBLE
            spnTracker.setSelection(1)
        } else {
            vTrackerOption.visibility = View.GONE
            spnTracker.setSelection(0)
        }
    }

    private fun requestPermission() {
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) -> {
                openCamera()
            }
            else -> {
                requestPermissionLauncher.launch(
                    Manifest.permission.CAMERA
                )
            }
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun startRecording() {
        lifecycleScope.launch {
            cameraSource?.startVideoRecording()
        }
    }

    private fun stopRecording() {
        lifecycleScope.launch {
            cameraSource?.stopVideoRecording()
            val similarityScore = computeSimilarityScore()
            tvSimilarityScore.text = "Similarity Score: ${"%.2f".format(similarityScore)}%"
            tvSimilarityScore.visibility = View.VISIBLE
        }
    }

    private suspend fun computeSimilarityScore(): Float {
        val baselineData = readCoordinateDataFromAssets("baseline.csv")
        val recordedData = cameraSource?.readRecordedCoordinateData()

        if (baselineData.isEmpty() || recordedData.isNullOrEmpty()) {
            return 0f
        }

        val frameCount = min(baselineData.size, recordedData.size)
        if (frameCount == 0) {
            return 0f
        }

        var totalDistance = 0.0
        var totalPossibleDistance = 0.0
        for (i in 0 until frameCount) {
            val baselineFrame = baselineData[i]
            val recordedFrame = recordedData[i]
            val keypointCount = min(baselineFrame.size, recordedFrame.size)
            for (j in 0 until keypointCount) {
                val baselineKeypoint = baselineFrame[j]
                val recordedKeypoint = recordedFrame[j]
                val dx = baselineKeypoint.first - recordedKeypoint.first
                val dy = baselineKeypoint.second - recordedKeypoint.second
                val distance = sqrt((dx * dx + dy * dy).toDouble())
                totalDistance += distance
                totalPossibleDistance += sqrt(1.0 + 1.0)
            }
        }

        val averageDistance = totalDistance / (frameCount * baselineData[0].size)
        val similarityPercentage = (1.0 - (averageDistance / totalPossibleDistance)) * 100.0
        return similarityPercentage.toFloat().coerceIn(0f, 100f)
    }

    private suspend fun readCoordinateDataFromAssets(fileName: String): List<List<Pair<Float, Float>>> {
        val data = mutableListOf<List<Pair<Float, Float>>>()
        return withContext(Dispatchers.IO) {
            try {
                val inputStream = assets.open(fileName)
                val reader = BufferedReader(InputStreamReader(inputStream))
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    val frameData = mutableListOf<Pair<Float, Float>>()
                    val tokens = line!!.split(",")
                    var index = 2
                    while (index + 4 <= tokens.size) {
                        val x = tokens[index + 1].toFloat()
                        val y = tokens[index + 2].toFloat()
                        frameData.add(Pair(x, y))
                        index += 4
                    }
                    data.add(frameData)
                }
                reader.close()
                data
            } catch (e: Exception) {
                e.printStackTrace()
                emptyList()
            }
        }
    }

    class ErrorDialog : DialogFragment() {
        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
            AlertDialog.Builder(activity)
                .setMessage(requireArguments().getString(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok) { _, _ -> }
                .create()

        companion object {
            @JvmStatic
            private val ARG_MESSAGE = "message"
            @JvmStatic
            fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
                arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
            }
        }
    }
}
