package com.deepecg.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var infoText: TextView
    private lateinit var measureText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var captureButton: Button
    private lateinit var galleryButton: Button

    private var photoFile: File? = null
    private var classifier: EcgClassifier? = null
    private val executor = Executors.newSingleThreadExecutor()

    private val takePicture =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                photoFile?.let { analyzeFile(it) }
            } else {
                toast(getString(R.string.photo_annulee))
            }
        }

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { analyzeUri(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        infoText = findViewById(R.id.infoText)
        measureText = findViewById(R.id.measureText)
        progressBar = findViewById(R.id.progressBar)
        captureButton = findViewById(R.id.captureButton)
        galleryButton = findViewById(R.id.galleryButton)

        captureButton.setOnClickListener { dispatchTakePicture() }
        galleryButton.setOnClickListener { pickImage.launch("image/*") }

        classifier = try {
            EcgClassifier(applicationContext)
        } catch (e: Exception) {
            resultText.text = getString(R.string.model_missing)
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
        executor.shutdown()
    }

    private fun dispatchTakePicture() {
        val dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: return
        val file = File.createTempFile("ecg_", ".jpg", dir)
        photoFile = file
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        takePicture.launch(uri)
    }

    private fun analyzeUri(uri: Uri) {
        val file = copyToCache(uri)
        if (file == null) {
            toast(getString(R.string.error_decode))
            return
        }
        analyzeFile(file)
    }

    private fun copyToCache(uri: Uri): File? {
        return try {
            val file = File.createTempFile("ecg_", ".jpg", cacheDir)
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(file).use { output -> input.copyTo(output) }
            }
            file
        } catch (e: Exception) {
            null
        }
    }

    private fun analyzeFile(file: File) {
        val bitmap = decodeRotated(file)
        if (bitmap == null) {
            toast(getString(R.string.error_decode))
            return
        }
        imageView.setImageBitmap(bitmap)

        val clf = classifier
        if (clf == null) {
            resultText.text = getString(R.string.model_missing)
            return
        }

        setLoading(true)
        executor.execute {
            try {
                val multiLead = clf.numLeads > 1
                val ecg = if (multiLead) EcgDigitizer.digitize12(bitmap) else EcgDigitizer.digitize(bitmap)
                val result = clf.classify(ecg)
                val meas = EcgMeasure.measure(ecg.leads, ecg.samplesPerSec)
                val leadSamples = ecg.leads.firstOrNull()?.size ?: 0
                val durationSec = if (ecg.samplesPerSec > 0) leadSamples / ecg.samplesPerSec else 0.0
                runOnUiThread {
                    setLoading(false)
                    infoText.text = getString(
                        R.string.info_format,
                        durationSec,
                        ecg.leads.sumOf { it.size },
                        ecg.note
                    )
                    measureText.text = if (meas != null) {
                        buildMeasureText(meas, if (multiLead) "II" else "I")
                    } else {
                        ""
                    }
                    resultText.text = buildResultText(result)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    setLoading(false)
                    resultText.text = getString(R.string.error_analysis, e.message ?: "")
                }
            }
        }
    }

    private fun buildMeasureText(m: EcgMeasurements, leadName: String): String {
        val sb = StringBuilder()
        sb.append(getString(R.string.measures_header, leadName, m.beats)).append("\n")
        val rhythmFr = when (m.rhythm) {
            "sinus" -> getString(R.string.rhythm_sinus)
            "fa" -> getString(R.string.rhythm_fa)
            else -> getString(R.string.rhythm_irregular)
        }
        sb.append(getString(R.string.heart_rate, m.heartRate))
            .append(" — ")
            .append(getString(R.string.rhythm, rhythmFr))
            .append("\n")
        sb.append(
            getString(
                R.string.intervals,
                m.prMs?.let { "%.0f".format(it) } ?: getString(R.string.not_available),
                m.qrsMs,
                m.qtMs?.let { "%.0f".format(it) } ?: getString(R.string.not_available),
                m.qtcMs?.let { "%.0f".format(it) } ?: getString(R.string.not_available)
            )
        ).append("  ").append(getString(R.string.intervals_ref)).append("\n")
        if (m.axisDeg != null) {
            val ax = m.axisDeg
            val label = when {
                ax in -30.0..90.0 -> getString(R.string.axis_normal)
                ax in -90.0 until -30.0 -> getString(R.string.axis_left)
                ax in 90.0..180.0 -> getString(R.string.axis_right)
                else -> getString(R.string.axis_extreme)
            }
            sb.append(getString(R.string.axis, getString(R.string.axis_value, ax, label))).append("\n")
        }
        sb.append(getString(R.string.measures_note))
        return sb.toString()
    }

    private fun buildResultText(result: ClassificationResult): String {
        val sb = StringBuilder()
        if (result.multilabel) {
            sb.append(getString(R.string.multilabel_header))
            for (i in result.probabilities.indices) {
                val hit = result.thresholds != null &&
                    i < result.thresholds.size &&
                    result.probabilities[i] >= result.thresholds[i]
                sb.append("\n")
                    .append(if (hit) "✓ " else "· ")
                    .append(result.labels[i])
                    .append(" : ")
                    .append("%.0f %%".format(result.probabilities[i] * 100f))
            }
        } else {
            sb.append(
                getString(
                    R.string.result_label,
                    result.label,
                    result.confidence * 100f
                )
            )
            for (i in result.probabilities.indices) {
                sb.append("\n")
                    .append(result.labels[i])
                    .append(" : ")
                    .append("%.0f %%".format(result.probabilities[i] * 100f))
            }
        }
        sb.append("\n\n").append(getString(R.string.disclaimer))
        return sb.toString()
    }

    private fun decodeRotated(file: File): Bitmap? {
        val bitmap = BitmapFactory.decodeFile(file.absolutePath) ?: return null
        return try {
            val exif = ExifInterface(file.absolutePath)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                else -> return bitmap
            }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            bitmap
        }
    }

    private fun setLoading(loading: Boolean) {
        progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        captureButton.isEnabled = !loading
        galleryButton.isEnabled = !loading
        if (loading) resultText.text = getString(R.string.analyzing)
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }
}
