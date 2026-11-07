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
                val ecg = EcgDigitizer.digitize(bitmap)
                val result = clf.classify(ecg)
                val durationSec = ecg.samplesMv.size / ecg.samplesPerSec
                runOnUiThread {
                    setLoading(false)
                    infoText.text = getString(
                        R.string.info_format,
                        durationSec,
                        ecg.samplesMv.size,
                        ecg.note
                    )
                    val sb = StringBuilder()
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
                    sb.append("\n\n").append(getString(R.string.disclaimer))
                    resultText.text = sb.toString()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    setLoading(false)
                    resultText.text = getString(R.string.error_analysis, e.message ?: "")
                }
            }
        }
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
