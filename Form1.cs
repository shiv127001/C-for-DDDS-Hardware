using System;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace DrowsinessDetection
{
    public partial class Form1 : Form
    {
        private VideoCapture? _capture;
        private CascadeClassifier _eyeCascade;

        public Form1()
        {
            InitializeComponent();
            _eyeCascade = new CascadeClassifier("haarcascade_eye.xml");
        }

        private void ProcessFrame(object? sender, EventArgs? e)
        {
            if (_capture != null && _capture.Ptr != IntPtr.Zero)
            {
                using (Mat frame = _capture.QueryFrame())
                {
                    if (frame != null)
                    {
                        DetectEyes(frame);
                        pictureBox.Image = ConvertMatToBitmap(frame);
                    }
                }
            }
        }

        private void DetectEyes(Mat frame)
        {
            using (var grayFrame = new UMat())
            {
                CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);
                CvInvoke.EqualizeHist(grayFrame, grayFrame);

                var eyes = _eyeCascade.DetectMultiScale(grayFrame, 1.1, 10, Size.Empty);
                foreach (var eye in eyes)
                {
                    CvInvoke.Rectangle(frame, eye, new Bgr(Color.Blue).MCvScalar, 2);
                }

                // Simple logic to detect if eyes are closed
                if (eyes.Length == 0)
                {
                    labelStatus.Text = "Drowsy!";
                    labelStatus.ForeColor = Color.Red;
                }
                else
                {
                    labelStatus.Text = "Alert";
                    labelStatus.ForeColor = Color.Green;
                }
            }
        }

        private void Form1_Load(object? sender, EventArgs? e)
        {
            _capture = new VideoCapture();
            _capture.ImageGrabbed += ProcessFrame;
            _capture.Start();
        }

        private void Form1_FormClosed(object? sender, FormClosedEventArgs e)
        {
            if (_capture != null)
            {
                _capture.Stop();
                _capture.Dispose();
            }
        }

        private Bitmap ConvertMatToBitmap(Mat mat)
        {
            // Validate input Mat object
            if (mat == null || mat.IsEmpty)
            {
                // Handle invalid or empty Mat
                throw new ArgumentException("Input Mat object is invalid or empty.");
            }

            try
            {
                // Verify Emgu CV configuration
                if (!CvInvoke.UseOpenCL || !CvInvoke.UseOpenCLCompatibleGpuMemory)
                {
                    // If OpenCL is not enabled or compatible GPU memory is not used,
                    // it might lead to compatibility issues or performance degradation.
                    throw new InvalidOperationException("Emgu CV is not properly configured. Ensure that OpenCL is enabled and compatible GPU memory is used.");
                }

                // Convert Mat to Image<Bgr, byte>
                using (BitmapConverter converter = new BitmapConverter())
                {
                    return converter.ToBitmap(mat);
                }
            }
            catch (Exception ex)
            {
                // Handle any exceptions during conversion
                throw new ApplicationException("Failed to convert Mat to Bitmap.", ex);
            }
        }


    }
}