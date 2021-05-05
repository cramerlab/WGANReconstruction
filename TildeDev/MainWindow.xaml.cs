using LiveCharts;
using LiveCharts.Defaults;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Warp;
using Warp.Headers;
using Warp.NNModels;
using Warp.Tools;

namespace TildeDev
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public decimal LearningRate
        {
            get { return (decimal)GetValue(LearningRateProperty); }
            set { SetValue(LearningRateProperty, value); }
        }

        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate", typeof(decimal), typeof(MainWindow), new PropertyMetadata(0.0001M));


        private bool ShouldSaveModel = false;
        private bool ShouldSaveRecs = false;


        private string WorkingDirectory = @"D:\particleWGAN\";
        private string DirectoryParticles = "particles";

        private int Dim = 64;

        private int BatchSize = 128;
        float Lambda = 10;

        int NThreads = 3;
        int ProcessingDevice = 0;

        public MainWindow()
        {
            InitializeComponent();

            int DiscIters = 5;
            SliderLearningRate.DataContext = this;
        }

        private void ButtonStart_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonStart.IsEnabled = false;

            Task.Run(() =>
            {
                //ConsoleAllocator.ShowConsoleWindow();

                WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
                C2DNet TrainModel = new C2DNet(Dim, 128, new[] { 1 }, BatchSize);
                WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

                //TrainModel.Load(Path.Combine(WorkingDirectory, "C2DNet_20210126_155455.pt"));

                GPU.SetDevice(ProcessingDevice);

                Image ImagesReal = new Image(new int3(Dim, Dim, BatchSize));
                Image ImagesTarget = new Image(new int3(Dim, Dim, BatchSize));
                Image ImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);
                float[] RefDotProd = new float[BatchSize / 2];

                Semaphore ReloadBlock = new Semaphore(1, 1);
                bool HasBeenProcessed = true;

                Star TableIn = new Star(Path.Combine(WorkingDirectory, "c4_coords.star"), "particles");

                string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
                HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
                UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryParticles, s)));
                int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
                TableIn = TableIn.CreateSubset(KeepRows);

                int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryParticles, UniqueStackNames.First())).Dimensions.X;

                TableIn.AddColumn("rlnVoltage", "200.0");
                TableIn.AddColumn("rlnSphericalAberration", "2.7");
                TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
                TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
                TableIn.AddColumn("rlnMagnification", "10000");

                var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
                float2[] AllShifts = new float2[TableIn.RowCount];
                float[] AllAngles = new float[TableIn.RowCount];
                float3[] AllEulers = TableIn.GetRelionAngles();
                {
                    ColumnStackNames = TableIn.GetColumn("rlnImageName");
                    float[] ColumnX = TableIn.GetColumn("rlnOriginXAngst").Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                    float[] ColumnY = TableIn.GetColumn("rlnOriginYAngst").Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                    float[] ColumnPsi = TableIn.GetColumn("rlnAnglePsi").Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToArray();

                    for (int r = 0; r < TableIn.RowCount; r++)
                    {
                        string s = ColumnStackNames[r];
                        int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
                        string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
                        AllParticleAddresses[r] = (ID, Name);

                        AllShifts[r] = new float2(ColumnX[r], ColumnY[r]) / 6f;
                        AllAngles[r] = ColumnPsi[r] * Helper.ToRad;
                    }
                }
                CTF[] AllParticleCTF = TableIn.GetRelionCTF();
                int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

                float Lowpass = 1.0f;

                ParameterizedThreadStart ReloadLambda = (par) =>
                {
                    GPU.SetDevice(ProcessingDevice);

                    Random ReloadRand = new Random((int)par);
                    RandomNormal ReloadRandN = new RandomNormal((int)par);
                    bool OwnBatchUsed = true;

                    Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                    Image TImagesReal = new Image(new int3(Dim, Dim, BatchSize));
                    Image TImagesRealFT = new Image(new int3(Dim, Dim, BatchSize), true, true);
                    Image TImagesTrans = new Image(new int3(Dim, Dim, BatchSize));
                    Image TImagesTarget = new Image(new int3(Dim, Dim, BatchSize));
                    Image TImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);
                    float[] TRefDotProd = new float[BatchSize / 2];

                    Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);

                    Image Ref = Image.FromFile(Path.Combine(WorkingDirectory, "c4.mrc")).AsScaled(new int3(Dim)).AndDisposeParent();
                    Ref.MaskSpherically(Dim / 2, 8, true);
                    Projector RefProj = new Projector(Ref, 2);

                    int PlanForw = 0, PlanBack = 0, PlanForwSmall = 0;
                    PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                    PlanForwSmall = GPU.CreateFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);

                    while (true)
                    {
                        // If this thread succeeded at pushing its previously loaded batch to processing
                        if (OwnBatchUsed)
                        {
                            int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize / 2, ReloadRand.Next());
                            SubsetIDs = Helper.ArrayOfFunction(i => SubsetIDs[i / 2], BatchSize);

                            // Read, and copy or rescale images from prepared stacks

                            CTF[] CTFs = Helper.IndexedSubset(AllParticleCTF, SubsetIDs);
                            foreach (var ctf in CTFs)
                                ctf.DefocusDelta = 0;

                            GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                            CTFCoords.GetDevice(Intent.Read),
                                            IntPtr.Zero,
                                            (uint)CTFCoords.ElementsSliceComplex,
                                            CTFs.Select(c => c.ToStruct()).ToArray(),
                                            false,
                                            (uint)BatchSize);

                            float[] AngleOffsets = Helper.ArrayOfFunction(i => (float)ReloadRand.Next(360), BatchSize / 2);
                            TImagesTrans = RefProj.ProjectToRealspace(new int2(Dim), Helper.IndexedSubset(AllEulers, SubsetIDs).Select((v, i) => new float3(v.X, v.Y, i % 2 == 0 ? 0 : AngleOffsets[i / 2]) * Helper.ToRad).ToArray());

                            TRefDotProd = AngleOffsets.Select(v => (float)Math.Cos(v * Helper.ToRad)).ToArray();


                            TImagesTrans.Bandpass(0, Lowpass, false, 0.05f);
                            TImagesTrans.Normalize();

                            TImagesTrans.ShiftSlices(Helper.ArrayOfFunction(i => new float3(ReloadRandN.NextSingle(0, 2), ReloadRandN.NextSingle(0, 2), 0), BatchSize));

                            TImagesCTF.Fill(1);

                            TImagesTrans.MaskSpherically(Dim / 2, 8, false);

                            OwnBatchUsed = false;
                        }

                        ReloadBlock.WaitOne();
                        // If previously pushed batch from any thread has already been processed
                        if (HasBeenProcessed)
                        {
                            GPU.CopyDeviceToDevice(TImagesTrans.GetDevice(Intent.Read), ImagesReal.GetDevice(Intent.Write), TImagesTrans.ElementsReal);
                            //GPU.CopyDeviceToDevice(TImagesReal.GetDevice(Intent.Read), ImagesReal.GetDevice(Intent.Write), TImagesReal.ElementsReal);
                            GPU.CopyDeviceToDevice(TImagesTrans.GetDevice(Intent.Read), ImagesTarget.GetDevice(Intent.Write), TImagesTarget.ElementsReal);
                            GPU.CopyDeviceToDevice(TImagesCTF.GetDevice(Intent.Read), ImagesCTF.GetDevice(Intent.Write), TImagesCTF.ElementsReal);

                            RefDotProd = TRefDotProd.ToArray();

                            TImagesTrans.Dispose();

                            OwnBatchUsed = true;
                            HasBeenProcessed = false;
                            //Debug.WriteLine($"Using T {par}");
                        }
                        ReloadBlock.Release();
                    }
                };
                Thread[] ReloadThreads = Helper.ArrayOfFunction(i => new Thread(ReloadLambda), NThreads);
                for (int i = 0; i < NThreads; i++)
                    ReloadThreads[i].Start(i);

                GPU.SetDevice(ProcessingDevice);

                Random Rand = new Random(123);

                List<ObservablePoint> LossPointsReal = new List<ObservablePoint>();
                List<ObservablePoint> LossPointsFake = new List<ObservablePoint>();

                long IterationsDone = 0;
                List<float> AllLosses = new List<float>();
                List<float> AllKLD = new List<float>();
                List<float> AllLossesFake = new List<float>();

                int AlignAfter = 100000;

                while (true)
                {
                    if (HasBeenProcessed)
                        continue;

                    ReloadBlock.WaitOne();

                    float[] Loss = null;
                    float[] LossKLD = null;
                    float[] LossFake = null;
                    Image Prediction = null;
                    Image PredictionDeconv = null;
                    Image AlignedTarget = null;
                    float[] SourceData = null;
                    float[] TargetData = null;
                    float[] AverageData = null;

                    {
                        float CurrentLearningRate = 0;
                        Dispatcher.Invoke(() => CurrentLearningRate = (float)LearningRate);

                        TrainModel.Train(ImagesReal,
                                         ImagesTarget,
                                         ImagesCTF,
                                         RefDotProd,
                                         CurrentLearningRate,
                                         IterationsDone >= AlignAfter,
                                         Lowpass,
                                         out Prediction,
                                         out PredictionDeconv,
                                         out AlignedTarget,
                                         out Loss,
                                         out LossKLD);

                        AllLosses.Add(Loss[0]);
                        AllKLD.Add(LossKLD[0]);

                        HasBeenProcessed = true;
                    }

                    if (IterationsDone % 20 == 0)
                    {
                        WriteToLog($"{MathHelper.Mean(AllLosses):F6}, KLD = {MathHelper.Mean(AllKLD):F6}");

                        LossPointsReal.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLosses)));
                        Dispatcher.Invoke(() => SeriesLossReal.Values = new ChartValues<ObservablePoint>(LossPointsReal));

                        AlignedTarget.WriteMRC("d_alignedtarget.mrc", true);
                        PredictionDeconv.WriteMRC("d_decoded.mrc", true);

                        Func<float[], ImageSource> MakeImage = (data) =>
                        {
                            float[] OneSlice = data.ToArray();
                            float2 GlobalMeanStd = MathHelper.MeanAndStd(OneSlice);

                            byte[] BytesXY = new byte[OneSlice.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSlice[y * Dim + x] - GlobalMeanStd.X) / GlobalMeanStd.Y;
                                    Value = (Value + 7f) / 14f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceImage.Freeze();

                            return SliceImage;
                        };

                        {
                            ImageSource SliceImage = MakeImage(ImagesReal.GetHost(Intent.Read)[4]);
                            Dispatcher.Invoke(() => ImageSource.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(PredictionDeconv.GetHost(Intent.Read)[4]);
                            Dispatcher.Invoke(() => ImageTarget.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(AlignedTarget.GetHost(Intent.Read)[4]);
                            Dispatcher.Invoke(() => ImageAverage.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(AlignedTarget.GetHost(Intent.Read)[5]);
                            Dispatcher.Invoke(() => ImagePrediction.Source = SliceImage);
                        }

                        if (ShouldSaveModel)
                        {
                            ShouldSaveModel = false;

                            TrainModel.Save(WorkingDirectory + @"C2DNet_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt");
                            Thread.Sleep(10000);

                            Dispatcher.Invoke(() => ButtonSave.IsEnabled = true);
                        }
                    }

                    while (AllLosses.Count > 10)
                        AllLosses.RemoveAt(0);
                    while (AllKLD.Count > 10)
                        AllKLD.RemoveAt(0);

                    IterationsDone++;
                    Dispatcher.Invoke(() => TextCoverage.Text = $"{IterationsDone} iterations done");

                    ReloadBlock.Release();
                }
            });
        }

        private void ButtonSave_OnClick(object sender, RoutedEventArgs e)
        {
            ShouldSaveModel = true;
            ButtonSave.IsEnabled = false;
        }

        private void WriteToLog(string line)
        {
            Dispatcher.Invoke(() =>
            {
                TextOutput.Text += line + "\n";
                TextOutput.ScrollToLine(TextOutput.LineCount - 1);
            });
        }

        private void ButtonTest_OnClick(object sender, RoutedEventArgs e)
        {
        }
    }

    internal static class ConsoleAllocator
    {
        [DllImport(@"kernel32.dll", SetLastError = true)]
        static extern bool AllocConsole();

        [DllImport(@"kernel32.dll")]
        static extern IntPtr GetConsoleWindow();

        [DllImport(@"user32.dll")]
        static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        const int SwHide = 0;
        const int SwShow = 5;


        public static void ShowConsoleWindow()
        {
            var handle = GetConsoleWindow();

            if (handle == IntPtr.Zero)
            {
                AllocConsole();
            }
            else
            {
                ShowWindow(handle, SwShow);
            }
        }

        public static void HideConsoleWindow()
        {
            var handle = GetConsoleWindow();

            ShowWindow(handle, SwHide);
        }
    }
}
