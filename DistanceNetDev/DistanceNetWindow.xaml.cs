using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using LiveCharts;
using LiveCharts.Defaults;
using Warp;
using Warp.Headers;
using Warp.NNModels;
using Warp.Tools;

namespace DistanceNetDev
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
        private string DirectoryReal = "raw";
        private string DirectoryFake = "sim";

        private int Dim = 128;

        private int BatchSize = 52;

        int NThreads = 1;
        int ProcessingDevice = 0;

        public MainWindow()
        {
            InitializeComponent();

            SliderLearningRate.DataContext = this;
        }

        private void ButtonStart_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonStart.IsEnabled = false;

            Task.Run(() =>
            {
                WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
                DistanceNet TrainModel = new DistanceNet(new int2(Dim), new[] { 1 }, BatchSize);
                ParticleWGAN GeneratorModel = new ParticleWGAN(new int2(Dim), 64, new[] { 2 }, BatchSize);
                GeneratorModel.Load(Path.Combine(WorkingDirectory, "ParticleWGAN_20210111_210604.pt"));
                WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

                GPU.SetDevice(ProcessingDevice);

                Image ImagesReference = new Image(new int3(Dim, Dim, BatchSize));
                Image ImagesData = new Image(new int3(Dim, Dim, BatchSize));
                Image ImagesDiff = new Image(new int3(1, 1, BatchSize));
                float[] CorrNaive = new float[BatchSize];

                Semaphore ReloadBlock = new Semaphore(1, 1);
                bool HasBeenProcessed = true;

                Star TableIn = new Star(Path.Combine(WorkingDirectory, "c4_coords.star"), "particles");

                string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
                HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
                UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryReal, s)));
                int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
                TableIn = TableIn.CreateSubset(KeepRows);

                int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

                TableIn.AddColumn("rlnVoltage", "200.0");
                TableIn.AddColumn("rlnSphericalAberration", "2.7");
                TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
                TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
                TableIn.AddColumn("rlnMagnification", "10000");

                var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
                {
                    ColumnStackNames = TableIn.GetColumn("rlnImageName");
                    for (int r = 0; r < TableIn.RowCount; r++)
                    {
                        string s = ColumnStackNames[r];
                        int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
                        string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
                        AllParticleAddresses[r] = (ID, Name);
                    }
                }
                CTF[] AllParticleCTF = TableIn.GetRelionCTF();
                int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

                ParameterizedThreadStart ReloadLambda = (par) =>
                {
                    GPU.SetDevice(ProcessingDevice);

                    Random ReloadRand = new Random((int)par);
                    RandomNormal ReloadRandN = new RandomNormal((int)par * 100);
                    bool OwnBatchUsed = true;

                    Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                    Image TImagesReference = new Image(new int3(Dim, Dim, BatchSize));
                    Image TImagesData = new Image(new int3(Dim, Dim, BatchSize));
                    Image TImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);
                    Image TImagesDiff = new Image(new int3(1, 1, BatchSize));
                    Image TImagesDiffNaive = new Image(new int3(1, 1, BatchSize));
                    Image TImagesDataClean = new Image(new int3(Dim, Dim, BatchSize));
                    Image TImagesDataCleanFT = new Image(new int3(Dim, Dim, BatchSize), true, true);

                    float[] TCorrNaive = null;

                    Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);
                    Image ProjFT = new Image(new int3(Dim, Dim, BatchSize), true, true);
                    Image Proj = new Image(new int3(Dim, Dim, BatchSize));

                    Image Ref = Image.FromFile(Path.Combine(WorkingDirectory, "c4_scaled.mrc"));
                    Ref.MaskSpherically(Dim - 32, 10, true);
                    Projector RefProjector = new Projector(Ref, 2);
                    Ref.Dispose();

                    int PlanBack = 0, PlanForw = 0;
                    PlanForw = GPU.CreateFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);

                    while (true)
                    {
                        //ReloadBlock.WaitOne();
                        // If this thread succeeded at pushing its previously loaded batch to processing
                        if (OwnBatchUsed)
                        {
                            int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());
                            float3[] Angles = Helper.ArrayOfFunction(v => new float3((float)(ReloadRand.NextDouble() * 2 * Math.PI),
                                                                                     (float)(ReloadRand.NextDouble() * 2 * Math.PI),
                                                                                     (float)(ReloadRand.NextDouble() * 2 * Math.PI)), BatchSize);

                            {

                                RefProjector.Project(new int2(Dim), Angles, ProjFT);

                                GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                                CTFCoords.GetDevice(Intent.Read),
                                                IntPtr.Zero,
                                                (uint)CTFCoords.ElementsSliceComplex,
                                                Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                                false,
                                                (uint)BatchSize);

                                ProjFT.Multiply(TImagesCTF);
                                ProjFT.Multiply(TImagesCTF);

                                ProjFT.ShiftSlices(Helper.ArrayOfFunction(v => new float3(ReloadRandN.NextSingle(Dim / 2, 0),
                                                                                          ReloadRandN.NextSingle(Dim / 2, 0),
                                                                                          ReloadRandN.NextSingle(Dim / 2, 0)), BatchSize));

                                GPU.IFFT(ProjFT.GetDevice(Intent.Read),
                                         TImagesReference.GetDevice(Intent.Write),
                                         new int3(Dim, Dim, 1),
                                         (uint)BatchSize,
                                         PlanBack,
                                         false);
                                TImagesReference.Normalize();
                            }

                            {
                                //int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());

                                RefProjector.Project(new int2(Dim),
                                                     Helper.ArrayOfFunction(i => new float3(ReloadRandN.NextSingle(0, 0) * Helper.ToRad,
                                                                                            ReloadRandN.NextSingle(0, 0) * Helper.ToRad,
                                                                                            ReloadRandN.NextSingle(0, 0) * Helper.ToRad) + Angles[i], BatchSize),
                                                     ProjFT);

                                GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                                CTFCoords.GetDevice(Intent.Read),
                                                IntPtr.Zero,
                                                (uint)CTFCoords.ElementsSliceComplex,
                                                Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                                false,
                                                (uint)BatchSize);

                                ProjFT.Multiply(TImagesCTF);

                                float3[] Shifts = Helper.ArrayOfFunction(v => new float3(ReloadRandN.NextSingle(0, 1),
                                                                                         ReloadRandN.NextSingle(0, 1),
                                                                                         0), 
                                                                         BatchSize);

                                ProjFT.ShiftSlices(Shifts.Select(v => v + new float3(Dim / 2, Dim / 2, 0)).ToArray());

                                //GPU.CopyDeviceToDevice(ProjFT.GetDevice(Intent.Read), TImagesDataCleanFT.GetDevice(Intent.Write), ProjFT.ElementsReal);
                                //{
                                //    TImagesDataCleanFT.Multiply(TImagesCTF);
                                //    GPU.IFFT(TImagesDataCleanFT.GetDevice(Intent.Read),
                                //            TImagesDataClean.GetDevice(Intent.Write),
                                //            new int3(Dim, Dim, 1),
                                //            (uint)BatchSize,
                                //            PlanBack,
                                //            false);
                                //    TImagesDataClean.Normalize();

                                //    TImagesDataClean.Multiply(TImagesReference);
                                //    //TImagesDataClean.Abs();
                                //    //TImagesDataClean.WriteMRC("d_diff.mrc", true);
                                //    GPU.ReduceMean(TImagesDataClean.GetDevice(Intent.Read),
                                //                    TImagesDiff.GetDevice(Intent.Write),
                                //                    1,
                                //                    (uint)TImagesDataClean.ElementsSliceReal,
                                //                    (uint)BatchSize);
                                //    TImagesDiff.Multiply(1);
                                //    //TImagesDiff.WriteMRC("d_diff_reduced.mrc", true);
                                //}

                                GPU.IFFT(ProjFT.GetDevice(Intent.Read),
                                         Proj.GetDevice(Intent.Write),
                                         new int3(Dim, Dim, 1),
                                         (uint)BatchSize,
                                         PlanBack,
                                         false);
                                Proj.Multiply(1f);

                                Image Prediction = null;
                                GeneratorModel.Predict(Proj, TImagesCTF, out Prediction);

                                GPU.FFT(Prediction.GetDevice(Intent.Read),
                                        ProjFT.GetDevice(Intent.Write),
                                        new int3(Dim, Dim, 1),
                                        (uint)BatchSize,
                                        PlanForw);

                                ProjFT.Multiply(TImagesCTF);

                                GPU.IFFT(ProjFT.GetDevice(Intent.Read),
                                         TImagesData.GetDevice(Intent.Write),
                                         new int3(Dim, Dim, 1),
                                         (uint)BatchSize,
                                         PlanBack,
                                         true);
                                TImagesData.Normalize();

                                TImagesDiff.TransformValues((x, y, z, v) => Shifts[z].Length());// MathHelper.Gauss(Shifts[z].Length(), 0, 1.4f));

                                //{
                                //    GPU.MultiplySlices(TImagesReference.GetDevice(Intent.Read),
                                //                       TImagesData.GetDevice(Intent.Read),
                                //                       TImagesDataClean.GetDevice(Intent.Write),
                                //                       TImagesReference.ElementsReal,
                                //                       1);
                                //    //TImagesDataClean.Abs();
                                //    GPU.ReduceMean(TImagesDataClean.GetDevice(Intent.Read),
                                //                    TImagesDiffNaive.GetDevice(Intent.Write),
                                //                    1,
                                //                    (uint)TImagesDataClean.ElementsSliceReal,
                                //                    (uint)BatchSize);
                                //    TImagesDiffNaive.Multiply(1);
                                //    //TImagesDiffNaive.Subtract(TImagesDiff);
                                //    //TImagesDiffNaive.Multiply(TImagesDiffNaive);

                                //    TCorrNaive = Helper.Combine(TImagesDiffNaive.GetHost(Intent.Read));
                                //}
                            }

                            OwnBatchUsed = false;
                        }

                        ReloadBlock.WaitOne();
                        // If previously pushed batch from any thread has already been processed
                        if (HasBeenProcessed)
                        {
                            GPU.CopyDeviceToDevice(TImagesReference.GetDevice(Intent.Read), ImagesReference.GetDevice(Intent.Write), ImagesReference.ElementsReal);
                            GPU.CopyDeviceToDevice(TImagesData.GetDevice(Intent.Read), ImagesData.GetDevice(Intent.Write), ImagesData.ElementsReal);
                            GPU.CopyDeviceToDevice(TImagesDiff.GetDevice(Intent.Read), ImagesDiff.GetDevice(Intent.Write), ImagesDiff.ElementsReal);

                            //CorrNaive = TCorrNaive.ToArray();

                            OwnBatchUsed = true;
                            HasBeenProcessed = false;
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
                List<ObservablePoint> LossPointsNaive = new List<ObservablePoint>();

                long IterationsDone = 0;
                int PlotEvery = 20;
                List<float> AllLosses = new List<float>();

                while (true)
                {
                    if (HasBeenProcessed)
                        continue;

                    ReloadBlock.WaitOne();

                    float[] Loss = null;
                    float[] Prediction = null;
                    Image DebugReference = null;
                    Image DebugData = null;

                    {

                        float CurrentLearningRate = 0;
                        Dispatcher.Invoke(() => CurrentLearningRate = (float)LearningRate);

                        TrainModel.Train(ImagesReference,
                                                  ImagesData,
                                                  ImagesDiff,
                                                  CurrentLearningRate,
                                                  out Prediction,
                                                  out DebugReference,
                                                  out DebugData,
                                                  out Loss);

                        AllLosses.Add(Loss[0]);

                        HasBeenProcessed = true;
                    }

                    if (IterationsDone % PlotEvery == 0)
                    {
                        float[] CorrReal = Helper.Combine(ImagesDiff.GetHost(Intent.Read));

                        WriteToLog($"{MathHelper.Mean(AllLosses):F4}, " +
                                   $"{MathHelper.CrossCorrelateNormalized(Prediction, CorrReal):F4}");

                        LossPointsReal.Clear();
                        LossPointsReal.AddRange(CorrReal.Select((v, i) => new ObservablePoint(i, v)));
                        Dispatcher.Invoke(() => SeriesLossReal.Values = new ChartValues<ObservablePoint>(LossPointsReal));

                        LossPointsFake.Clear();
                        LossPointsFake.AddRange(Prediction.Select((v, i) => new ObservablePoint(i, v)));
                        Dispatcher.Invoke(() => SeriesLossFake.Values = new ChartValues<ObservablePoint>(LossPointsFake));

                        //LossPointsNaive.Clear();
                        //LossPointsNaive.AddRange(CorrNaive.Select((v, i) => new ObservablePoint(i, v)));
                        //Dispatcher.Invoke(() => SeriesLossNaive.Values = new ChartValues<ObservablePoint>(LossPointsNaive));

                        Func<float[], ImageSource> MakeImage = (data) =>
                        {
                            float[] OneSlice = data.ToArray();
                            float2 GlobalMeanStd = MathHelper.MeanAndStd(OneSlice);

                            byte[] BytesXY = new byte[OneSlice.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSlice[y * Dim + x] - GlobalMeanStd.X) / GlobalMeanStd.Y;
                                    Value = (Value + 4f) / 8f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceImage.Freeze();

                            return SliceImage;
                        };

                        {
                            ImageSource SliceImage = MakeImage(ImagesReference.GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImageSource.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(ImagesData.GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImageTarget.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(DebugReference.GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImageAverage.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(DebugData.GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImagePrediction.Source = SliceImage);
                        }

                        DebugData.WriteMRC("d_gen.mrc", true);

                        if (ShouldSaveModel)
                        {
                            ShouldSaveModel = false;

                            TrainModel.Save(WorkingDirectory + @"DistanceNet_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt");
                            Thread.Sleep(10000);

                            Dispatcher.Invoke(() => ButtonSave.IsEnabled = true);
                        }
                    }

                    if (AllLosses.Count >= PlotEvery)
                        AllLosses.RemoveAt(0);

                    IterationsDone++;
                    Dispatcher.Invoke(() => TextCoverage.Text = $"{IterationsDone} iterations done");

                    if ((IterationsDone + 1) % 5000 == 0)
                        Dispatcher.Invoke(() => LearningRate /= 2);

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

        private void ButtonTest_OnClick_old(object sender, RoutedEventArgs e)
        {
            //WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
            //ParticleWGAN TrainModel = new ParticleWGAN(new int2(Dim), 32, new[] { 0 }, BatchSize);
            //TrainModel.Load(WorkingDirectory + "ParticleWGAN_20210108_002634.pt");
            //WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

            //GPU.SetDevice(ProcessingDevice);

            //Image[] ImagesReal = null;
            //Image[] ImagesFake = null;
            //Image[] ImagesCTF = null;

            //Semaphore ReloadBlock = new Semaphore(1, 1);
            //bool HasBeenProcessed = true;

            //Star TableIn = new Star(@"E:\particleWGAN\c4_coords.star", "particles");

            //string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            //HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
            //UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(@"E:\particleWGAN\real", s)));
            //int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
            //TableIn = TableIn.CreateSubset(KeepRows);

            //int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

            //TableIn.AddColumn("rlnVoltage", "200.0");
            //TableIn.AddColumn("rlnSphericalAberration", "2.7");
            //TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
            //TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
            //TableIn.AddColumn("rlnMagnification", "10000");

            //var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
            //{
            //    ColumnStackNames = TableIn.GetColumn("rlnImageName");
            //    for (int r = 0; r < TableIn.RowCount; r++)
            //    {
            //        string s = ColumnStackNames[r];
            //        int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
            //        string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
            //        AllParticleAddresses[r] = (ID, Name);
            //    }
            //}
            //CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            //int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            //ParameterizedThreadStart ReloadLambda = (par) =>
            //{
            //    GPU.SetDevice(ProcessingDevice);

            //    Random ReloadRand = new Random((int)par);
            //    bool OwnBatchUsed = false;

            //    Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

            //    Image TImagesFake = new Image(new int3(Dim, Dim, BatchSize));
            //    Image TImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);

            //    Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);

            //    int PlanForw = 0, PlanBack = 0;
            //    if (DimRaw != Dim)
            //    {
            //        PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
            //        PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
            //    }

            //    int NDone = 0;

            //    while (true)
            //    {
            //        // If this thread succeeded at pushing its previously loaded batch to processing
            //        int[] SubsetIDs = Helper.ArrayOfConstant(0, BatchSize);

            //        float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

            //        for (int b = 0; b < BatchSize; b++)
            //        {
            //            int id = SubsetIDs[b];
            //            IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, DirectoryFake, AllParticleAddresses[id].name),
            //                                    new int2(1),
            //                                    0,
            //                                    typeof(float),
            //                                    new[] { AllParticleAddresses[id].id },
            //                                    null,
            //                                    new[] { LoadStackData[b] });
            //        }

            //        if (Dim == DimRaw)
            //            GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
            //                                    TImagesFake.GetDevice(Intent.Write),
            //                                    LoadStack.ElementsReal);
            //        else
            //            GPU.Scale(LoadStack.GetDevice(Intent.Read),
            //                        TImagesFake.GetDevice(Intent.Write),
            //                        LoadStack.Dims.Slice(),
            //                        TImagesFake.Dims.Slice(),
            //                        (uint)BatchSize,
            //                        PlanForw,
            //                        PlanBack,
            //                        IntPtr.Zero,
            //                        IntPtr.Zero);

            //        GPU.CheckGPUExceptions();

            //        GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
            //                        CTFCoords.GetDevice(Intent.Read),
            //                        IntPtr.Zero,
            //                        (uint)CTFCoords.ElementsSliceComplex,
            //                        Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
            //                        false,
            //                        (uint)BatchSize);

            //        Image Predicted = null;

            //        TrainModel.Predict(TImagesFake, TImagesCTF, out Predicted);

            //        Predicted.WriteMRC(Path.Combine(WorkingDirectory, "ganned", $"{NDone++}.mrc"), true);

            //        if (NDone == 100)
            //            break;
            //    }
            //};
            //Thread[] ReloadThreads = Helper.ArrayOfFunction(i => new Thread(ReloadLambda), NThreads);
            //for (int i = 0; i < NThreads; i++)
            //    ReloadThreads[i].Start(i);
        }

        private void ButtonTest_OnClick(object sender, RoutedEventArgs e)
        {
            WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");

            DistanceNet TrainModel = new DistanceNet(new int2(Dim), new[] { 1 }, BatchSize);
            TrainModel.Load(Path.Combine(WorkingDirectory, "DistanceNet_20210115_101304.pt"));

            ParticleWGAN GeneratorModel = new ParticleWGAN(new int2(Dim), 64, new[] { 2 }, BatchSize);
            GeneratorModel.Load(Path.Combine(WorkingDirectory, "ParticleWGAN_20210111_210604.pt"));

            WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

            GPU.SetDevice(ProcessingDevice);


            Semaphore ReloadBlock = new Semaphore(1, 1);
            bool HasBeenProcessed = true;

            Star TableIn = new Star(Path.Combine(WorkingDirectory, "c4_coords.star"), "particles");

            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
            UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryReal, s)));
            int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
            TableIn = TableIn.CreateSubset(KeepRows);

            int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

            TableIn.AddColumn("rlnVoltage", "200.0");
            TableIn.AddColumn("rlnSphericalAberration", "2.7");
            TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
            TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
            TableIn.AddColumn("rlnMagnification", "10000");

            var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
            {
                ColumnStackNames = TableIn.GetColumn("rlnImageName");
                for (int r = 0; r < TableIn.RowCount; r++)
                {
                    string s = ColumnStackNames[r];
                    int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
                    string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
                    AllParticleAddresses[r] = (ID, Name);
                }
            }
            CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            {
                GPU.SetDevice(ProcessingDevice);

                Random ReloadRand = new Random(1);
                RandomNormal ReloadRandN = new RandomNormal(1 * 100);
                bool OwnBatchUsed = true;

                Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                Image TImagesReference = new Image(new int3(Dim, Dim, BatchSize));
                Image TImagesData = new Image(new int3(Dim, Dim, BatchSize));
                Image TImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);
                Image TImagesDiff = new Image(new int3(1, 1, BatchSize));
                Image TImagesDiffNaive = new Image(new int3(1, 1, BatchSize));
                Image TImagesDataClean = new Image(new int3(Dim, Dim, BatchSize));
                Image TImagesDataCleanFT = new Image(new int3(Dim, Dim, BatchSize), true, true);

                float TLossNaive = 0;

                Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);
                Image ProjFT = new Image(new int3(Dim, Dim, BatchSize), true, true);
                Image Proj = new Image(new int3(Dim, Dim, BatchSize));

                Image Ref = Image.FromFile(Path.Combine(WorkingDirectory, "c4_scaled.mrc"));
                Ref.MaskSpherically(Dim - 32, 10, true);
                Projector RefProjector = new Projector(Ref, 2);
                Ref.Dispose();

                int PlanBack = 0, PlanForw = 0;
                PlanForw = GPU.CreateFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);

                while (true)
                {
                    Image ResultCorr = new Image(new int3(11, 11, BatchSize));

                    int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());

                    RefProjector.Project(new int2(Dim),
                                            Helper.ArrayOfFunction(v => new float3((float)(ReloadRand.NextDouble() * 2 * Math.PI),
                                                                                (float)(ReloadRand.NextDouble() * 2 * Math.PI),
                                                                                (float)(ReloadRand.NextDouble() * 2 * Math.PI)), BatchSize),
                                            ProjFT);

                    GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                    CTFCoords.GetDevice(Intent.Read),
                                    IntPtr.Zero,
                                    (uint)CTFCoords.ElementsSliceComplex,
                                    Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                    false,
                                    (uint)BatchSize);

                    ProjFT.Multiply(TImagesCTF);

                    ProjFT.ShiftSlices(Helper.ArrayOfFunction(v => new float3(ReloadRandN.NextSingle(Dim / 2, 0),
                                                                                ReloadRandN.NextSingle(Dim / 2, 0),
                                                                                ReloadRandN.NextSingle(Dim / 2, 0)), BatchSize));

                    GPU.CopyDeviceToDevice(ProjFT.GetDevice(Intent.Read), TImagesDataCleanFT.GetDevice(Intent.Write), ProjFT.ElementsReal);
                    {
                        TImagesDataCleanFT.Multiply(TImagesCTF);
                        GPU.IFFT(TImagesDataCleanFT.GetDevice(Intent.Read),
                                TImagesDataClean.GetDevice(Intent.Write),
                                new int3(Dim, Dim, 1),
                                (uint)BatchSize,
                                PlanBack,
                                false);
                        TImagesDataClean.Normalize();
                        TImagesDataClean.WriteMRC("d_dataclean.mrc", true);
                    }

                    GPU.IFFT(ProjFT.GetDevice(Intent.Read),
                                Proj.GetDevice(Intent.Write),
                                new int3(Dim, Dim, 1),
                                (uint)BatchSize,
                                PlanBack,
                                false);

                    Image Prediction = null;
                    GeneratorModel.Predict(Proj, TImagesCTF, out Prediction);

                    GPU.FFT(Prediction.GetDevice(Intent.Read),
                            ProjFT.GetDevice(Intent.Write),
                            new int3(Dim, Dim, 1),
                            (uint)BatchSize,
                            PlanForw);

                    ProjFT.Multiply(TImagesCTF);

                    GPU.IFFT(ProjFT.GetDevice(Intent.Read),
                                TImagesData.GetDevice(Intent.Write),
                                new int3(Dim, Dim, 1),
                                (uint)BatchSize,
                                PlanBack,
                                true);
                    TImagesData.Normalize();
                    TImagesData.WriteMRC("d_datanoisy.mrc", true);

                    for (int y = 0; y < ResultCorr.Dims.Y; y++)
                    {
                        for (int x = 0; x < ResultCorr.Dims.X; x++)
                        {
                            Image NoisyCopy = TImagesData.GetCopyGPU();
                            NoisyCopy.ShiftSlices(Helper.ArrayOfFunction(i => new float3(y - ResultCorr.Dims.Y / 2, x - ResultCorr.Dims.X / 2, 0)*0.5f, BatchSize));

                            float[] CorrPrediction = null;
                            Image DebugRef = null, DebugData = null;

                            TrainModel.Predict(TImagesDataClean,
                                               NoisyCopy,
                                               out CorrPrediction,
                                               out DebugRef,
                                               out DebugData);

                            float[][] ResultCorrData = ResultCorr.GetHost(Intent.ReadWrite);
                            for (int z = 0; z < BatchSize; z++)
                                ResultCorrData[z][y * ResultCorr.Dims.X + x] = CorrPrediction[z];

                            //NoisyCopy.Multiply(TImagesDataClean);
                            //CorrPrediction = NoisyCopy.GetHost(Intent.Read).Select(v => MathHelper.Mean(v)).ToArray();
                            //for (int z = 0; z < BatchSize; z++)
                            //    ResultCorrData[z][y * ResultCorr.Dims.X + x] = CorrPrediction[z];
                            NoisyCopy.Dispose();
                        }
                    }

                    ResultCorr.WriteMRC("d_resultcorr_naive.mrc", true);
                }
            };
        }
    }
}
