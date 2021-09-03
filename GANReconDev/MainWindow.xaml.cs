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
using TorchSharp;
using TorchSharp.NN;
using TorchSharp.Tensor;
using Warp;
using Warp.Headers;
using Warp.NNModels;
using Warp.Tools;

namespace ParticleWGANDev
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

        
        private void CheckSaveRecs_Checked(object sender, RoutedEventArgs e) { ShouldSaveRecs = true; }
        private void CheckSaveRecs_Unchecked(object sender, RoutedEventArgs e) { ShouldSaveRecs = false; }

        private bool ShouldSaveModel = false;
        private bool ShouldSaveRecs = false;


        private string WorkingDirectory = @"D:\GanRecon\";
        private string DirectoryReal = "particles";
        private string DirectoryFake = "sim";

        private int Dim = 64;

        private double LowPass = 1.0;

        private int BatchSize = 16;
        float Lambda = 0.6f;
        int DiscIters = 8;
        bool TrainGen = true;

        int NThreads = 3;
        int ProcessingDevice = 0;

        public MainWindow()
        {
            InitializeComponent();

            SliderLearningRate.DataContext = this;
        }

        private void ButtonStartNoise_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonStartNoise.IsEnabled = false;
            /*
            Task.Run(() =>
            {
                WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
                NoiseNet2DTorch TrainModel = new NoiseNet2DTorch(new int2(Dim), new[] { 1 }, BatchSize);
                WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

                GPU.SetDevice(ProcessingDevice);

                Image[] ImagesReal = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize)), DiscIters + 1);
                Image[] ImagesFake = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize)), DiscIters + 1);
                Image[] ImagesCTF = Helper.ArrayOfFunction(i => new Image(new int3(2 * Dim, 2 * Dim, BatchSize), true), DiscIters + 1);

                Semaphore ReloadBlock = new Semaphore(1, 1);
                bool HasBeenProcessed = true;

                Image inVol = Image.FromFile(Path.Combine(WorkingDirectory, "run_1k_unfil.mrc"));
                
                Star TableIn = new Star(Path.Combine(WorkingDirectory, "run_data.star"));

                string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
                HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
                UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryReal, s)));
                int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
                TableIn = TableIn.CreateSubset(KeepRows);

                int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

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
                float3[] AllParticleAngles = TableIn.GetRelionAngles();
                int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

                ParameterizedThreadStart ReloadLambda = (par) =>
                {
                    GPU.SetDevice(ProcessingDevice);

                    Random ReloadRand = new Random((int)par);
                    bool OwnBatchUsed = true;

                    Projector proj = new(inVol, 2);

                    Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                    Image[] TImagesReal = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize)), DiscIters + 1);
                    Image[] TImagesFake = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize)), DiscIters + 1);
                    Image[] TImagesCTF = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize), true), DiscIters + 1);

                    Image CTFCoords = CTF.GetCTFCoords(2*Dim, DimRaw);

                    int PlanForw = 0, PlanBack = 0;
                    if (DimRaw != Dim)
                    {
                        PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                        PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                    }

                    while (true)
                    {
                        // If this thread succeeded at pushing its previously loaded batch to processing
                        if (OwnBatchUsed)
                        {
                            for (int iterTrain = 0; iterTrain < DiscIters + 1; iterTrain++)
                            {
                                int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());

                                Image[] CopyTargets = { TImagesReal[iterTrain], TImagesFake[iterTrain] };
                                string[] Directories = { DirectoryReal, DirectoryFake };

                                // Read, and copy or rescale real and fake images from prepared stacks
                                for (int i = 0; i < 2; i++)
                                {
                                    float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

                                    for (int b = 0; b < BatchSize; b++)
                                    {
                                        int id = SubsetIDs[b];
                                        IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, Directories[i], AllParticleAddresses[id].name),
                                                                new int2(1),
                                                                0,
                                                                typeof(float),
                                                                new[] { AllParticleAddresses[id].id },
                                                                null,
                                                                new[] { LoadStackData[b] });
                                    }

                                    if (Dim == DimRaw)
                                        GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
                                                               CopyTargets[i].GetDevice(Intent.Write),
                                                               LoadStack.ElementsReal);
                                    else
                                        GPU.Scale(LoadStack.GetDevice(Intent.Read),
                                                  CopyTargets[i].GetDevice(Intent.Write),
                                                  LoadStack.Dims.Slice(),
                                                  CopyTargets[i].Dims.Slice(),
                                                  (uint)BatchSize,
                                                  PlanForw,
                                                  PlanBack,
                                                  IntPtr.Zero,
                                                  IntPtr.Zero);

                                    //CopyTargets[i].ShiftSlices(Helper.ArrayOfFunction(v => new float3(ReloadRand.Next(Dim), 
                                    //                                                                  ReloadRand.Next(Dim), 
                                    //                                                                  0), 
                                    //                           CopyTargets[i].Dims.Z));

                                    //GPU.CheckGPUExceptions();
                                }

                                GPU.CreateCTF(TImagesCTF[iterTrain].GetDevice(Intent.Write),
                                              CTFCoords.GetDevice(Intent.Read),
                                              IntPtr.Zero,
                                              (uint)CTFCoords.ElementsSliceComplex,
                                              Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                              false,
                                              (uint)BatchSize);
                            }

                            OwnBatchUsed = false;
                        }

                        ReloadBlock.WaitOne();
                        // If previously pushed batch from any thread has already been processed
                        if (HasBeenProcessed)
                        {
                            for (int discIter = 0; discIter < DiscIters + 1; discIter++)
                            {
                                GPU.CopyDeviceToDevice(TImagesReal[discIter].GetDevice(Intent.Read), ImagesReal[discIter].GetDevice(Intent.Write), TImagesReal[discIter].ElementsReal);
                                GPU.CopyDeviceToDevice(TImagesFake[discIter].GetDevice(Intent.Read), ImagesFake[discIter].GetDevice(Intent.Write), TImagesFake[discIter].ElementsReal);
                                GPU.CopyDeviceToDevice(TImagesCTF[discIter].GetDevice(Intent.Read), ImagesCTF[discIter].GetDevice(Intent.Write), TImagesCTF[discIter].ElementsReal);
                            }

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

                long IterationsDone = 0;

                while (true)
                {
                    if (HasBeenProcessed)
                        continue;

                    ReloadBlock.WaitOne();

                    List<float> AllLossesReal = new List<float>();
                    List<float> AllLossesFake = new List<float>();
                    float[] Loss = null;
                    float[] LossReal = null;
                    float[] LossFake = null;
                    Image Prediction = null;
                    Image PredictionDeconv = null;
                    float[] SourceData = null;
                    float[] TargetData = null;
                    float[] AverageData = null;

                    {

                        float CurrentLearningRate = 0;
                        Dispatcher.Invoke(() => CurrentLearningRate = (float)LearningRate);

                        for (int iterDisc = 0; iterDisc < DiscIters; iterDisc++)
                        {
                            TrainModel.Train(ImagesReal[iterDisc],
                                                          ImagesFake[iterDisc],
                                                          ImagesCTF[iterDisc],
                                                          CurrentLearningRate,
                                                          Lambda,
                                                          out Prediction,
                                                          out Loss,
                                                          out LossReal,
                                                          out LossFake);

                            AllLossesReal.Add(LossReal[0]);
                            AllLossesFake.Add(LossFake[0]);
                        }

                        TrainModel.TrainGenerator(ImagesFake[DiscIters],
                                                  ImagesCTF[DiscIters],
                                                  CurrentLearningRate,
                                                  out Prediction,
                                                  out Loss);

                        HasBeenProcessed = true;
                    }

                    if (IterationsDone % 10 == 0)
                    {
                        WriteToLog($"{MathHelper.Mean(AllLossesReal):F4}, {MathHelper.Mean(AllLossesFake):F4}");

                        //LossPointsReal.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLossesReal)));
                        //Dispatcher.Invoke(() => SeriesLossReal.Values = new ChartValues<ObservablePoint>(LossPointsReal));

                        LossPointsFake.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLossesFake) - MathHelper.Mean(AllLossesReal)));
                        Dispatcher.Invoke(() => SeriesLossFake.Values = new ChartValues<ObservablePoint>(LossPointsFake));

                        float2 GlobalMeanStd = MathHelper.MeanAndStd(ImagesReal[0].GetHost(Intent.Read)[0]);

                        Func<float[], ImageSource> MakeImage = (data) =>
                        {
                            float[] OneSlice = data.ToArray();
                            GlobalMeanStd = MathHelper.MeanAndStd(OneSlice);

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
                            ImageSource SliceImage = MakeImage(ImagesReal[DiscIters].GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImageSource.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(ImagesFake[DiscIters].GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImageTarget.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(Prediction.GetHost(Intent.Read)[0]);
                            Dispatcher.Invoke(() => ImageAverage.Source = SliceImage);
                        }

                        Prediction.WriteMRC("d_gen.mrc", true);

                        if (ShouldSaveModel)
                        {
                            ShouldSaveModel = false;

                            TrainModel.Save(WorkingDirectory + @"ParticleWGAN_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt");
                            Thread.Sleep(10000);

                            Dispatcher.Invoke(() => ButtonSave.IsEnabled = true);
                        }

                        AllLossesReal.Clear();
                        AllLossesFake.Clear();
                    }

                    IterationsDone++;
                    Dispatcher.Invoke(() => TextCoverage.Text = $"{IterationsDone} iterations done");

                    if ((IterationsDone + 1) % 1400 == 0)
                        Dispatcher.Invoke(() => LearningRate /= 2);

                    ReloadBlock.Release();
                }
            });*/
        }

        private void ButtonStartParticle_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonStartParticle.IsEnabled = false;

            Task.Run(() =>
            {
                
                WriteToLog("Loading model... (" + GPU.GetFreeMemory(1) + " MB free)");

                //ParticleWGAN TrainModel = new ParticleWGAN(new int2(Dim), 32, new[] { 1 }, BatchSize);
                Image refVolume = Image.FromFile(Path.Combine(WorkingDirectory, "run_1k_unfil.mrc")).AsScaled(new int3(Dim));
                ReconstructionWGAN TrainModel = new ReconstructionWGAN(new int2(Dim), 10, new[] { 1 }, BatchSize);
                //TrainModel.Load(@"D:\GANRecon\ParticleWGAN_SN_20210830_170236.pt");
                WriteToLog("Done. (" + GPU.GetFreeMemory(1) + " MB free)");

                GPU.SetDevice(ProcessingDevice);

                Image[] ImagesReal = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize)), DiscIters + 1);
                Image[] ImagesCTF = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize), true), DiscIters + 1);
                float[][] ImagesAngles = Helper.ArrayOfFunction(i => new float[BatchSize*3], DiscIters + 1);

                Semaphore ReloadBlock = new Semaphore(1, 1);
                bool HasBeenProcessed = true;

                Star TableIn = new Star(Path.Combine(WorkingDirectory, "run_data.star"));

                
                
                Random rand = new Random();


                string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
                HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
                UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryReal, s)));
                int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
                TableIn = TableIn.CreateSubset(KeepRows);

                int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

                //TableIn.AddColumn("rlnVoltage", "200.0");
                //TableIn.AddColumn("rlnSphericalAberration", "2.7");
                //TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
                //TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
                //TableIn.AddColumn("rlnMagnification", "10000");

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
                float3[] AllParticleAngles = TableIn.GetRelionAngles().Select(s=>s*Helper.ToRad).ToArray();
                int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);
                float3[] RandomParticleAngles = Helper.GetHealpixAngles(3);
                ParameterizedThreadStart ReloadLambda = (par) =>
                {
                    GPU.SetDevice(ProcessingDevice);
                    Image TrefVolume = Image.FromFile(Path.Combine(WorkingDirectory, "run_1k_unfil.mrc")).AsScaled(new int3(Dim));
                    var tensorRefVolume = TensorExtensionMethods.ToTorchTensor(TrefVolume.GetHostContinuousCopy(), new long[] { 1, 1, Dim, Dim, Dim }).ToDevice(TorchSharp.DeviceType.CUDA, ProcessingDevice);
                    ReconstructionWGANGenerator gen = Modules.ReconstructionWGANGenerator(tensorRefVolume, Dim, 10);
                    gen.ToCuda(ProcessingDevice);
                    var TensorAngles = Float32Tensor.Zeros(new long[] { BatchSize, 3 }, DeviceType.CUDA, ProcessingDevice);

                    //Projector refProjector = new Projector(refVolume, 2);

                    Random ReloadRand = new Random((int)par);
                    bool OwnBatchUsed = true;

                    Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                    Image[] TImagesReal = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize)), DiscIters + 1);
                    Image[] TImagesCTF = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, BatchSize), true), DiscIters + 1);
                    float[][] TImagesAngles = Helper.ArrayOfFunction(i => new float[BatchSize*3], DiscIters + 1);
                    Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);

                    int PlanForw = 0, PlanBack = 0;
                    if (DimRaw != Dim)
                    {
                        PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                        PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                    }

                    while (true)
                    {
                        // If this thread succeeded at pushing its previously loaded batch to processing
                        if (OwnBatchUsed)
                        {
                            for (int iterTrain = 0; iterTrain < DiscIters + 1; iterTrain++)
                            {
                                int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());
                                int[] AngleIds = Helper.ArrayOfFunction(i => ReloadRand.Next(0, RandomParticleAngles.Length), BatchSize);

                                // Read, and copy or rescale real and fake images from prepared stacks
                                float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

                                for (int b = 0; b < BatchSize; b++)
                                {
                                    int id = SubsetIDs[b];
                                    IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, DirectoryReal, AllParticleAddresses[id].name),
                                                            new int2(1),
                                                            0,
                                                            typeof(float),
                                                            new[] { AllParticleAddresses[id].id },
                                                            null,
                                                            new[] { LoadStackData[b] });
                                }

                                if (Dim == DimRaw)
                                    GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
                                                            TImagesReal[iterTrain].GetDevice(Intent.Write),
                                                            LoadStack.ElementsReal);
                                else
                                    GPU.Scale(LoadStack.GetDevice(Intent.Read),
                                                TImagesReal[iterTrain].GetDevice(Intent.Write),
                                                LoadStack.Dims.Slice(),
                                                TImagesReal[iterTrain].Dims.Slice(),
                                                (uint)BatchSize,
                                                PlanForw,
                                                PlanBack,
                                                IntPtr.Zero,
                                                IntPtr.Zero);
                                //TensorAngles.RandomNInPlace(TensorAngles.Shape);
                                //TensorAngles *= 2 * Math.PI;

                                float3[] theseAngles = Helper.IndexedSubset(RandomParticleAngles, AngleIds);
                                TImagesAngles[iterTrain] = Helper.ToInterleaved(theseAngles);
                                //GPU.CopyHostToDevice(TImagesAngles[iterTrain], TensorAngles.DataPtr(), TImagesAngles[iterTrain].Length);
                                //using (var projected = gen.ForwardParticle(Float32Tensor.Zeros(new long[] { 1 }), TensorAngles, false, 0.0d)) 
                                //{
                                //    GPU.CopyDeviceToDevice(projected.DataPtr(), TImagesReal[iterTrain].GetDevice(Intent.Write), TImagesReal[iterTrain].ElementsReal);
                                //    //TImagesReal[iterTrain].WriteMRC($@"{WorkingDirectory}\Thread_{par}_TImagesReal[{iterTrain}]_afterProj.mrc", true);
                                //}
                                GPU.CheckGPUExceptions();
                                //TImagesReal[iterTrain] = refProjector.ProjectToRealspace(new int2(Dim), Helper.ArrayOfFunction(i => 
                                //        new float3((float)rand.NextDouble(), (float)rand.NextDouble(), (float)rand.NextDouble()) * ((float)Math.PI * 2), BatchSize));

                                GPU.CreateCTF(TImagesCTF[iterTrain].GetDevice(Intent.Write),
                                              CTFCoords.GetDevice(Intent.Read),
                                              IntPtr.Zero,
                                              (uint)CTFCoords.ElementsSliceComplex,
                                              Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                              false,
                                              (uint)BatchSize);

                                Image thisCTFSign = TImagesCTF[iterTrain].GetCopy();
                                thisCTFSign.Sign();
                                TImagesCTF[iterTrain].Multiply(thisCTFSign);
                                Image fft = TImagesReal[iterTrain].AsFFT();
                                TImagesReal[iterTrain].Dispose();
                                fft.Multiply(thisCTFSign);
                                TImagesReal[iterTrain] = fft.AsIFFT();
                                fft.Dispose();
                                thisCTFSign.Dispose();
                                //TImagesCTF[iterTrain].Multiply(TImagesCTF[iterTrain]);

                                GPU.NormParticles(TImagesReal[iterTrain].GetDevice(Intent.Read),
                                                  TImagesReal[iterTrain].GetDevice(Intent.Write),
                                                  TImagesReal[iterTrain].Dims.Slice(),
                                                  (uint)Dim / 4,
                                                  false,
                                                  (uint)BatchSize);
                                
                                //TImagesReal[iterTrain].WriteMRC($@"{WorkingDirectory}\Thread_{par}_TImagesReal[{iterTrain}]_afterNorm.mrc", true);



                                //TImagesReal[iterTrain].Bandpass(0, (float)LowPass, false, 0.05f);
                                TImagesReal[iterTrain].MaskSpherically(Dim / 2, Dim / 8, false);
                            }

                            OwnBatchUsed = false;
                        }

                        ReloadBlock.WaitOne();
                        // If previously pushed batch from any thread has already been processed
                        if (HasBeenProcessed)
                        {
                            for (int discIter = 0; discIter < DiscIters + 1; discIter++)
                            {
                                GPU.CopyDeviceToDevice(TImagesReal[discIter].GetDevice(Intent.Read), ImagesReal[discIter].GetDevice(Intent.Write), TImagesReal[discIter].ElementsReal);
                                GPU.CopyDeviceToDevice(TImagesCTF[discIter].GetDevice(Intent.Read), ImagesCTF[discIter].GetDevice(Intent.Write), TImagesCTF[discIter].ElementsReal);
                                TImagesAngles[discIter].CopyTo(ImagesAngles[discIter], 0);
                                GPU.CheckGPUExceptions();
                            }

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

                long IterationsDone = 0;

                while (true)
                {
                    if (HasBeenProcessed)
                        continue;

                    ReloadBlock.WaitOne();
                    
                    List<float> AllLossesReal = new List<float>();
                    List<float> AllLossesFake = new List<float>();
                    float[] Loss = null;
                    float[] LossReal = null;
                    float[] LossFake = null;
                    Image PredictionDisc = null;
                    Image PredictionGen = null;
                    Image PredictionGenNoisy = null;
                    float[] SourceData = null;
                    float[] TargetData = null;
                    float[] AverageData = null;
                    var allImages = Image.getObjectIDs();
                    {

                        float CurrentLearningRate = 0;
                        Dispatcher.Invoke(() => CurrentLearningRate = (float)LearningRate);

                        for (int iterDisc = 0; iterDisc < DiscIters; iterDisc++)
                        {
                            TrainModel.TrainDiscriminatorParticle(ImagesAngles[iterDisc], 
                                                                  ImagesReal[iterDisc],
                                                                  ImagesCTF[iterDisc],
                                                                  CurrentLearningRate,
                                                                  Lambda,
                                                                  out PredictionDisc,
                                                                  out Loss,
                                                                  out LossReal,
                                                                  out LossFake);
                            
                            AllLossesReal.Add(LossReal[0]);
                            AllLossesFake.Add(LossFake[0]);
                        }
                        
                        if (TrainGen)
                            TrainModel.TrainGeneratorParticle( ImagesAngles[DiscIters], ImagesCTF[DiscIters], ImagesReal[DiscIters],
                                                              CurrentLearningRate,
                                                              out PredictionGen,
                                                              out PredictionGenNoisy,
                                                              out Loss);
                         HasBeenProcessed = true;
                    }

                    if (IterationsDone % 10 == 0)
                    {
                        WriteToLog($"{MathHelper.Mean(AllLossesReal):F4}, {MathHelper.Mean(AllLossesFake):F4}");

                        LossPointsReal.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLossesReal)));
                        Dispatcher.Invoke(() => SeriesLossReal.Values = new ChartValues<ObservablePoint>(LossPointsReal));

                        LossPointsFake.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLossesFake)));
                        Dispatcher.Invoke(() => SeriesLossFake.Values = new ChartValues<ObservablePoint>(LossPointsFake));

                        float2 GlobalMeanStd = MathHelper.MeanAndStd(ImagesReal[0].GetHost(Intent.Read)[0]);

                        Func<float[], float2, ImageSource> MakeImage = (data, stat) =>
                        {
                            float[] OneSlice = data.ToArray();
                            GlobalMeanStd = MathHelper.MeanAndStd(OneSlice);

                            byte[] BytesXY = new byte[OneSlice.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSlice[y * Dim + x] - stat.X) / stat.Y;
                                    Value = (Value + 4f) / 8f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceImage.Freeze();

                            return SliceImage;
                        };

                        float[] all_data = new List<float>().Concat(ImagesReal[DiscIters].GetHost(Intent.Read)[0]).Concat(PredictionGenNoisy.GetHost(Intent.Read)[0]).Concat(PredictionGen.GetHost(Intent.Read)[0]).ToArray();
                        float2 stat = MathHelper.MeanAndStd(all_data);
                        {
                            ImageSource SliceImage = MakeImage(ImagesReal[DiscIters].GetHost(Intent.Read)[0], stat);
                            Dispatcher.Invoke(() => ImageSource.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(PredictionGenNoisy.GetHost(Intent.Read)[0], stat);
                            Dispatcher.Invoke(() => ImageTarget.Source = SliceImage);
                        }

                        {
                            ImageSource SliceImage = MakeImage(PredictionGen.GetHost(Intent.Read)[0], stat);
                            Dispatcher.Invoke(() => ImageAverage.Source = SliceImage);
                        }

                        PredictionGen.WriteMRC("d_gen.mrc", true);

                        if (ShouldSaveModel)
                        {
                            ShouldSaveModel = false;
                            string datestring = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                            TrainModel.Save(WorkingDirectory + @"ParticleWGAN_SN_" + datestring + ".pt");
                            var imageVolume = TrainModel.getVolume();
                            imageVolume.WriteMRC(WorkingDirectory + @"ParticleWGAN_SN_" + datestring + ".mrc", true);
                            imageVolume.Dispose();
                            Thread.Sleep(10000);

                            Dispatcher.Invoke(() => ButtonSave.IsEnabled = true);

                            if (ShouldSaveRecs)
                            {
                                Image.Stack(ImagesReal).WriteMRC(WorkingDirectory + @"ParticleWGAN_SN_" + datestring + "_real.mrc", true);
                                Image.Stack(ImagesCTF).WriteMRC(WorkingDirectory + @"ParticleWGAN_SN_" + datestring + "_ctf.mrc", true);
                                PredictionGen.WriteMRC(WorkingDirectory + @"ParticleWGAN_SN_" + datestring + "_prediction.mrc", true);
                                PredictionGenNoisy.WriteMRC(WorkingDirectory + @"ParticleWGAN_SN_" + datestring + "_predictionNoisy.mrc", true);
                            }

                        }

                        AllLossesReal.Clear();
                        AllLossesFake.Clear();
                    }
                    
                    IterationsDone++;
                    Dispatcher.Invoke(() => TextCoverage.Text = $"{IterationsDone} iterations done");

                    if ((IterationsDone + 1) % 1400 == 0)
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
            WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
            ParticleWGAN TrainModel = new ParticleWGAN(new int2(Dim), 32, new[] { 0 }, BatchSize);
            TrainModel.Load(WorkingDirectory + "ParticleWGAN_20210108_002634.pt");
            WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

            GPU.SetDevice(ProcessingDevice);

            Image[] ImagesReal = null;
            Image[] ImagesFake = null;
            Image[] ImagesCTF = null;

            Semaphore ReloadBlock = new Semaphore(1, 1);
            bool HasBeenProcessed = true;

            Star TableIn = new Star(@"E:\particleWGAN\c4_coords.star", "particles");

            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
            UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(@"E:\particleWGAN\real", s)));
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
                bool OwnBatchUsed = false;

                Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                Image TImagesFake = new Image(new int3(Dim, Dim, BatchSize));
                Image TImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);

                Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);

                int PlanForw = 0, PlanBack = 0;
                if (DimRaw != Dim)
                {
                    PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                }

                int NDone = 0;

                while (true)
                {
                    // If this thread succeeded at pushing its previously loaded batch to processing
                    int[] SubsetIDs = Helper.ArrayOfConstant(0, BatchSize);

                    float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

                    for (int b = 0; b < BatchSize; b++)
                    {
                        int id = SubsetIDs[b];
                        IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, DirectoryFake, AllParticleAddresses[id].name),
                                                new int2(1),
                                                0,
                                                typeof(float),
                                                new[] { AllParticleAddresses[id].id },
                                                null,
                                                new[] { LoadStackData[b] });
                    }

                    if (Dim == DimRaw)
                        GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
                                                TImagesFake.GetDevice(Intent.Write),
                                                LoadStack.ElementsReal);
                    else
                        GPU.Scale(LoadStack.GetDevice(Intent.Read),
                                    TImagesFake.GetDevice(Intent.Write),
                                    LoadStack.Dims.Slice(),
                                    TImagesFake.Dims.Slice(),
                                    (uint)BatchSize,
                                    PlanForw,
                                    PlanBack,
                                    IntPtr.Zero,
                                    IntPtr.Zero);

                    GPU.CheckGPUExceptions();

                    GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                    CTFCoords.GetDevice(Intent.Read),
                                    IntPtr.Zero,
                                    (uint)CTFCoords.ElementsSliceComplex,
                                    Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                    false,
                                    (uint)BatchSize);

                    Image Predicted = null;

                    TrainModel.Predict(TImagesFake, TImagesCTF, out Predicted);

                    Predicted.WriteMRC(Path.Combine(WorkingDirectory, "ganned", $"{NDone++}.mrc"), true);

                    if (NDone == 100)
                        break;
                }
            };
            Thread[] ReloadThreads = Helper.ArrayOfFunction(i => new Thread(ReloadLambda), NThreads);
            for (int i = 0; i < NThreads; i++)
                ReloadThreads[i].Start(i);
        }

        private void ButtonTest_OnClick(object sender, RoutedEventArgs e)
        {
            NThreads = 1;

            WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
            ParticleWGAN TrainModel = new ParticleWGAN(new int2(Dim), 64, new[] { 1 }, BatchSize);
            TrainModel.Load(WorkingDirectory + "ParticleWGAN_20210111_210604.pt");
            WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

            GPU.SetDevice(ProcessingDevice);

            Image[] ImagesReal = null;
            Image[] ImagesFake = null;
            Image[] ImagesCTF = null;

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
                Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                Image TImagesFake = new Image(new int3(Dim, Dim, BatchSize));
                Image TImagesCTF = new Image(new int3(Dim, Dim, BatchSize), true);

                Image CTFCoords = CTF.GetCTFCoords(Dim, DimRaw);

                int PlanForw = 0, PlanBack = 0;
                if (DimRaw != Dim)
                {
                    PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(Dim, Dim, 1), (uint)BatchSize);
                }

                List<Star> AllTables = new List<Star>();

                foreach (var stackName in UniqueStackNames)
                {
                    int[] StackIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1).Where(i => AllParticleAddresses[i].name == stackName).ToArray();
                    int NParticles = StackIDs.Length;

                    Image ParticleStack = new Image(new int3(Dim - 16, Dim - 16, NParticles));

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);

                        // If this thread succeeded at pushing its previously loaded batch to processing
                        int[] SubsetIDs = Helper.Combine(StackIDs.Skip(batchStart).Take(CurBatch).ToArray(), Helper.ArrayOfConstant(StackIDs.Last(), BatchSize - CurBatch));

                        float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

                        for (int b = 0; b < BatchSize; b++)
                        {
                            int id = SubsetIDs[b];
                            IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, DirectoryFake, AllParticleAddresses[id].name),
                                                    new int2(1),
                                                    0,
                                                    typeof(float),
                                                    new[] { AllParticleAddresses[id].id },
                                                    null,
                                                    new[] { LoadStackData[b] });
                        }

                        if (Dim == DimRaw)
                            GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
                                                    TImagesFake.GetDevice(Intent.Write),
                                                    LoadStack.ElementsReal);
                        else
                            GPU.Scale(LoadStack.GetDevice(Intent.Read),
                                        TImagesFake.GetDevice(Intent.Write),
                                        LoadStack.Dims.Slice(),
                                        TImagesFake.Dims.Slice(),
                                        (uint)BatchSize,
                                        PlanForw,
                                        PlanBack,
                                        IntPtr.Zero,
                                        IntPtr.Zero);

                        GPU.CheckGPUExceptions();

                        GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                        CTFCoords.GetDevice(Intent.Read),
                                        IntPtr.Zero,
                                        (uint)CTFCoords.ElementsSliceComplex,
                                        Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                        false,
                                        (uint)BatchSize);

                        Image Predicted = null;

                        //TImagesFake.WriteMRC(Path.Combine(WorkingDirectory, $"ref.mrc"), true);

                        TrainModel.Predict(TImagesFake, TImagesCTF, out Predicted);

                        //Predicted.WriteMRC(Path.Combine(WorkingDirectory, "ganned", $"{NDone++}_crap.mrc"), true);

                        GPU.Pad(Predicted.GetDevice(Intent.Read),
                                ParticleStack.GetDeviceSlice(batchStart, Intent.Write),
                                new int3(Dim, Dim, 1),
                                ParticleStack.Dims.Slice(),
                                (uint)CurBatch);
                    }

                    ParticleStack.SubtractMeanGrid(new int2(1));
                    GPU.NormParticles(ParticleStack.GetDevice(Intent.Read),
                                      ParticleStack.GetDevice(Intent.Write),
                                      ParticleStack.Dims.Slice(),
                                      (uint)(Dim / 4),
                                      false,
                                      (uint)NParticles);

                    ParticleStack.WriteMRC(Path.Combine(WorkingDirectory, "ganparticles_std", stackName), true);
                    ParticleStack.Dispose();

                    Star StackTable = TableIn.CreateSubset(StackIDs);
                    for (int r = 0; r < NParticles; r++)
                    {
                        StackTable.SetRowValue(r, "rlnImageName", $"{(r + 1):D5}@{stackName}");
                    }

                    StackTable.SetColumn("rlnDetectorPixelSize", Helper.ArrayOfConstant("3.0", NParticles));
                    AllTables.Add(StackTable);
                }

                new Star(AllTables.ToArray()).Save(Path.Combine(WorkingDirectory, "ganparticles_std", "particles.star"));
            }
        }
    }
}
